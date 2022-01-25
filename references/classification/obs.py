import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG
from asdfghjkl.utils import add_value_to_diagonal, cholesky_inv


def to_vector(parameters):
    return nn.utils.parameters_to_vector(parameters)


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


class Scope(object):
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.n_weight = torch.numel(self.weight)
        self.n_bias = torch.numel(self.bias) if self.has_bias else 0
        self.n = self.n_weight + self.n_bias
        self.n_zero = 0
        self.init_mask()
        self.ifisher = None
        self.ifisher_diag = None
        self.pruned = set()

    @property
    def weight(self):
        return self.module.weight

    @weight.setter
    def weight(self, w):
        self.module.weight.data = w.reshape(self.module.weight.shape)

    def weight_iadd(self, w):
        self.module.weight.data += w.reshape(self.module.weight.shape)

    @property
    def has_bias(self):
        return self.module.bias is not None

    @property
    def bias(self):
        return self.module.bias

    @bias.setter
    def bias(self, b):
        self.module.bias.data = b.reshape(self.module.bias.shape)

    def bias_iadd(self, b):
        self.module.bias.data += b.reshape(self.module.bias.shape)

    @property
    def parameters(self):
        if self.has_bias:
            return to_vector([self.weight, self.bias])
        else:
            return to_vector([self.weight])

    @parameters.setter
    def parameters(self, p):
        self.weight = p[:self.n_weight]
        if self.has_bias:
            self.bias = p[self.n_weight:]

    def parameters_iadd(self, p):
        self.weight_iadd(p[:self.n_weight])
        if self.has_bias:
            self.bias_iadd(p[self.n_weight:])

    def init_mask(self):
        self.module.register_buffer("weight_mask",
                                    torch.ones_like(self.module.weight))
        self.module.weight.register_hook(lambda g: g * self.module.weight_mask)
        if self.has_bias:
            self.module.register_buffer("bias_mask",
                                        torch.ones_like(self.module.bias))
            self.module.bias.register_hook(lambda g: g * self.module.bias_mask)

    @property
    def weight_mask(self):
        return self.module.weight_mask

    @property
    def bias_mask(self):
        return self.module.bias_mask

    @property
    def mask(self):
        if self.has_bias:
            return to_vector([self.weight_mask, self.bias_mask])
        else:
            return to_vector([self.weight_mask])

    @mask.setter
    def mask(self, mask):
        self.module.weight_mask = mask[:self.n_weight].reshape(
            self.module.weight.shape)
        if self.has_bias:
            self.module.bias_mask = mask[self.n_weight:].reshape(
                self.module.bias.shape)

    @property
    def grad(self):
        if self.has_bias:
            return to_vector([self.weight.grad, self.bias.grad])
        else:
            return to_vector([self.weight.grad])

    def score(self, diag_fisher_inv):
        scores = self.parameters.pow(2) / diag_fisher_inv
        return scores.masked_fill(self.mask == 0.0, float("inf"))

    def prune(self, indices, check=False):
        if check:
            assert torch.all(indices < self.n)
            assert torch.all(self.mask[indices] == 1.0)
            assert len(self.pruned & set(indices.tolist())) == 0
            self.pruned |= set(indices.tolist())

        with torch.no_grad():
            p = self.parameters
            p[indices] = 0.0
            self.parameters = p
            mask = self.mask
            mask[indices] = 0.0
            self.mask = mask
        self.n_zero += indices.shape[0]

        if check:
            assert torch.all(self.parameters[indices] == 0.0)
            assert torch.all(self.mask[indices] == 0.0)
            masked = self.parameters.masked_select(self.mask < 1)
            zeros = torch.zeros(self.n_zero).to(masked.device)
            torch.testing.assert_close(masked, zeros)

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def __str__(self):
        return "\n".join([
            #"=" * 80,
            f"{self.name} sparsity: {self.n_zero}/({self.n_weight}+{self.n_bias})={self.sparsity}",
            #"parameters:", f"{self.parameters}", "mask:", f"{self.mask}"
        ])


class OptimalBrainSurgeon(object):
    def __init__(self, model, scopes, fisher_type, world_size=0):
        self.model = model
        self.scopes = scopes
        offset = 0
        for i, s in enumerate(self.scopes):
            s.l = offset
            s.r = s.l + s.n
            s.index = i
            offset = s.r
        self.n = sum([s.n for s in self.scopes])
        self.n_zero = 0
        self.fisher_type = fisher_type
        self.ifisher = None
        self.ifisher_diag = None
        self.device = next(self.model.parameters()).device
        self.world_size = world_size

    def _get_scope_by_indice(self, i):
        for s in self.scopes:
            if s.l <= i and i < s.r:
                return s
        assert False

    @property
    def parameters(self):
        return to_vector([s.parameters for s in self.scopes])

    @parameters.setter
    def parameters(self, p):
        if isinstance(p, list):
            for s, v in zip(self.scopes, p):
                if v is not None:
                    s.parameters = v
        else:
            for s in self.scopes:
                s.parameters = p[s.l:s.r]

    def parameters_iadd(self, p):
        if isinstance(p, list):
            for s, v in zip(self.scopes, p):
                if v is not None:
                    s.parameters += v
        else:
            for s in self.scopes:
                s.parameters += p[s.l:s.r]

    @property
    def mask(self):
        return to_vector([s.mask for s in self.scopes])

    @property
    def grad(self):
        return to_vector([s.grad for s in self.scopes])

    def _gen_samples(self, loader, n_samples):
        for inputs, targets in loader:
            if n_samples != -1 and len(inputs) > n_samples:
                inputs = inputs[:n_samples]
                targets = targets[:n_samples]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            yield inputs, targets
            if n_samples != -1:
                n_samples -= len(inputs)
                if n_samples <= 0:
                    break

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        for i, (inputs,
                targets) in enumerate(self._gen_samples(loader, n_samples)):
            fisher_for_cross_entropy(self.model,
                                     fisher_type=self.fisher_type,
                                     fisher_shapes=[self.fisher_shape],
                                     inputs=inputs,
                                     targets=targets,
                                     accumulate=True if i > 0 else False,
                                     data_average=False,
                                     scale=1 / n_samples)
            print("+", end="")
        print()

    def prune(self,
              loader,
              sparsity,
              damping=1e-3,
              n_recompute=1,
              n_recompute_samples=4096,
              fisher_gb=10,
              cb=lambda: None,
              check=False):
        init_n_zero = self.n_zero
        target_n_zero = int(self.n * sparsity)

        if n_recompute == -1:
            n_recompute = target_n_zero - init_n_zero
            schedule = lambda i: self.n_zero + 1
        else:
            schedule = lambda i: polynomial_schedule(
                init_n_zero, target_n_zero, i, n_recompute)

        for i in range(1, n_recompute + 1):
            self._calc_fisher(loader, n_recompute_samples, damping)
            with torch.no_grad():
                n_pruned = int(schedule(i)) - self.n_zero
                scores = self._get_scores()
                _, indices = torch.sort(scores)
                indices = indices[:n_pruned]
                self.add_pruning_direction(indices, fisher_gb)
                for s in self.scopes:
                    j = torch.where(indices < s.r, indices, 0)
                    j = torch.where(s.l <= indices, j, 0)
                    j = indices[j.nonzero()].view(-1)
                    s.prune(j - s.l, check=check)
                    self.n_zero += len(j)
            cb()
        #print(torch.cuda.memory_allocated())

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def __str__(self):
        info = [str(s) for s in self.scopes]
        info += [f"Total sparsity: {self.n_zero}/{self.n}={self.sparsity}"]
        return "\n".join(info)


class FullOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type, world_size):
        super().__init__(model, scopes, fisher_type, world_size)
        self.fisher_shape = SHAPE_FULL

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super()._calc_fisher(loader, n_samples, damping)
        fisher = getattr(self.model, self.fisher_type)
        mask = self.mask
        fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
        if self.world_size > 1:
            dist.all_reduce(fisher.data, op=dist.ReduceOp.SUM)
            fisher.data /= self.world_size
        fisher.update_inv(damping)
        self.ifisher = fisher.inv
        self.ifisher_diag = fisher.inv.diag()

    def _get_scores(self):
        scores = self.parameters.pow(2) / self.ifisher_diag
        scores = scores.masked_fill(self.mask == 0.0, float("inf"))
        return scores

    def _pruning_direction(self, i):
        pi = self.parameters[i]
        d = (-pi / self.ifisher_diag[i] * self.ifisher[:, i]).sum(1)
        d *= self.mask
        d[i] = -pi
        return d

    def add_pruning_direction(self, indices, fisher_gb):
        stride = fisher_gb * 1024 * 1024 * 1024 // self.n // 4
        d = torch.zeros(self.n).to(self.device)
        for k in range(0, len(indices), stride):
            d += self._pruning_direction(indices[k:k + stride])
        self.parameters_iadd(d)


class LayerOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type, world_size):
        super().__init__(model, scopes, fisher_type, world_size)
        self.fisher_shape = SHAPE_LAYER_WISE
        self.normalize = False

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super()._calc_fisher(loader, n_samples, damping)
        for s in self.scopes:
            fisher = getattr(s.module, self.fisher_type)
            mask = s.mask
            fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
            if self.world_size > 1:
                dist.all_reduce(fisher.data, op=dist.ReduceOp.SUM)
                fisher.data /= self.world_size
            fisher.update_inv(damping)
            s.ifisher = fisher.inv
            s.ifisher_diag = fisher.inv.diag()

    def _get_scores(self):
        flatten_scores = []
        for s in self.scopes:
            scores = s.parameters.pow(2) / s.ifisher_diag
            if self.normalize:
                scores = scores.masked_fill(s.mask == 0.0, 0.0)
                scores /= torch.sum(scores)
            scores = scores.masked_fill(s.mask == 0.0, float("inf"))
            flatten_scores.append(scores)
        return to_vector(flatten_scores)

    def _pruning_direction(self, s, i):
        j = i - s.l
        pj = s.parameters[j]
        d = (-pj / s.ifisher_diag[j] * s.ifisher[:, j]).sum(1) * s.mask
        d[j] = -pj
        return d

    def add_pruning_direction(self, indices, fisher_gb):
        for s in self.scopes:
            j = torch.where(indices < s.r, indices, 0)
            j = torch.where(s.l <= indices, j, 0)
            j = indices[j.nonzero()].view(-1)
            d = torch.zeros(s.n).to(self.device)
            stride = fisher_gb * 1024 * 1024 * 1024 // s.n // 4
            for k in range(0, len(j), stride):
                d += self._pruning_direction(s, indices[k:k + stride])
            s.parameters_iadd(d)


class KronOBS(LayerOBS):
    def __init__(self, model, scopes, fisher_type, world_size):
        super().__init__(model, scopes, fisher_type, world_size)
        self.fisher_shape = SHAPE_KRON
        self.normalize = False
        self.fast_inv = False

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super(LayerOBS, self)._calc_fisher(loader, n_samples, damping)
        for s in self.scopes:
            fisher = getattr(s.module, self.fisher_type).kron
            if self.world_size > 1:
                dist.all_reduce(fisher.A, op=dist.ReduceOp.SUM)
                fisher.A /= self.world_size
                dist.all_reduce(fisher.B, op=dist.ReduceOp.SUM)
                fisher.B /= self.world_size
            if self.fast_inv:
                # This method is wrong
                # kron(inverse(A), inverse(B))*mask
                fisher.update_inv(damping)
                s.ifisher_diag = torch.kron(fisher.A_inv.diag(),
                                            fisher.B_inv.diag()) * s.mask
            else:
                # (kron(A,B)*mask).inverse()
                f = torch.kron(fisher.A, fisher.B)
                mask = s.mask
                f.mul_(mask.reshape([1, -1])).mul_(mask.reshape([-1, 1]))
                f.as_strided([f.shape[0]], [f.shape[0] + 1]).add_(damping)
                f = torch.linalg.cholesky(f)
                f = torch.cholesky_inverse(f)
                s.ifisher = f
                s.ifisher_diag = f.diag()

    def _pruning_direction(self, s, i):
        j = i - s.l
        pj = s.parameters[j]
        if self.fast_inv:
            fisher = getattr(s.module, self.fisher_type).kron
            a = fisher.A_inv[:, j // fisher.B.shape[1]]
            b = fisher.B_inv[:, j % fisher.B.shape[1]]
            vec = torch.einsum("aj,bj->jab", a, b).view(j.shape[0],
                                                        -1).transpose(0, 1)
        else:
            vec = s.ifisher[:, j]
        d = vec.mul_(-pj / s.ifisher_diag[j]).sum(1) * s.mask
        d[j] = -pj
        return d


class NoneOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type, world_size):
        super().__init__(model, scopes, fisher_type, world_size)
        self.fisher_shape = "none"

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        return None

    def _get_scores(self):
        return torch.abs(self.parameters).masked_fill(self.mask == 0.0,
                                                      float("inf"))

    def _pruning_direction(self, i):
        return None

    def add_pruning_direction(self, indices, fisher_gb):
        pass


class IFVP():
    def __init__(self, grads, damping, check=False):
        self.n, self.d = grads.shape[0], grads.shape[1]
        self.damping = 1. / damping

        self.v = torch.zeros((self.n, self.d), device=grads.device)
        self.q = torch.zeros(self.n, device=grads.device)

        self.v[0, :] = self.damping * grads[0, :]
        self.q[0] = self.n + torch.inner(self.v[0, :], grads[0, :])
        for i, g in enumerate(grads[1:]):
            i += 1
            self.v[i, :] = self.damping * g - (self.v[:i, :] @ g /
                                               self.q[:i]).T @ self.v[:i]
            self.q[i] = self.n + torch.inner(self.v[i, :], g)

        self.check = check
        if self.check:
            self.iF = self.damping * torch.eye(self.d)
            for i, g in enumerate(grads):
                ifg = self.iF @ g
                print(ifg)
                print(self.v[i, :])
                torch.testing.assert_close(ifg, self.v[i, :])
                print(self.n + g.T @ ifg)
                print(self.q[i])
                torch.testing.assert_close(self.n + g.T @ ifg, self.q[i])
                self.iF -= torch.outer(ifg, ifg) / (self.n + g.T @ ifg)
            print("ifsher")
            print(self.iF)

    def __call__(self, x):
        assert x.ndim == 2
        res = self.damping * x - self.v.T @ (self.v @ x /
                                             self.q.reshape(-1, 1))
        if self.check:
            print(res)
            print(self.iF @ x)
            torch.testing.assert_close(res, self.iF @ x, rtol=1e-4, atol=1e-4)
        return res

    def diag(self):
        res = self.damping * torch.ones(self.d, device=self.v.device)
        for i in range(self.n):
            res -= (self.v[i, :]**2) / self.q[i]
        if self.check:
            print(res)
            print(self.iF.diag())
            torch.testing.assert_close(res, self.iF.diag())
        return res

    def column(self, i):
        assert i.ndim == 1
        res = self(F.one_hot(i, self.d).type(self.v.dtype).transpose(1, 0))
        if self.check:
            torch.testing.assert_close(res, self.iF[:, i])
        return res


class FullWoodOBS(FullOBS):
    def __init__(self, model, scopes, fisher_type, world_size):
        assert fisher_type == FISHER_EMP
        super().__init__(model, scopes, fisher_type, world_size)
        self.fisher_shape = "full_wood"
        self.ifvp = None

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        grads = []
        for inputs, targets in self._gen_samples(loader, n_samples):
            nn.CrossEntropyLoss()(self.model(inputs), targets).backward()
            grads.append(self.grad)
        grads = torch.vstack(grads)
        self.ifvp = IFVP(grads * mask, damping)
        self.ifisher_diag = self.ifvp.diag()

    def _pruning_direction(self, i):
        pi = self.parameters[i]
        d = (-pi / self.ifisher_diag[i] * self.ifvp.column(i)).sum(1)
        d *= self.mask
        d[i] = -pi
        return d


class BlockWoodOBS(FullWoodOBS):
    def __init__(self, model, scopes, fisher_type, world_size):
        assert fisher_type == FISHER_EMP
        super().__init__(model, scopes, fisher_type, world_size)
        self.fisher_shape = "block_wood"
        self.block_size = None

    def set_block_size(self, block_size):
        self.block_size = block_size

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        grads = []
        for inputs, targets in self._gen_samples(loader, n_samples):
            nn.CrossEntropyLoss()(self.model(inputs), targets).backward()
            grads.append(self.grad)
        grads = torch.vstack(grads)
        grads = torch.split(grads, self.block_size, dim=1)
        masks = torch.split(self.mask, self.block_size)
        self.ifvp = []
        for g, m in zip(grads, masks):
            self.ifvp.append(IFVP(g * m, damping))
        self.ifisher_diag = torch.hstack([x.diag() for x in self.ifvp])

    def _pruning_direction(self, ifvp, l, i):
        column = torch.zeros(self.n, len(i)).to(self.device)
        column[l:l + self.block_size, :] = ifvp.column(i - l)
        pi = self.parameters[i]
        d = (-pi / self.ifisher_diag[i] * column).sum(1) * self.mask
        d[i] = -pi
        return d

    def add_pruning_direction(self, indices, fisher_gb):
        stride = fisher_gb * 1024 * 1024 * 1024 // self.block_size // 4
        d = torch.zeros(self.n).to(self.device)
        for i in range(0, self.n, self.block_size):
            j = torch.where(indices < i + self.block_size, indices, 0)
            j = torch.where(i <= indices, j, 0)
            j = indices[j.nonzero()].view(-1)
            ifvp = self.ifvp[i // self.block_size]
            for k in range(0, len(j), stride):
                d += self._pruning_direction(ifvp, i, j[k:k + stride])
        self.parameters_iadd(d)
