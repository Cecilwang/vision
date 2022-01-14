import math

import torch
from torch import nn
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
        self.module.weight.register_hook(
            lambda grad: grad * getattr(self.module, "weight_mask"))
        if self.has_bias:
            self.module.register_buffer("bias_mask",
                                        torch.ones_like(self.module.bias))
            self.module.bias.register_hook(
                lambda grad: grad * getattr(self.module, "bias_mask"))

    @property
    def weight_mask(self):
        return getattr(self.module, "weight_mask")

    @property
    def bias_mask(self):
        return getattr(self.module, "bias_mask")

    @property
    def mask(self):
        if self.has_bias:
            return to_vector([self.weight_mask, self.bias_mask])
        else:
            return to_vector([self.weight_mask])

    @property
    def grad(self):
        if self.has_bias:
            return to_vector([self.weight.grad, self.bias.grad])
        else:
            return to_vector([self.weight.grad])

    def score(self, diag_fisher_inv):
        scores = self.parameters.pow(2) / diag_fisher_inv
        return scores.masked_fill(self.mask == 0.0, float("inf"))

    def prune(self, i, d=None, check=False):
        assert i not in self.pruned
        assert i < self.n
        assert self.mask[i] == 1.0
        self.pruned.add(i)

        with torch.no_grad():
            if d is not None:
                self.parameters_iadd(d)
            if i < self.n_weight:
                self.weight.view(-1)[i] = 0.0
                self.weight_mask.view(-1)[i] = 0.0
            else:
                self.bias.view(-1)[i - self.n_weight] = 0.0
                self.bias_mask.view(-1)[i - self.n_weight] = 0.0
        self.n_zero += 1

        if check:
            assert self.parameters[i] == 0.0
            assert self.mask[i] == 0.0
            self.check()

    def check(self):
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
    def __init__(self, model, scopes, fisher_type, world_size=0, check=False):
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
        self.pruned = set()
        self._device = next(self.model.parameters()).device
        self.world_size = world_size
        self._check = check

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
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
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

    def _prune_one(self, i):
        assert i not in self.pruned
        self.pruned.add(i)

        scope = self._get_scope_by_indice(i)
        d = self._pruning_direction(i)
        if d is None:
            scope.prune(i - scope.l, check=self._check, log=False)
        elif len(d) == scope.n:
            scope.prune(i - scope.l, d, check=self._check, log=False)
        elif len(d) == self.n:
            self.parameters_iadd(d)
            scope.prune(i - scope.l, check=self._check, log=False)
        else:
            assert False

        self.n_zero += 1
        self.check()

    def prune(self,
              loader,
              sparsity,
              damping=1e-3,
              n_recompute=1,
              n_recompute_samples=4096,
              cb=lambda : None):
        init_n_zero = self.n_zero
        target_n_zero = int(self.n * sparsity)

        if n_recompute == -1:
            n_recompute = target_n_zero - init_n_zero
            schedule = lambda i: self.n_zero + 1
        else:
            schedule = lambda i: polynomial_schedule(
                init_n_zero, target_n_zero, i, n_recompute)

        for i in range(1, n_recompute + 1):
            # We are accumulating fisher across recompute iteration
            # Should we clear fisher at beginning of the iteration?
            self._calc_fisher(loader, n_recompute_samples, damping)
            with torch.no_grad():
                n_pruned = int(schedule(i)) - self.n_zero
                scores = self._get_scores()
                _, indices = torch.sort(scores)
                indices = indices[:n_pruned]
                for j in indices:
                    self._prune_one(j.item())
                    cb()

    def check(self):
        if self._check:
            mask = torch.ones(self.n)
            mask[list(self.pruned)] = 0.0
            torch.testing.assert_close(mask.to(self.mask.device), self.mask)

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def __str__(self):
        info = [str(s) for s in self.scopes]
        info += [f"Total sparsity: {self.n_zero}/{self.n}={self.sparsity}"]
        return "\n".join(info)


class FullOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type,world_size, check):
        super().__init__(model, scopes, fisher_type,world_size, check=check)
        self.fisher_shape = SHAPE_FULL

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super()._calc_fisher(loader, n_samples, damping)
        fisher = getattr(self.model, self.fisher_type)
        mask = self.mask
        fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
        if self.world_size>1:
            dist.all_reduce(fisher.data, op=dist.ReduceOp.SUM)
            fisher.data /= self.world_size
        fisher.update_inv(damping)
        self.ifisher = fisher.inv
        self.ifisher_diag = torch.diagonal(self.ifisher)

    def _get_scores(self):
        scores = self.parameters.pow(2) / self.ifisher_diag
        scores = scores.masked_fill(self.mask == 0.0, float("inf"))
        return scores

    def _pruning_direction(self, i):
        s = self._get_scope_by_indice(i)
        pi = s.parameters[i - s.l]
        return -pi * self.ifisher[:, i] / self.ifisher_diag[i] * self.mask


class LayerOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type,world_size, check):
        super().__init__(model, scopes, fisher_type,world_size, check=check)
        self.fisher_shape = SHAPE_LAYER_WISE
        self.normalize = False

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super()._calc_fisher(loader, n_samples, damping)
        for s in self.scopes:
            fisher = getattr(s.module, self.fisher_type)
            mask = s.mask
            fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
            fisher.update_inv(damping)
            s.ifisher = fisher.inv
            s.ifisher_diag = torch.diagonal(s.ifisher)

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

    def _pruning_direction(self, i):
        s = self._get_scope_by_indice(i)
        pi = s.parameters[i - s.l]
        return -pi * s.ifisher[:, i - s.l] / s.ifisher_diag[i - s.l] * s.mask


class KronOBS(LayerOBS):
    def __init__(self, model, scopes, fisher_type,world_size, check):
        super().__init__(model, scopes, fisher_type,world_size, check=check)
        self.fisher_shape = SHAPE_KRON
        self.normalize = False
        self.fast_inv = False

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super(LayerOBS, self)._calc_fisher(loader, n_samples, damping)
        for s in self.scopes:
            fisher = getattr(s.module, self.fisher_type).kron
            if self.fast_inv:
                # This method is wrong
                fisher.update_inv(damping)
                fisher.inv = torch.kron(fisher.A_inv, fisher.B_inv)
                mask = s.mask
                fisher.inv *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
            else:
                f = torch.kron(fisher.A, fisher.B)
                mask = s.mask
                f *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
                fisher.inv = cholesky_inv(add_value_to_diagonal(f, damping))
            s.ifisher = fisher.inv
            s.ifisher_diag = torch.diagonal(s.ifisher)


class NoneOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type,world_size, check):
        super().__init__(model, scopes, fisher_type,world_size, check=check)
        self.fisher_shape = "none"

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        return None

    def _get_scores(self):
        return torch.abs(self.parameters).masked_fill(self.mask == 0.0,
                                                      float("inf"))

    def _pruning_direction(self, i):
        return None


class FullWoodOBS(FullOBS):
    def __init__(self, model, scopes, fisher_type,world_size, check):
        assert fisher_type == FISHER_EMP
        super().__init__(model, scopes, FISHER_EMP,world_size, check=check)

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        N = None
        fisher_inv = torch.eye(self.n) / (damping**2)

        for inputs, targets in self._gen_samples(loader, n_samples):
            if N is None:
                if n_samples == -1:
                    N = len(loader)
                else:
                    N = math.ceil(n_samples / len(inputs))
            nn.CrossEntropyLoss()(self.model(inputs), targets).backward()
            with torch.no_grad():
                g = self.grad * self.mask
                fg = fisher_inv @ g
                fisher_inv -= torch.outer(fg, fg) / (N + g.T @ fg)

        self.ifisher = fisher_inv
        self.ifisher_diag = torch.diagonal(self.ifisher)
