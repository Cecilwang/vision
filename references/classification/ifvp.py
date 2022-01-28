import torch
import torch.nn.functional as F
from torch.testing import assert_close

EPS = 1e-3


def batch_inner(x, y):
    assert x.ndim == 2
    assert x.shape == y.shape
    return torch.einsum('bd,bd->b', x, y)


class IFVP():
    def __init__(self, grads, block_size, damping, check=False):
        assert grads.ndim == 2

        self.real_d = grads.shape[1]
        self.pad = 0
        if block_size == -1:
            block_size = grads.shape[1]
        if grads.shape[1] % block_size != 0:
            self.pad = block_size - (grads.shape[1] % block_size)
            grads = F.pad(grads, (0, self.pad), "constant", 0)
        grads = grads.reshape(grads.shape[0], -1, block_size).transpose(0, 1)

        self.b, self.n, self.d = grads.shape
        self.damping = 1. / damping

        self.v = torch.zeros((self.b, self.n, self.d), device=grads.device)
        self.q = torch.zeros((self.b, self.n), device=grads.device)

        self.v[:, 0, :] = self.damping * grads[:, 0, :]
        self.q[:, 0] = self.n + batch_inner(self.v[:, 0, :], grads[:, 0, :])
        for i in range(1, self.n):
            g = grads[:, i, :]  # bxd
            v = torch.einsum('bkd,bd->bk', self.v[:, :i, :], g)  # bxk
            v /= self.q[:, :i]  # bxk
            v = torch.einsum('bk,bkd->bd', v, self.v[:, :i, :])  # bxd
            self.v[:, i, :] = self.damping * g - v
            self.q[:, i] = self.n + batch_inner(self.v[:, i, :], g)

        self.check = check
        if self.check:
            grads = grads.transpose(0, 1).reshape(self.n, -1)[:, :self.real_d]
            grads = torch.split(grads, block_size, -1)
            iFs = []
            for j, grad in enumerate(grads):
                d = grad.shape[-1]
                iF = self.damping * torch.eye(d)
                for i, g in enumerate(grad):
                    ifg = iF @ g
                    print(ifg)
                    print(self.v[j, i, :d])
                    assert_close(ifg, self.v[j, i, :d], rtol=EPS, atol=EPS)
                    q = self.n + g.T @ ifg
                    print(q)
                    print(self.q[j, i])
                    assert_close(q, self.q[j, i], rtol=EPS, atol=EPS)
                    iF -= torch.outer(ifg, ifg) / (self.n + g.T @ ifg)
                iFs.append(iF)
            self.iF = torch.block_diag(*iFs)
            print("ifsher")
            print(self.iF)

    def __call__(self, x):
        assert x.ndim == 2
        assert x.shape[0] == self.real_d
        if self.pad != 0:
            x = F.pad(x, (0, 0, 0, self.pad), "constant", 0)
        x = x.reshape(self.b, self.d, -1)
        res = torch.einsum('bnd,bdm->bnm', self.v, x)
        res /= self.q.unsqueeze(-1)
        res = torch.einsum('bnd,bnm->bdm', self.v, res)
        res = self.damping * x - res
        res = res.reshape(self.b * self.d, -1)[:self.real_d, :]
        x = x.reshape(self.b * self.d, -1)[:self.real_d, :]
        if self.check:
            gt = self.iF @ x
            print(res)
            print(gt)
            assert_close(res, gt, rtol=EPS, atol=EPS)
        return res

    def diag(self):
        res = self.damping * torch.ones((self.b, self.d), device=self.v.device)
        for i in range(self.n):
            res -= (self.v[:, i, :]**2) / self.q[:, i].unsqueeze(-1)
        res = res.reshape(-1)[:self.real_d]
        if self.check:
            gt = self.iF.diag()
            print(res)
            print(gt)
            assert_close(res, gt, rtol=EPS, atol=EPS)
        return res

    def column(self, i):
        assert i.ndim == 1
        x = F.one_hot(i, self.real_d).type(self.v.dtype).transpose(1, 0)
        res = self(x)
        if self.check:
            assert_close(res, self.iF[:, i], rtol=EPS, atol=EPS)
        return res
