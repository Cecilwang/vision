import torch

from ifvp import IFVP


def test(ifvp):
    print("test ifvp")
    #ifvp(torch.ones(d)) # failed
    ifvp(torch.rand(d, m))
    print("test diag")
    ifvp.diag()
    print("test column")
    #ifvp.column(torch.tensor(0)) # failed
    ifvp.column(torch.tensor([0]))
    ifvp.column(torch.tensor([0, 4]))
    print("test accumulate column")
    ifvp.accumulate_column(torch.tensor([4]))
    ifvp.accumulate_column(torch.tensor([0, 4]))
    ifvp.accumulate_column(torch.tensor([0, 1, 2, 3, 4, 5]))


m = 3
d = 6
grads = torch.rand((m, d))

print("--------------------------------")
test(IFVP(grads, -1, 1e-3, check=True))
print("--------------------------------")
test(IFVP(grads, 2, 1e-3, check=True))
print("--------------------------------")
test(IFVP(grads, 3, 1e-3, check=True))
print("--------------------------------")
test(IFVP(grads, 4, 1e-3, check=True))
print("--------------------------------")
test(IFVP(grads, 5, 1e-3, check=True))
