import torch

from obs import IFVP

m = 3
d = 5
grads = torch.rand((m, d))
mask = torch.randint(0, 2, (d, ))

ifvp = IFVP(grads * mask, 1e-3, check=True)

print("test ifvp")
#ifvp(torch.ones(d)) # failed
ifvp(torch.rand(d, m))
print("test diag")
ifvp.diag()
print("test column")
#ifvp.column(torch.tensor(0)) # failed
ifvp.column(torch.tensor([0]))
ifvp.column(torch.tensor([0, 1]))
