import torch
import torch.fx as fx
import json
from torch_registry import lower_graph

class Tiny(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(out_f, in_f))
        self.b = torch.nn.Parameter(torch.randn(out_f))
    def forward(self, x):

        y = torch.ops.aten.linear(x, self.w, self.b)
        a = torch.ops.aten.narrow(y, 1, 0, 4)
        z = torch.zeros(15, 4)
        a.copy_(z)  # becomes SLICE_UPDATE
        y = y + self.b
        return z

m = Tiny(8, 16).eval()
ex_inputs = (torch.randn(15, 8),)

ep = torch.export.export(m, ex_inputs)

print(ep)

# Real build
P, gm2, manifest = lower_graph(ep, draft=False, name="tiny")
print(json.dumps(manifest, indent=2))
