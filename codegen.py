from __future__ import print_function

# Solve some symbolic problem using SymPy
from sympy.abc import a,b,c,x
from sympy.solvers import solve
roots = solve(a*x**2 + b*x + c, x)
print('roots =', roots)

# Reuse the solution for numeric computation
from sympy.utilities.lambdify import lambdify
f = lambdify((a,b,c), roots)
print('f(1,-1,-1) =', f(1,-1,-1))
print('f(2, 0,-8) =', f(2, 0,-8))

# Reuse the solution for differentiable tensorized computation
import torch
f = lambdify((a,b,c), roots, torch)
a0 = torch.tensor([1,2],dtype=torch.float)
b0 = torch.tensor([-1,0],dtype=torch.float)
c0 = torch.tensor([-1,-8],dtype=torch.float,requires_grad=True)
roots0 = f(a0,b0,c0)
print('roots0 =', roots0)
out = torch.sum(sum(map(torch.abs, roots0)))
print('out =', out)
out.backward()
print('c0.grad =', c0.grad)
# https://github.com/pytorch/pytorch/issues/38230
