from sympy import Symbol, Dummy, sympify, exp
from sympy.abc import mu, x, y, w
from sympy.stats import density, Normal
from sympy.solvers import solve
from sympy.holonomic.holonomic import expr_to_holonomic

sigma = Symbol('sigma', positive=True)
want_dens = w * density(Normal(Dummy(), mu, sigma))(y)
have_dens = exp(-x**2-y**2+x*y)
want_hol = expr_to_holonomic(want_dens, y)
have_hol = expr_to_holonomic(have_dens, y)
print(want_hol)
print(have_hol)

def annihilator_coeff(hol, i):
    return hol.annihilator.listofpoly[i] if i <= hol.annihilator.order else 0

equations = set(h-w for (h,w) in zip(have_hol.y0, want_hol.y0))
order = max(want_hol.annihilator.order, have_hol.annihilator.order)
have_top_coeff = annihilator_coeff(have_hol, order)
want_top_coeff = annihilator_coeff(want_hol, order)
for i in range(0, order):
    equations.update(c.as_expr() if hasattr(c,'as_expr') else sympify(c)
        for c in (annihilator_coeff(have_hol, i) * want_top_coeff -
                  annihilator_coeff(want_hol, i) * have_top_coeff).all_coeffs())
print(equations)
print(solve(equations, [w, mu, sigma]))
