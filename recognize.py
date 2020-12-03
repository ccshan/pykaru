from sympy.holonomic.holonomic import expr_to_holonomic
from sympy import symbols, sympify, exp, sqrt, pi

def continued_fraction(f, g):
    # continued_fraction(f, g) expresses f/g as a continued fraction like
    # quotients[0] + 1/(quotients[1] + 1/(quotients[2] + 1/(quotients[3])))
    quotients = []
    while g:
        q = f.quo(g)
        quotients.append([c.as_expr() if hasattr(c,'as_expr') else sympify(c) for c in q.all_coeffs()] if q else [])
        (f, g) = (g, f.rem(g))
    return quotients

x = symbols('x')
y = symbols('y')
dens = exp(-x**2-y**2+x*y)
hol = expr_to_holonomic(dens, y)
print(hol)
if hol.annihilator.order == 1:
    cf = continued_fraction(*hol.annihilator.listofpoly)
    if [len(q) for q in cf] == [2]:
        [[b1,b0]] = cf
        mean = -b0/b1
        precision = b1
        weight = dens.subs(y, mean) * sqrt(2 * pi / precision)
        print(mean, precision, weight)
