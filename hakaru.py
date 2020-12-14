from sympy import sympify, Symbol, Function, Expr, Lambda, Dummy, IndexedBase, Idx, Mul, Add, S, simplify, expand, integrate, Sum, separatevars, exp, pi, sqrt, oo
from sympy.stats import density, Normal

class Ret(Function('Ret', nargs=1, commutative=False)):

    def __new__(cls, value):
        return Expr.__new__(cls, sympify(value))

    @property
    def value(self):
        return self._args[0]

class Lebesgue(Function('Lebesgue', nargs=2, commutative=False)):

    def __new__(cls, lower, upper):
        return Expr.__new__(cls, sympify(lower), sympify(upper))

    @property
    def lower(self):
        return self._args[0]

    @property
    def upper(self):
        return self._args[1]

class Counting(Function('Counting', nargs=2, commutative=False)):

    def __new__(cls, lower, upper):
        return Expr.__new__(cls, sympify(lower), sympify(upper))

    @property
    def lower(self):
        return self._args[0]

    @property
    def upper(self):
        return self._args[1]

class Bind(Function('Bind', nargs=3, commutative=False)):

    def __new__(cls, before, variable, after):
        if not variable.is_symbol:
            variable = Symbol(variable)
        return Expr.__new__(cls,
                sympify(before),
                variable,
                sympify(after, {variable.name: variable}))

    @property
    def before(self):
        return self._args[0]

    def enter(self, avoid):
        # Rename self.variable in self.after to avoid free_symbols
        var = self._args[1]
        aft = self._args[2]
        if isinstance(self._args[0], Lebesgue):
            x = Dummy(var.name,
                      real = True,
                      positive = self._args[0].lower >= 0 and self._args[0].upper >= 0,
                      negative = self._args[0].lower <= 0 and self._args[0].upper <= 0)
            return (x, aft.subs(var, x))
        elif isinstance(self._args[0], Counting):
            x = Dummy(var.name,
                      integer = True,
                      positive = self._args[0].lower >= 0 and self._args[0].upper >= 0,
                      negative = self._args[0].lower <= 0 and self._args[0].upper <= 0)
            x = Idx(x, self._args[0].lower, self._args[0].upper)
            return (x, aft.subs(var, x))
        elif any(self.variable in object.free_symbols for object in avoid):
            x = Dummy(var.name)
            return (x, aft.subs(var, x))
        else:
            return (var, aft)

    @property
    def variable(self):
        return self._args[1]

    @property
    def after(self):
        return self._args[2]

    @property
    def bound_symbols(self):
        return (self.variable,)

    @property
    def free_symbols(self):
        return (self.before.free_symbols |
                (self.after.free_symbols - {self.variable}))

    def _eval_subs(self, old, new, **hints):
        if self.variable == old:
            new_before = self.before._subs(old, new, **hints)
            if new_before == None:
                return None
            else:
                return self.func(new_before, self.variable, self.after)
        else:
            return None

class WeightError(ValueError):
    def __str__(self):
        return ("Cannot parse %s as measure" % self.args[0])

def unweight(m):
    # unweight(Mul(...)) splits m into a weight and a measure proper-subterm
    # unweight(Add(...)) finds the total weight of the terms of the Add
    # unweight(0) declares the total weight to be zero
    # unweight(m) otherwise declares the total weight to be one
    if m.is_Mul:
        (c, nc, o) = Mul.flatten(m.args)
        if len(nc) != 1 or o != None:
            raise WeightError(m)
        return (Mul(*c), nc[0])
    if m.is_zero:
        return (S.Zero, m)
    if m.is_Add:
        wms = [*(unweight(term) for term in m.args)]
        w = Add(*(wm[0] for wm in wms))
        w1 = 1/w
        return (w, Add(*(Mul(w1, wm[0], wm[1]) for wm in wms)))
    return (S.One, m)

class TestUnweight:

    def test_Ret(self):
        from sympy.abc import x
        assert unweight(Ret(x+1)) == (1, Ret(x+1))

    def test_zero(self):
        assert unweight(S.Zero) == (0, S.Zero)

    def test_Add(self):
        from sympy.abc import x, y
        assert unweight(Ret(x) + Ret(y)*2) == (3, Ret(x)/3 + Ret(y)*2/3)
        assert unweight(x * Ret(1) + y * Ret(2)) == (x+y, Ret(1)*x/(x+y) + Ret(2)*y/(x+y))

    def test_Mul(self):
        from sympy.abc import x, y, z
        assert unweight(x * y * Ret(z)) == (x * y, Ret(z))
        assert unweight(x * (Ret(y) + Ret(z))) == (x, Ret(y) + Ret(z))

def parameter(g, default):
    # Returns a symbol that is guaranteed to be free in g
    # (in particular, if g==Lambda(x,...) then return x)
    if (isinstance(g, Lambda) and
        len(g.signature) == 1 and
        g.signature[0].is_symbol):
        return g.signature[0]
    else:
        return Dummy(default)

class TestParameter:

    def test_Lambda(self):
        from sympy.abc import x, y
        assert parameter(Lambda(x, y**2), 'z') == x
        assert parameter(Lambda((x,y), y**2), 'z').name == 'z'

    def test_other(self):
        from sympy.abc import f
        assert parameter(f, 'z').name == 'z'

def straighten(m, add=False):
    # straighten(m) rewrites Bind(Bind(m,x,n),y,k) to Bind(m,x,Bind(n,y,k))
    # and rewrites Bind(m,x,w*k) to w*Bind(m,x,k)
    # and optionally rewrites Bind(m1+m2,x,k) to Bind(m1,x,k)+Bind(m2,x,k)

    def _straighten(m, k):
        if m.is_Add:
            if add:
                return Add(*(_straighten(term, k) for term in m.args))
            else:
                m = Add(*(_straighten(term, Ret) for term in m.args))
                # fall through
        if m.is_zero:
            return S.Zero
        if isinstance(m, Ret):
            return k(m.value)
        if m.is_Mul:
            (w, m) = unweight(m)
            return w * _straighten(m, k)
        if isinstance(m, Bind):
            (x, after) = m.enter([k])
            (w, afterk) = unweight(_straighten(after, k))
            separation = separatevars(w, [x], dict=True)
            return separation['coeff'] * _straighten(m.before, Lambda(x, separation[x] * afterk))
        if k == Ret:
            return m
        r = parameter(k, 'r')
        return Bind(m, r, k(r))

    return _straighten(m, Ret)

class TestStraighten:

    def test_left_identity(self):
        from sympy.abc import x
        assert straighten(Bind(Ret(3), x, Lebesgue(x+1,x+2))) == Lebesgue(4,5)

    def test_Bind(self):
        from sympy.abc import x, y, z
        m = Symbol  ('m', commutative=False)
        n = Function('n', commutative=False)
        k = Function('k', commutative=False)
        assert straighten(Bind(Bind(m,x,n(x)),y,k(y))) == Bind(m,x,Bind(n(x),y,k(y)))
        assert straighten(Bind(Bind(m,x,n(x)),y,k(x,y))).dummy_eq(Bind(m,z,Bind(n(z),y,k(x,y))), z)

    def test_Mul(self):
        from sympy.abc import a, b, c, x, y
        m = Symbol  ('m', commutative=False)
        n = Function('n', commutative=False)
        k = Function('k', commutative=False)
        assert straighten(Bind(a*m, x, b*Bind(c*n(x), y, k(x,y)))) == a*b*c*Bind(m,x,Bind(n(x),y,k(x,y)))
        assert straighten(Bind(a*Bind(b*m, x, c*n(x)), y, k(y))) == a*b*c*Bind(m,x,Bind(n(x),y,k(y)))
        assert straighten(Bind(m, x, Bind(n(x), y, a*exp(b*x+c*y)*k(x,y)))) == a*Bind(m, x, exp(b*x)*Bind(n(x), y, exp(c*y)*k(x,y)))
        a = Function('a')
        b = Function('b')
        c = Function('c')
        assert straighten(Bind(a(x)*m, x, b(x)*Bind(c(x)*n(x), y, k(x,y)))) == a(x)*Bind(m,x,b(x)*c(x)*Bind(n(x),y,k(x,y)))
        assert straighten(Bind(a(x)*Bind(b(x)*m, x, c(x)*n(x)), y, k(y))) == a(x)*b(x)*Bind(m,x,c(x)*Bind(n(x),y,k(y)))

    def test_Add(self):
        from sympy.abc import x, y
        m1= Symbol  ('m1',commutative=False)
        m2= Symbol  ('m2',commutative=False)
        n1= Function('n1',commutative=False)
        n2= Function('n2',commutative=False)
        k = Function('k', commutative=False)
        assert straighten(Bind(Bind(m1 + m2, x, n1(x)), y, k(y))) == Bind(m1 + m2, x, Bind(n1(x), y, k(y)))
        assert straighten(Bind(Bind(m1, x, n1(x) + n2(x)), y, k(y))) == Bind(m1, x, Bind(n1(x) + n2(x), y, k(y)))
        assert straighten(Bind(Bind((m1 + m2)/2, x, n1(x)), y, k(y)), add=True) == (Bind(m1, x, Bind(n1(x), y, k(y))) + Bind(m2, x, Bind(n1(x), y, k(y))))/2
        assert straighten(Bind(Bind(m1, x, (n1(x) + n2(x))/2), y, k(y)), add=True) == (Bind(m1, x, Bind(n1(x), y, k(y)) + Bind(n2(x), y, k(y))))/2

    def test_zero(self):
        from sympy.abc import x, y
        m = Symbol  ('m', commutative=False)
        n1= Function('n1',commutative=False)
        n2= Function('n2',commutative=False)
        k = Function('k', commutative=False)
        assert straighten(Bind(n1, y, 0)) == 0
        assert straighten(Bind(m, x, exp(x) * Bind(n1(x), y, 0) + exp(pi) * Bind(n2(x), y, k(x,y)))) == exp(pi) * Bind(m, x, Bind(n2(x), y, k(x,y)))

def continued_fraction(f, g):
    # continued_fraction(f, g) expresses f/g as a continued fraction like
    # quotients[0] + 1/(quotients[1] + 1/(quotients[2] + 1/(quotients[3])))
    quotients = []
    while g:
        q = f.quo(g)
        quotients.append([c.as_expr() if hasattr(c,'as_expr') else sympify(c)
                          for c in q.all_coeffs()]
                         if q else [])
        (f, g) = (g, f.rem(g))
    return quotients

def recognize(m, x, w):
    # recognize(Lebesgue(-oo,oo), x, w * density(Normal(Dummy(),mean,std))(x))
    #   == (w, Normal, mean, std)
    from sympy.holonomic.holonomic import expr_to_holonomic
    if isinstance(m, Lebesgue):
        try:
            hol = expr_to_holonomic(w, x)
        except NotImplementedError:
            return None
        if hol.annihilator.order == 1:
            cf = continued_fraction(*hol.annihilator.listofpoly)
            if [len(q) for q in cf] == [2] and m.lower == -oo and m.upper == oo:
                [[b1,b0]] = cf
                mean = -b0/b1
                std = 1/sqrt(b1)
                weight = w.subs(x, mean) * sqrt(2 * pi) * std
                return (simplify(weight), Normal, mean, std)
    return None

class TestRecognize:

    def test_Normal(self):
        from sympy.abc import mu, x, y, z, w
        assert recognize(Lebesgue(-oo,oo), y, exp(-x**2-y**2+x*y)) == (sqrt(pi)*exp(-3*x**2/4), Normal, x/2, 1/sqrt(2))
        sigma = Symbol('sigma', positive=True)
        assert recognize(Lebesgue(-oo,oo), x, w * density(Normal(Dummy(),mu,sigma))(x)) == (w, Normal, mu, sigma)
        assert recognize(Lebesgue(-oo,oo), x,     density(Normal(Dummy(),mu,sigma))(x)) == (1, Normal, mu, sigma)
        sigma = 3
        assert recognize(Lebesgue(-oo,oo), x, w * density(Normal(Dummy(),mu,sigma))(x)) == (w, Normal, mu, sigma)
        assert recognize(Lebesgue(-oo,oo), x,     density(Normal(Dummy(),mu,sigma))(x)) == (1, Normal, mu, sigma)
        mu = y+z
        assert recognize(Lebesgue(-oo,oo), x, w * density(Normal(Dummy(),mu,sigma))(x)) == (w, Normal, mu, sigma)
        assert recognize(Lebesgue(-oo,oo), x, expand(w * density(Normal(Dummy(),mu,sigma))(x))) == (w, Normal, mu, sigma)
        w = exp((y+z)**2/18)
        assert recognize(Lebesgue(-oo,oo), x, expand(w * density(Normal(Dummy(),mu,sigma))(x))) == (simplify(w), Normal, mu, sigma)

    def test_other(self):
        from sympy.abc import x
        assert recognize(Lebesgue(0,1), x, density(Normal(Dummy(),0,1))(x)) == None
        assert recognize(Lebesgue(-oo,oo), x, exp(-x**4)) == None

def holonomic_integrate(integrand, range):
    # holonomic_integrate(w * density(Normal(Dummy(),mean,std)), (x,-oo,oo))
    #   == w
    if isinstance(range, tuple) and len(range) == 3:
        recognition = recognize(Lebesgue(*range[1:3]), range[0], integrand)
        if isinstance(recognition, tuple):
            return recognition[0]
    return integrate(integrand, range)

class Expect(Function):
    # Expect(m, g) computes and represents the expectation (integral) of the
    # function g with respect to the measure m

    @classmethod
    def eval(cls, m, g):
        if isinstance(m, Ret):
            return g(m.value)
        if m.is_Mul:
            (w, m) = unweight(m)
            return w * Expect(m, g)
        if m.is_zero:
            return S.Zero
        if m.is_Add:
            return Add(*(Expect(term, g) for term in m.args))
        if isinstance(m, Bind):
            (x, after) = m.enter([g])
            return Expect(m.before, Lambda(x, Expect(after, g)))
        if isinstance(m, Lebesgue):
            l = parameter(g, 'l')
            return holonomic_integrate(g(l), (l, m.lower, m.upper))
        if isinstance(m, Counting):
            k = parameter(g, 'k')
            return Sum(g(k), (k, m.lower, m.upper))

class TestExpect:

    def test_Ret(self):
        from sympy.abc import x
        assert Expect(Ret(3), Lambda(x, x+10)) == 13
        assert Expect(Ret(x+1), Lambda(x, x*10)) == (x+1)*10

    def test_nondet(self):
        from sympy.abc import x
        g = Lambda(x, x+10)
        assert Expect(0, g) == 0
        assert Expect(Ret(3) + Ret(4), g) == 27
        assert Expect(Ret(3) * 5, g) == 65
        assert Expect(Ret(3) * 5 + Ret(4), g) == 79
        assert Expect((Ret(3) + Ret(4)) * 5, g) == 135

    def test_Bind(self):
        from sympy.abc import x, y, z
        g = Function('g')
        assert Expect(Bind(Ret(x+1)+Ret(x+2), y, Ret(y*10)), Lambda(z, g(z))) == g((x+1)*10) + g((x+2)*10)
        assert Expect(Bind(Ret(x+1)+Ret(x+2), y, Ret(y*10)), Lambda(y, g(y))) == g((x+1)*10) + g((x+2)*10)
        assert Expect(Bind(Ret(x+1)+Ret(x+2), y, Ret(y*10)), g)               == g((x+1)*10) + g((x+2)*10)
        assert Expect(Bind(Ret(x+1)+Ret(x+2), y, Ret(y*10)), Lambda(z, g(y,z))) == g(y, (x+1)*10) + g(y, (x+2)*10)

    def test_Lebesgue(self):
        from sympy.abc import x
        g = Lambda(x, x**2)
        assert Expect(Lebesgue(3,6), g) == 63
        assert Expect(Bind(Lebesgue(3,6), x, Ret(x**2)), S.IdentityFunction) == 63

    def test_var(self):
        from sympy.abc import x, y, z
        m = Function('m', commutative=False)
        n = Function('n', commutative=False)
        g = Function('g')
        assert Expect(Bind(Bind(Lebesgue(3,6), x, m(x)), y, n(y)), Lambda(z, g(z))).dummy_eq(integrate(Expect(m(x), Lambda(y, Expect(n(y), Lambda(z, g(z))))), (x,3,6)))

class BanishFailure(ValueError):
    def __str__(self):
        return ("Dependency prevents banishing %s in %s" % self.args)

def banish(m):
    # banish(Bind(b,x,m)) eliminates (integrates out) the variable x<~b
    # as long as x is used in m only in weights

    def _banish(b, x, w, m):
        if x not in m.free_symbols:
            return Mul(Expect(b, Lambda(x, w)), m)
        if m.is_Mul:
            (w1, m) = unweight(m)
            separation = separatevars(w1, [x], dict=True)
            return separation['coeff'] * _banish(b, x, w * separation[x], m)
        if m.is_Add:
            return Add(*(_banish(b, x, w, term) for term in m.args))
        if isinstance(m, Bind) and x not in m.before.free_symbols:
            return Bind(m.before, m.variable, _banish(b, x, w, m.after))
        raise BanishFailure(x, m)

    assert type(m) == Bind, m
    return _banish(m.before, m.variable, S.One, m.after)

class TestBanish:

    def test_normal(self):
        from sympy.abc import x, y
        assert banish(Bind(Lebesgue(-oo,oo),y,exp(-x**2-y**2+x*y)*Ret(x))) == sqrt(pi)*exp(-3*x**2/4)*Ret(x)

    def test_categorical(self):
        from sympy.abc import i, j, m, n
        M = IndexedBase('M')
        assert banish(Bind(Counting(0,n-1),j,M[i,j]*Ret(i))) == Sum(M[i,j],(j,0,n-1))*Ret(i)
