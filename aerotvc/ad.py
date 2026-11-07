"""Float64 JAX AD or NumPy reverse mode with forward-over-reverse HVPs."""

import os

import numpy as np

if os.environ.get("AEROTVC_DISABLE_JAX") == "1":
    jax = None
else:
    try:
        import jax
    except ImportError:
        jax = None

if jax is not None:
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

BACKEND = "jax" if jax is not None else "numpy"


def _unbroadcast(value, shape):
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


class Var:
    """A differentiable array supporting arithmetic, constant powers and sum."""

    __array_priority__ = 1000

    def __init__(self, value, tangent=None, _parents=()):
        self.value = np.asarray(value, dtype=float)
        self.tangent = (np.zeros_like(self.value) if tangent is None
                        else np.broadcast_to(tangent, self.value.shape).astype(float))
        self.grad = np.zeros_like(self.value)
        self._grad_tangent = np.zeros_like(self.value)
        self._parents = _parents

    @staticmethod
    def _var(value):
        return value if isinstance(value, Var) else Var(value)

    def __add__(self, other):
        other = self._var(other)
        return Var(self.value + other.value, self.tangent + other.tangent,
                   ((self, 1., 0.), (other, 1., 0.)))

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self._var(other)

    def __rsub__(self, other):
        return self._var(other) - self

    def __mul__(self, other):
        other = self._var(other)
        return Var(self.value * other.value,
                   self.tangent * other.value + self.value * other.tangent,
                   ((self, other.value, other.tangent),
                    (other, self.value, self.tangent)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * self._var(other)**-1

    def __rtruediv__(self, other):
        return self._var(other) * self**-1

    def __pow__(self, power):
        if not np.isscalar(power):
            raise TypeError("Var powers must be scalar constants")
        if power == 0:
            return Var(np.ones_like(self.value), _parents=((self, 0., 0.),))
        if power == 1:
            return self + 0
        derivative = power * self.value**(power - 1)
        derivative_tangent = (power * (power - 1) * self.value**(power - 2)
                              * self.tangent)
        return Var(self.value**power, derivative * self.tangent,
                   ((self, derivative, derivative_tangent),))

    def sum(self):
        return Var(self.value.sum(), self.tangent.sum(),
                   ((self, np.ones_like(self.value), np.zeros_like(self.value)),))

    def backward(self):
        if self.value.ndim != 0:
            raise ValueError("backward requires a scalar output")
        order, visited = [], set()
        # Iterative postorder also handles long expression chains.
        stack = [(self, False)]
        while stack:
            node, ready = stack.pop()
            if ready:
                order.append(node)
            elif id(node) not in visited:
                visited.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p, _, _ in node._parents)
        for node in order:
            node.grad = np.zeros_like(node.value)
            node._grad_tangent = np.zeros_like(node.value)
        self.grad = np.ones_like(self.value)
        for node in reversed(order):
            for parent, derivative, derivative_tangent in node._parents:
                parent.grad += _unbroadcast(np.asarray(node.grad * derivative),
                                            parent.value.shape)
                parent._grad_tangent += _unbroadcast(np.asarray(
                    node._grad_tangent * derivative + node.grad * derivative_tangent),
                    parent.value.shape)


def make_energy(a, b=0.0, c=0.0):
    module = jnp if jax is not None else np
    a, b, c = (module.asarray(v, dtype=module.float64) for v in (a, b, c))

    def energy(x):
        return (a * x**2 / 2 + b * x**4 / 4 + c * x).sum()

    return energy


def value_and_grad(function):
    if jax is not None:
        return jax.value_and_grad(function)

    def evaluate(x):
        variable = Var(x)
        result = function(variable)
        if not isinstance(result, Var):
            result = Var(result)
        result.backward()
        return float(result.value), variable.grad

    return evaluate


def grad(function):
    evaluate = value_and_grad(function)
    return lambda x: evaluate(x)[1]


def hvp(function, x, v):
    x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
    if x.shape != v.shape:
        raise ValueError("x and v must have the same shape")
    if jax is not None:
        return jax.jvp(jax.grad(function), (jnp.asarray(x),), (jnp.asarray(v),))[1]
    variable = Var(x, tangent=v)
    result = function(variable)
    if not isinstance(result, Var):
        result = Var(result)
    result.backward()
    return variable._grad_tangent
