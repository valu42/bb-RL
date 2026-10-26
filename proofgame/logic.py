"""Sorted syntax, capture-avoiding substitution, and bounded definitional reduction.

Free variables have stable integer IDs. Bound variables use de Bruijn indices.
There is no eval, user code execution, arithmetic oracle, or theorem database.
"""
from dataclasses import dataclass
import re


class Invalid(ValueError):
    pass


class Limit(Invalid):
    pass


SORTS = ("Nat", "Bit", "List", "State", "Cfg")
SIG = {
    "s": (("Nat",), "Nat"), "add": (("Nat", "Nat"), "Nat"),
    "mul": (("Nat", "Nat"), "Nat"),
    "cons": (("Bit", "List"), "List"), "head": (("List",), "Bit"),
    "tail": (("List",), "List"), "cfg": (("State", "List", "List"), "Cfg"),
    "qof": (("Cfg",), "State"), "left": (("Cfg",), "List"),
    "right": (("Cfg",), "List"), "next": (("Cfg",), "Cfg"),
    "at": (("Nat", "Cfg"), "Cfg"), "qnext": (("State", "Bit"), "State"),
}
CONSTS = {"z": "Nat", "b0": "Bit", "b1": "Bit", "nil": "List"}


def freeze(x):
    return tuple(freeze(y) for y in x) if isinstance(x, (list, tuple)) else x


def parse(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    pos = 0
    def read(depth=0):
        nonlocal pos
        if depth > 160 or pos >= len(tokens):
            raise Invalid("incomplete or too deeply nested expression")
        t = tokens[pos]; pos += 1
        if t == "(":
            out = []
            while pos < len(tokens) and tokens[pos] != ")":
                out.append(read(depth + 1))
            if pos >= len(tokens) or not out:
                raise Invalid("unclosed or empty expression")
            pos += 1
            return tuple(out)
        if t == ")":
            raise Invalid("unexpected closing parenthesis")
        if re.fullmatch(r"v\d+", t): return ("var", int(t[1:]))
        if re.fullmatch(r"bvar\d+", t): return ("bound", int(t[4:]))
        return t
    out = read()
    if pos != len(tokens): raise Invalid("trailing input")
    return out


def show(x):
    if isinstance(x, str): return x
    if x[0] == "var": return f"v{x[1]}"
    if x[0] == "bound": return f"bvar{x[1]}"
    return "(" + " ".join(show(y) for y in x) + ")"


def size(x):
    return 1 if isinstance(x, (str, int)) else 1 + sum(size(y) for y in x[1:])


def shift(x, amount, cutoff=0):
    if not isinstance(x, tuple): return x
    if x[0] == "bound": return ("bound", x[1] + amount) if x[1] >= cutoff else x
    if x[0] == "var": return x
    if x[0] in ("all", "ex"):
        return (x[0], x[1], shift(x[2], amount, cutoff + 1))
    return (x[0], *(shift(a, amount, cutoff) for a in x[1:]))


def inst(body, term, depth=0):
    """Remove the outer binder; substitute term without capturing bound names."""
    if not isinstance(body, tuple): return body
    if body[0] == "var": return body
    if body[0] == "bound":
        i = body[1]
        return shift(term, depth) if i == depth else ("bound", i - 1) if i > depth else body
    if body[0] in ("all", "ex"):
        return (body[0], body[1], inst(body[2], term, depth + 1))
    return (body[0], *(inst(a, term, depth) for a in body[1:]))


def replace(x, old, new, depth=0):
    """Replace free occurrences of a well-scoped term, including under binders."""
    if x == shift(old, depth): return shift(new, depth)
    if not isinstance(x, tuple) or x[0] in ("var", "bound"): return x
    if x[0] in ("all", "ex"):
        return (x[0], x[1], replace(x[2], old, new, depth + 1))
    return (x[0], *(replace(a, old, new, depth) for a in x[1:]))


def type_of(x, machine, variables=(), bound=()):
    env = dict(variables)
    if isinstance(x, str):
        if x in CONSTS: return CONSTS[x]
        if x in machine.states: return "State"
        raise Invalid(f"unknown constant {x}")
    if not isinstance(x, tuple) or not x: raise Invalid("malformed term")
    op = x[0]
    if op in ("var", "bound"):
        if len(x) != 2 or type(x[1]) is not int or x[1] < 0: raise Invalid("invalid variable")
        if op == "var":
            if x[1] not in env: raise Invalid("variable is out of scope")
            return env[x[1]]
        if x[1] >= len(bound): raise Invalid("unbound variable")
        return bound[-1-x[1]]
    if op not in SIG: raise Invalid(f"unknown term constructor {op}")
    args, result = SIG[op]
    if len(x)-1 != len(args): raise Invalid("wrong term arity")
    if tuple(type_of(t, machine, variables, bound) for t in x[1:]) != args:
        raise Invalid("term sort mismatch")
    return result


def wf(f, machine, variables=(), bound=()):
    if f == "false": return
    if not isinstance(f, tuple) or not f: raise Invalid("expected a formula")
    op = f[0]
    if op == "eq" and len(f) == 3:
        if type_of(f[1], machine, variables, bound) != type_of(f[2], machine, variables, bound):
            raise Invalid("equality sort mismatch")
    elif op in ("and", "or", "imp") and len(f) == 3:
        wf(f[1], machine, variables, bound); wf(f[2], machine, variables, bound)
    elif op in ("all", "ex") and len(f) == 3 and f[1] in SORTS:
        wf(f[2], machine, variables, bound + (f[1],))
    else: raise Invalid("unknown formula or wrong arity")


@dataclass(frozen=True)
class Machine:
    # rows in q0/b0, q0/b1, q1/b0, ... order; (next state, written bit, direction)
    rows: tuple

    def __post_init__(self):
        object.__setattr__(self, "rows", freeze(self.rows))
        if not self.rows or len(self.rows) % 2 or len(self.rows) > 64:
            raise Invalid("machine needs 1..32 complete binary states")
        object.__setattr__(self,"_states",tuple(f"q{i}" for i in range(len(self.rows)//2))+("qH",))
        for row in self.rows:
            if len(row) != 3 or row[0] not in self.states or row[1] not in ("b0", "b1") or row[2] not in ("L", "R"):
                raise Invalid("invalid machine row")

    @property
    def states(self): return self._states

    def row(self, q, bit): return self.rows[2*int(q[1:]) + (bit == "b1")]

    def to_json(self): return {"rows": self.rows}


C0 = ("cfg", "q0", "nil", "nil")


def root_goal(answer):
    state = ("qof", ("at", ("bound", 0), C0))
    if answer == "HALT": return ("ex", "Nat", ("eq", state, "qH"))
    if answer == "NONHALT": return ("all", "Nat", ("imp", ("eq", state, "qH"), "false"))
    raise Invalid("unknown answer")


def normal(x, machine, fuel=20000):
    """Fixed defining equations only. Exhaustion raises Limit, never a truth value.

    next(qH,...) freezes a halted machine. HALT means reaches qH at some time;
    NONHALT means never reaches it. No external programming language is exposed.
    """
    remaining = [fuel]
    def go(t):
        remaining[0] -= 1
        if remaining[0] < 0: raise Limit("definition reduction budget exceeded")
        if not isinstance(t, tuple) or t[0] in ("var", "bound"): return t
        op = t[0]
        if op in ("all", "ex"): return (op, t[1], go(t[2]))
        args = tuple(go(a) for a in t[1:]); t = (op, *args)
        if op in ("s", "cons", "cfg", "eq", "and", "or", "imp"): return t
        a = args[0] if args else None
        result = t
        if op == "add":
            if a == "z": result = args[1]
            elif isinstance(a, tuple) and a[0] == "s": result = ("s", ("add", a[1], args[1]))
        elif op == "mul":
            if a == "z": result = "z"
            elif isinstance(a, tuple) and a[0] == "s": result = ("add", args[1], ("mul", a[1], args[1]))
        elif op in ("head", "tail"):
            if a == "nil": result = "b0" if op == "head" else "nil"
            elif isinstance(a, tuple) and a[0] == "cons": result = a[1 if op == "head" else 2]
        elif op in ("qof", "left", "right"):
            if isinstance(a, tuple) and a[0] == "cfg": result = a[{"qof":1,"left":2,"right":3}[op]]
            elif op == "qof" and isinstance(a, tuple) and a[0] == "next":
                result = ("qnext", ("qof", a[1]), ("head", ("right", a[1])))
        elif op == "at":
            if a == "z": result = args[1]
            elif isinstance(a, tuple) and a[0] == "s": result = ("next", ("at", a[1], args[1]))
        elif op == "qnext":
            if a == "qH": result = "qH"
            elif isinstance(a, str) and a in machine.states and args[1] in ("b0", "b1"):
                result = machine.row(a, args[1])[0]
        elif op == "next" and isinstance(a, tuple) and a[0] == "cfg":
            _, q, left, right = a
            if q == "qH": result = a
            elif isinstance(q, str) and q in machine.states:
                bit = go(("head", right))
                if bit in ("b0", "b1"):
                    q2, written, direction = machine.row(q, bit)
                    if direction == "R": result = ("cfg", q2, ("cons", written, left), ("tail", right))
                    else: result = ("cfg", q2, ("tail", left), ("cons", ("head", left), ("cons", written, ("tail", right))))
        return go(result) if result != t else t
    return go(x)


def disjoint(a, b, machine):
    """Constructor disjointness, including injectivity down a constructor tree."""
    constructors = {"z", "s", "b0", "b1", "nil", "cons", "cfg", *machine.states}
    head_a = a if isinstance(a, str) else a[0]
    head_b = b if isinstance(b, str) else b[0]
    if head_a not in constructors or head_b not in constructors: return False
    if head_a != head_b: return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        return any(disjoint(x, y, machine) for x, y in zip(a[1:], b[1:]))
    return False


def subterms(x):
    yield x
    if isinstance(x, tuple) and x[0] not in ("var", "bound"):
        for y in x[2:] if x[0] in ("all", "ex") else x[1:]:
            yield from subterms(y)
