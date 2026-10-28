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
    "dispatch": (("State", "Bit", "List", "List"), "Cfg"),
    "pack": (("Bit", "List"), "List"),
    "append": (("List", "List"), "List"),
    "repeat": (("Bit", "Nat"), "List"),
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
    if op == "num":
        if len(x)!=2 or not isinstance(x[1],str) or not x[1].isascii() or not x[1].isdigit() or str(int(x[1]))!=x[1] or len(x[1])>128: raise Invalid("invalid numeral")
        return "Nat"
    if op in ("var", "bound"):
        if len(x) != 2 or type(x[1]) is not int or x[1] < 0: raise Invalid("invalid variable")
        if op == "var":
            if x[1] not in env: raise Invalid("variable is out of scope")
            return env[x[1]]
        if x[1] >= len(bound): raise Invalid("unbound variable")
        return bound[-1-x[1]]
    sig = {**SIG, **getattr(machine,"signature",{})}
    if op not in sig: raise Invalid(f"unknown term constructor {op}")
    args, result = sig[op]
    if len(x)-1 != len(args): raise Invalid("wrong term arity")
    if tuple(type_of(t, machine, variables, bound) for t in x[1:]) != args:
        raise Invalid("term sort mismatch")
    return result


def wf(f, machine, variables=(), bound=()):
    if f == "false": return
    if not isinstance(f, tuple) or not f: raise Invalid("expected a formula")
    op = f[0]
    if op == "eq" and len(f) == 3:
        if type_of(f[1], machine, variables, bound) not in SORTS or type_of(f[1], machine, variables, bound) != type_of(f[2], machine, variables, bound):
            raise Invalid("equality sort mismatch")
    elif op in ("and", "or", "imp") and len(f) == 3:
        wf(f[1], machine, variables, bound); wf(f[2], machine, variables, bound)
    elif op in ("all", "ex") and len(f) == 3 and f[1] in SORTS:
        wf(f[2], machine, variables, bound + (f[1],))
    elif op in getattr(machine,"signature",{}) and machine.signature[op][1]=="Formula":
        type_of(f,machine,variables,bound)
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


def numeral(n):
    return "z" if n == 0 else ("num", str(n))


def nat_view(t):
    if t == "z": return (0, None)
    if isinstance(t,tuple) and t[0]=="num":
        n=int(t[1]); return (0,None) if n==0 else (1,numeral(n-1))
    if isinstance(t,tuple) and t[0]=="s": return (1,t[1])
    return None


def locations(t, path=()):
    yield path,t
    if isinstance(t,tuple) and t[0] not in ("var","bound","num"):
        start=2 if t[0] in ("all","ex") else 1
        for i in range(start,len(t)):
            yield from locations(t[i],path+(i,))


def at_path(t,path):
    for i in path:
        if type(i)!=int or not isinstance(t,tuple) or i<1 or i>=len(t) or t[0] in ("var","bound","num"): raise Invalid("bad occurrence")
        t=t[i]
    return t


def put_path(t,path,value):
    if not path: return value
    at_path(t,path)
    xs=list(t); xs[path[0]]=put_path(xs[path[0]],path[1:],value)
    return tuple(xs)


def unfold(t,m):
    """Exactly one equation. Never recursively evaluates a child or result."""
    if not isinstance(t,tuple): raise Invalid("no defining equation")
    op,*args=t
    a=args[0] if args else None
    if op in ("add","mul","at"):
        v=nat_view(a)
        if v:
            b=args[1]
            if op=="add": return b if v[0]==0 else ("s",("add",v[1],b))
            if op=="mul": return "z" if v[0]==0 else ("add",b,("mul",v[1],b))
            return b if v[0]==0 else ("next",("at",v[1],b))
    if op=="s" and (a=="z" or isinstance(a,tuple) and a[0]=="num"):
        return numeral(1 if a=="z" else int(a[1])+1)
    if op in ("head","tail"):
        if a=="nil": return "b0" if op=="head" else "nil"
        if isinstance(a,tuple) and a[0]=="cons": return a[1 if op=="head" else 2]
    if op in ("qof","left","right") and isinstance(a,tuple) and a[0]=="cfg":
        return a[{"qof":1,"left":2,"right":3}[op]]
    if op=="qof" and isinstance(a,tuple) and a[0]=="next":
        # This equation follows directly from the fixed transition definition.
        return ("qnext",("qof",a[1]),("head",("right",a[1])))
    if op=="qnext" and a in m.states and args[1] in ("b0","b1"):
        return "qH" if a=="qH" else m.row(a,args[1])[0]
    if op=="next" and isinstance(a,tuple) and a[0]=="cfg":
        _,q,l,r=a
        return a if q=="qH" else ("dispatch",q,("head",r),l,r)
    if op=="dispatch":
        q,b,l,r=args
        if q in m.states and q!="qH" and b in ("b0","b1"):
            q2,w,d=m.row(q,b)
            if d=="R": return ("cfg",q2,("pack",w,l),("tail",r))
            return ("cfg",q2,("tail",l),("pack",("head",l),("pack",w,("tail",r))))
    if op=="pack":
        b,x=args
        if b in ("b0","b1") and x=="nil": return "nil" if b=="b0" else ("cons",b,x)
        if isinstance(x,tuple) and x[0]=="cons": return ("cons",b,x)
    if op=="append":
        x,y=args
        if x=="nil":return y
        if isinstance(x,tuple) and x[0]=="cons":return ("cons",x[1],("append",x[2],y))
    if op=="repeat":
        b,n=args; v=nat_view(n)
        if v:return "nil" if v[0]==0 else ("cons",b,("repeat",b,v[1]))
    if op in getattr(m,"definitions",{}):
        return m.unfold_definition(op,args)
    raise Invalid("no defining equation")


def bounded(x, max_nodes=100000, max_depth=256):
    stack=[(x,0)]; count=0
    while stack:
        t,d=stack.pop();count+=1
        if count>max_nodes or d>max_depth: raise Limit("term resource limit")
        if isinstance(t,(tuple,list)):
            stack.extend((y,d+1) for y in t)
        elif isinstance(t,dict):stack.extend((y,d+1) for y in t.values())
        elif isinstance(t,str) and len(t)>10000:raise Limit("atom resource limit")
    return count
