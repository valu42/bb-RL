"""Human-written verification fixtures. Never imported by the trainer."""
from .logic import C0, type_of
from .environment import Game
from .tasks import fixtures


def play(game,action,trace):
    game.step(action); trace.append(action)
    if game.error: raise RuntimeError(f"{action}: {game.error}")


def build(game,x,trace,bound=()):
    """Drive the public constructor menu, not a private certificate shortcut."""
    g=game.current()
    if isinstance(x,str): play(game,"put:"+x,trace); return
    op=x[0]
    if op=="var": play(game,f"var:v{x[1]}",trace); return
    if op=="bound": play(game,f"bound:{x[1]}",trace); return
    if op in ("all","ex"):
        play(game,f"put:{op}:{x[1]}",trace)
        build(game,x[2],trace,bound+(x[1],)); return
    if op=="eq": play(game,"put:eq:"+type_of(x[1],game.machine,g.variables,bound),trace)
    else: play(game,"put:"+op,trace)
    for a in x[1:]: build(game,a,trace,bound)


def with_expression(game,action,expression,trace):
    play(game,action,trace); build(game,expression,trace); play(game,"commit",trace)


def run_demo(name):
    game=Game(fixtures()[name],budget=512,max_nodes=20000,max_depth=160); trace=[]
    if name in ("one_step","two_step","three_step"):
        play(game,"choose:HALT",trace)
        n="z"
        for _ in range({"one_step":1,"two_step":2,"three_step":3}[name]): n=("s",n)
        with_expression(game,"rule:exI",n,trace)
        play(game,"rule:refl",trace)
    elif name=="forever":
        play(game,"choose:NONHALT",trace)
        lemma=("all","Nat",("eq",("qof",("at",("bound",0),C0)),"q0"))
        with_expression(game,"rule:cut",lemma,trace)
        play(game,"rule:natI",trace)
        play(game,"rule:refl",trace)
        play(game,"rule:reduce",trace)
        play(game,"rule:rewrite:h0:forward",trace)
        scanned=("head",("right",("at",("var",0),C0)))
        with_expression(game,"rule:casesBit",scanned,trace)
        for _ in range(2):
            play(game,"rule:rewrite:h1:forward",trace)
            play(game,"rule:refl",trace)
        play(game,"rule:allI",trace)
        play(game,"rule:impI",trace)
        with_expression(game,"rule:specialize:h0",("var",0),trace)
        play(game,"rule:rewrite:h2:forward",trace)
        play(game,"rule:discriminate:h1",trace)
    elif name=="two_state_forever":
        # Prove the root directly by induction, without a pre-supplied invariant.
        # This action sequence is hand-written solely as a verification fixture.
        play(game,"choose:NONHALT",trace)
        play(game,"rule:natI",trace)
        play(game,"rule:impI",trace)
        play(game,"rule:discriminate:h0",trace)
        play(game,"rule:impI",trace)
        play(game,"rule:reduce",trace)
        state=("qof",("at",("var",0),C0))
        scanned=("head",("right",("at",("var",0),C0)))
        with_expression(game,"rule:casesState",state,trace)
        for _ in range(2):
            play(game,"rule:rewrite:h2:forward",trace)
            with_expression(game,"rule:casesBit",scanned,trace)
            for _ in range(2):
                play(game,"rule:rewrite:h3:forward",trace)
                play(game,"rule:discriminate:h1",trace)
        play(game,"rule:rewrite:h2:forward",trace)
        play(game,"rule:apply:h0",trace)
        play(game,"rule:refl",trace)
    elif name=="blank_sweeper":
        # The machine has a halting transition, but can never scan a written 1.
        # An existential tape invariant is hand-written here, not given in training.
        play(game,"choose:NONHALT",trace)
        lemma=("all","Nat",("ex","List",("eq",("at",("bound",1),C0),
                                            ("cfg","q0",("bound",0),"nil"))))
        with_expression(game,"rule:cut",lemma,trace)
        play(game,"rule:natI",trace)
        with_expression(game,"rule:exI","nil",trace)
        play(game,"rule:refl",trace)
        play(game,"rule:destruct:h0",trace)
        with_expression(game,"rule:exI",("cons","b1",("var",1)),trace)
        play(game,"rule:reduce",trace)
        play(game,"rule:rewrite:h1:forward",trace)
        play(game,"rule:refl",trace)
        play(game,"rule:allI",trace)
        play(game,"rule:impI",trace)
        with_expression(game,"rule:specialize:h0",("var",0),trace)
        play(game,"rule:destruct:h2",trace)
        play(game,"rule:rewrite:h3:forward",trace)
        play(game,"rule:discriminate:h1",trace)
    else: raise ValueError(name)
    play(game,"submit",trace)
    return game,trace
