import copy
import json
import random
import unittest
from proofgame.logic import (Machine,C0,Invalid,Limit,inst,replace,parse,show,normal,
                             wf,type_of,root_goal)
from proofgame.kernel import Goal,expand,check
from proofgame.environment import Game
from proofgame.demo import run_demo,with_expression,play
from proofgame.tasks import fixtures,sample,is_test,key


class LogicTests(unittest.TestCase):
    def setUp(self): self.m=fixtures()["one_step"]

    def test_capture_avoidance(self):
        body=parse("(all Nat (eq bvar1 bvar0))")
        self.assertEqual(inst(body,("var",10)),parse("(all Nat (eq v10 bvar0))"))
        self.assertEqual(inst(body,("bound",0)),parse("(all Nat (eq bvar1 bvar0))"))
        # Substitution under a nested binder never captures a free variable.
        self.assertEqual(replace(parse("(all Nat (eq v1 bvar0))"),("var",1),("var",2)),
                         parse("(all Nat (eq v2 bvar0))"))

    def test_roundtrip(self):
        for f in [root_goal("HALT"),root_goal("NONHALT"),parse("(all Nat (eq v15 bvar0))")]:
            self.assertEqual(parse(show(f)),f)

    def test_scoping_and_types(self):
        for x in [("var",0),("bound",0),("s","nil"),("var",True),"q99"]:
            with self.assertRaises(Invalid): type_of(x,self.m)
        with self.assertRaises(Invalid): wf(("eq","z","nil"),self.m)

    def test_motion_and_halt(self):
        m=Machine((("q0","b1","L"),("qH","b0","R")))
        self.assertEqual(normal(("next",C0),m),parse("(cfg q0 nil (cons b0 (cons b1 nil)))"))
        halted=parse("(cfg qH (cons b1 nil) nil)")
        self.assertEqual(normal(("next",halted),m),halted)
        self.assertEqual(normal(("qof",("at",("s","z"),C0)),self.m),"qH")

    def test_reducer_limit_is_not_truth(self):
        with self.assertRaises(Limit): normal(("s","z"),self.m,fuel=0)

    def test_random_concrete_step_agrees_with_reference(self):
        rng=random.Random(5)
        def ls(bits):
            x="nil"
            for b in reversed(bits): x=("cons",b,x)
            return x
        for _ in range(100):
            m=sample(rng,(2,3)); q=rng.choice(m.states)
            left=[rng.choice(("b0","b1")) for _ in range(rng.randrange(4))]
            right=[rng.choice(("b0","b1")) for _ in range(rng.randrange(4))]
            c=("cfg",q,ls(left),ls(right))
            if q=="qH": expected=c
            else:
                q2,w,d=m.row(q,right[0] if right else "b0")
                if d=="R": l2,r2=[w]+left,right[1:]
                else: l2,r2=left[1:],[left[0] if left else "b0",w]+right[1:]
                expected=("cfg",q2,ls(l2),ls(r2))
            self.assertEqual(normal(("next",c),m),expected)
            self.assertEqual(normal(("qof",("next",c)),m),expected[1])


class KernelTests(unittest.TestCase):
    def setUp(self): self.m=fixtures()["one_step"]

    def test_halting_and_inductive_nonhalting_certificates(self):
        for name,answer in [("one_step","HALT"),("two_step","HALT"),("three_step","HALT"),
                            ("forever","NONHALT"),("two_state_forever","NONHALT"),
                            ("blank_sweeper","NONHALT")]:
            game,trace=run_demo(name)
            self.assertEqual(game.status,"SOLVED")
            self.assertEqual(check(json.loads(json.dumps(game.certificate()))),answer)
            replay=Game(game.machine,budget=512)
            for action in trace: replay.step(action); self.assertIsNone(replay.error)
            self.assertEqual(replay.status,"SOLVED")

    def test_wrong_witness_and_machine_rejected(self):
        game,_=run_demo("one_step"); cert=game.certificate()
        cert["proof"]["data"]="z"
        with self.assertRaises(Invalid): check(cert)
        game,_=run_demo("forever"); cert=game.certificate()
        cert["machine"]=fixtures()["one_step"].to_json()
        with self.assertRaises(Invalid): check(cert)

    def test_missing_subproof_and_forged_status_rejected(self):
        game,_=run_demo("forever"); cert=game.certificate()
        cert["proof"]["children"].pop()
        with self.assertRaises(Invalid): check(cert)
        game,_=run_demo("one_step"); cert=game.certificate(); cert["proof"]["solved"]=True
        with self.assertRaises(Invalid): check(cert)

    def test_cut_cannot_use_itself(self):
        g=Goal(root_goal("HALT")); left,right=expand(self.m,g,"cut","false")
        self.assertEqual(left.facts,()); self.assertEqual(right.facts,("false",))
        with self.assertRaises(Invalid): expand(self.m,left,"exact",0)

    def test_existential_witness_scope(self):
        target=parse("(eq z z)"); hypothesis=parse("(ex Nat (eq bvar0 z))")
        child,=expand(self.m,Goal(target,(),(hypothesis,)),"destruct",0)
        self.assertEqual(child.variables,((0,"Nat"),))
        self.assertEqual(child.target,target)
        with self.assertRaises(Invalid): expand(self.m,Goal(parse("(eq v0 v0)")),"refl")

    def test_induction_keeps_outer_parameter_and_nested_binder(self):
        t=parse("(all Nat (all Nat (eq (add bvar1 v7) bvar0)))")
        base,step=expand(self.m,Goal(t,((7,"Nat"),)),"natI")
        self.assertEqual(step.variables,((7,"Nat"),(8,"Nat")))
        self.assertEqual(step.target,parse("(all Nat (eq (add (s v8) v7) bvar0))"))
        self.assertEqual(step.facts[-1],parse("(all Nat (eq (add v8 v7) bvar0))"))
        t=parse("(all List (eq bvar0 v9))")
        base,step=expand(self.m,Goal(t,((9,"List"),)),"listI")
        self.assertEqual(step.target,parse("(eq (cons v10 v11) v9)"))
        self.assertEqual(step.facts[-1],parse("(eq v11 v9)"))

    def test_rewrite_under_binder(self):
        g=Goal(parse("(all Nat (eq v2 bvar0))"),((2,"Nat"),), (parse("(eq v2 z)"),))
        child,=expand(self.m,g,"rewrite",(0,"forward"))
        self.assertEqual(child.target,parse("(all Nat (eq z bvar0))"))

    def test_constructor_injectivity(self):
        g=Goal(parse("(eq v0 v1)"),((0,"Nat"),(1,"Nat")),(parse("(eq (s v0) (s v1))"),))
        child,=expand(self.m,g,"inject",0)
        self.assertEqual(expand(self.m,child,"exact",1),[])
        with self.assertRaises(Invalid): expand(self.m,Goal("false",(),(("eq","z","z"),)),"inject",0)

    def test_classical_and_logical_rules(self):
        a=parse("(eq z z)"); g=Goal(("and",a,a))
        self.assertEqual(len(expand(self.m,g,"andI")),2)
        f=("or",a,a)
        self.assertEqual(len(expand(self.m,Goal(a,(),(f,)),"destruct",0)),2)
        child,=expand(self.m,Goal(("imp",a,a)),"impI")
        self.assertEqual(expand(self.m,child,"exact",0),[])
        self.assertEqual(expand(self.m,Goal("false",(),(("eq","z",("s","z")),)),"discriminate",0),[])
        with self.assertRaises(Invalid): expand(self.m,Goal("false",((0,"Nat"),),(("eq",("var",0),"z"),)),"discriminate",0)


class GameTests(unittest.TestCase):
    def setUp(self): self.m=fixtures()["one_step"]

    def test_reset_budget_and_invalid_action(self):
        g=Game(self.m,budget=3)
        g.step("choose:HALT"); g.step("reset")
        self.assertEqual(g.left,1); self.assertIsNone(g.answer)
        _,r,done=g.step("not-an-action")
        self.assertEqual((g.left,r,done,g.status),(0,0,True,"EXHAUSTED"))
        with self.assertRaises(Invalid): g.step("reset")

    def test_last_action_submission_wins(self):
        _,trace=run_demo("one_step"); g=Game(self.m,budget=len(trace))
        for a in trace: g.step(a)
        self.assertEqual(g.status,"SOLVED"); self.assertEqual(g.reward,1)

    def test_failed_rule_is_atomic_and_costs_action(self):
        g=Game(self.m); trace=[]; play(g,"choose:HALT",trace)
        with_expression(g,"rule:exI","z",trace)
        before=copy.deepcopy(g.proof); left=g.left
        g.step("rule:refl")
        self.assertEqual(g.proof,before); self.assertEqual(g.left,left-1); self.assertIsNotNone(g.error)

    def test_ten_variables_with_no_special_vocabulary(self):
        g=Game(self.m,budget=256); trace=[]; play(g,"choose:HALT",trace)
        lemma=parse("(eq bvar0 bvar0)")
        for _ in range(10): lemma=("all","Nat",lemma)
        with_expression(g,"rule:cut",lemma,trace)
        for _ in range(10): play(g,"rule:allI",trace)
        self.assertEqual(g.current().variables,tuple((i,"Nat") for i in range(10)))
        play(g,"rule:cut",trace); play(g,"put:eq:Nat",trace)
        self.assertTrue(all(f"var:v{i}" in g.available_actions() for i in range(10)))
        play(g,"cancel",trace); play(g,"rule:refl",trace)
        self.assertEqual(g.current().variables,())

    def test_random_walks_keep_every_state_serializable_and_well_scoped(self):
        rng=random.Random(19)
        for _ in range(12):
            g=Game(sample(rng),budget=50,max_nodes=400)
            while g.status=="ACTIVE":
                menu=g.available_actions(); self.assertEqual(len(menu),len(set(menu)))
                g.step(rng.choice(menu)); json.dumps(g.observation())
                for _,goal in g.goals(): wf(goal.target,g.machine,goal.variables)

    def test_train_test_partition(self):
        rng=random.Random(5)
        train={key(sample(rng,(2,3),"train")) for _ in range(100)}
        test={key(sample(rng,(2,3),"test")) for _ in range(100)}
        self.assertFalse(train & test)


if __name__=="__main__": unittest.main()
