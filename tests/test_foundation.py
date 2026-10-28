import builtins
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from proofgame.foundation.logic import Machine,Invalid,Limit,unfold,C0
from proofgame.foundation.kernel import check,expand,Goal,FORMAT
from proofgame.foundation.environment import Game
from proofgame.foundation.definitions import Theory
from proofgame.foundation.compiler import teach,replay,compile_halt,compile_closed,compile_cycle,Builder
from proofgame.foundation.data import family,partition,random_machine,generate_manifest,Corpus

HALT=Machine((('qH','b0','R'),('qH','b1','R')))
CYCLE=Machine((('q1','b0','R'),('qH','b0','R'),('q0','b0','L'),('qH','b0','R')))
FOREVER=Machine((('q0','b1','R'),('q0','b0','R')))

class FoundationTests(unittest.TestCase):
    def test_no_hidden_simulation_or_helpers(self):
        goal=Goal(('eq',('qof',('at',('num','1'),C0)),'qH'))
        for rule in ('refl','reduce','closedStates','exactCycle'):
            with self.assertRaises(Invalid):expand(HALT,goal,rule,None)
        obs=Game(HALT).observation()
        self.assertFalse(any('cert:' in a['id'] for a in obs['actions']))

    def test_one_equation_not_recursive_evaluation(self):
        t=('at',('num','100000'),C0)
        self.assertEqual(unfold(t,HALT),('next',('at',('num','99999'),C0)))
        with self.assertRaises(Invalid):unfold(('head',('tail',('cons','b1','nil'))),HALT)

    def test_three_compilers_and_legal_replay(self):
        for cert in (compile_halt(HALT,1),compile_closed(FOREVER,('q0',)),compile_cycle(CYCLE,0,2)):
            self.assertEqual(check(json.loads(json.dumps(cert))),cert['answer'])
            self.assertGreater(len(replay(cert)),len(cert['steps']))
            truncated=copy.deepcopy(cert);truncated['steps'].pop()
            with self.assertRaises(Invalid):check(truncated)
        with self.assertRaises(Invalid):compile_halt(HALT,0)
        with self.assertRaises(Invalid):compile_cycle(CYCLE,0,1)

    def test_mutated_machine_rejected(self):
        cert=compile_halt(HALT,1);cert['machine']=FOREVER.to_json()
        with self.assertRaises(Invalid):check(cert)
        cert=compile_closed(FOREVER,('q0',));cert['machine']=HALT.to_json()
        with self.assertRaises(Invalid):check(cert)

    def test_teacher_modules_not_needed_for_check_or_game(self):
        cert=compile_cycle(CYCLE,0,2);actions=replay(cert)
        original=builtins.__import__
        def no_teacher(name,*a,**kw):
            if any(x in name for x in ('compiler','nonhalt_certificates')):raise AssertionError('online teacher import')
            return original(name,*a,**kw)
        with patch('builtins.__import__',no_teacher):
            self.assertEqual(check(cert),'NONHALT');g=Game(CYCLE)
            for a in actions:g.step(a,observe=False)
            self.assertEqual(g.status,'SOLVED')

    def test_symbolic_induction_and_local_lemma(self):
        lemma=('all','List',('eq',('append',('bound',0),'nil'),('bound',0)))
        b=Builder(HALT,'HALT');b.emit('cut',lemma);b.emit('listI');b.norm();b.emit('refl')
        b.norm();b.rewrite_all(0);b.emit('refl');b.emit('exI',('num','1'));b.norm();b.emit('refl')
        cert=b.finish();self.assertTrue(replay(cert))

    def test_definitions_termination_and_scope(self):
        m=Theory(HALT)
        definition={'name':'f0','args':['Nat'],'result':'Nat','recursive':0,'base':'z','step':('s',('f0',('var',1)))}
        m.define(definition)
        self.assertEqual(unfold(('f0',('num','4')),m),('s',('f0',('num','3'))))
        for body in [('f0',('var',0)),('f0',('s',('var',1))),('var',5)]:
            bad={**definition,'step':body}
            with self.assertRaises(Invalid):Theory(HALT).define(bad)
        with self.assertRaises(Invalid):Theory(HALT).define({**definition,'base':('f0','z')})

    def test_define_through_finite_menu_and_verify(self):
        g=Game(HALT)
        actions=['choose:HALT','define','def:param:Nat','def:result:Nat','def:recursive:0',
                 'put:z','commit','put:s','put:f0','var:v1','commit']
        for a in actions:
            g.step(a,observe=False);self.assertIsNone(g.error)
        c=compile_halt(HALT,1)
        actions=replay(c)
        for a in actions[1:]:g.step(a,observe=False);self.assertIsNone(g.error)
        self.assertEqual(g.status,'SOLVED');self.assertEqual(check(g.certificate()),'HALT')

    def test_scope_and_limits(self):
        with self.assertRaises(Invalid):expand(HALT,Goal(('eq',('var',0),('var',0))),'refl')
        lemma=('eq','z','z');children=expand(HALT,Goal(lemma),'cut','false')
        self.assertEqual(children[0].facts,())
        with self.assertRaises(Invalid):expand(HALT,children[0],'exact',0)
        cert=compile_halt(HALT,1)
        with self.assertRaises(Limit):check(cert,max_steps=1)
        with self.assertRaises(Limit):check(cert,max_work=1)
        g=Game(HALT,budget=2);g.step('choose:HALT');g.step('reset');self.assertEqual(g.status,'EXHAUSTED')

    def test_random_compiler_output_and_reference_steps(self):
        rng=random.Random(41);checked=0
        for n in (2,3,4,5):
            for _ in range(8):
                m=random_machine(rng,n)
                try:
                    for source,cert in teach(m,12,max_steps=3000):
                        self.assertEqual(check(cert),cert['answer']);replay(cert,budget=4096);checked+=1
                except Limit:pass
        self.assertGreater(checked,10)

    def test_family_invariant_under_renaming_reflection_and_embedding(self):
        a=Machine((('q1','b1','R'),('qH','b0','L'),('q0','b0','L'),('q1','b1','R')))
        reflected=Machine(tuple((q,b,'L' if d=='R' else 'R') for q,b,d in a.rows))
        embedded=Machine(a.rows+(('q2','b0','L'),('qH','b1','R')))
        self.assertEqual(family(a),family(reflected));self.assertEqual(family(a),family(embedded))

    def test_dataset_splits_and_roundtrip(self):
        m=generate_manifest(9,2,4,4,4)
        fs={s:{r['family_id'] for r in rs} for s,rs in m['splits'].items()}
        self.assertFalse(fs['train']&fs['dev']);self.assertFalse(fs['train']&fs['test'])
        for row in m['jobs']:
            self.assertEqual(partition(row['family_id'],9),'train')
            self.assertNotIn(row['family_id'],m['excluded_families'])
        self.assertEqual(m,generate_manifest(9,2,4,4,4))

if __name__=='__main__':unittest.main()
