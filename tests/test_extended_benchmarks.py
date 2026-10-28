import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from proofgame.environment import Game
from proofgame.kernel import check
from proofgame.logic import Invalid, Machine
from proofgame.mixed_dataset import build_dataset, validate_dataset
from proofgame.nonhalt_certificates import check_closed_states
from proofgame.special_benchmarks import (benchmark_definitions, benchmark_rows,
    make_benchmark_report, format_compact_report, write_benchmark_report)
from tests.test_interesting_machines import transitions, unary_proof_actions, arithmetic_107_actions


class ExtendedBenchmarkTests(unittest.TestCase):
    def test_twelve_distinct_cases_and_reference_certificates(self):
        definitions = benchmark_definitions('extended')
        self.assertEqual(len(definitions), 12)
        self.assertEqual(len({r['machine_id'] for r in benchmark_rows('extended')}), 12)
        self.assertTrue(all(set(row)=={'machine_id','machine'}
                            for row in benchmark_rows('extended')))
        for definition in definitions:
            with self.subTest(case=definition['id']):
                machine = Machine(definition['rows'])
                if definition.get('halt_steps'):
                    t = definition['halt_steps']
                    trajectory = list(transitions(machine, t+1))
                    self.assertEqual(len(trajectory), t)
                    self.assertEqual(trajectory[-1][1], 'qH')
                    actions = arithmetic_107_actions() if t==107 else unary_proof_actions(t)
                elif definition.get('cycle'):
                    start, period = definition['cycle']
                    actions = ['choose:NONHALT','rule:exactCycle',f'cert:start:{start}',
                               f'cert:period:{period}','cert:verify','submit']
                    visited = {q for _,q,_,_ in transitions(machine,start+period)} | {'q0'}
                    self.assertEqual(len(visited), len(machine.rows)//2)
                    with self.assertRaises(Invalid):
                        check_closed_states(machine, tuple(q for q in machine.states if q!='qH'))
                else:
                    continue  # Existing legacy proofs and growth invariants have separate tests.
                game = Game(machine,budget=100)
                errors = []
                for action in actions:
                    game.step(action,observe=False)
                    if game.error: errors.append(game.error)
                if definition.get('halt_steps') == 107:
                    self.assertIn('definition reduction budget exceeded',errors)
                    self.assertNotEqual(game.status,'SOLVED')
                    self.assertIn('reference_note', definition)
                else:
                    self.assertEqual(game.status,'SOLVED',errors)
                    check(game.certificate())

    def test_growth_and_drift_invariants(self):
        cases = {d['id']:d for d in benchmark_definitions('extended')}
        for name, period in [('growing_frontier_4',4),('translated_cycle_3',3)]:
            machine = Machine(cases[name]['rows'])
            states = list(transitions(machine,period*200))
            self.assertEqual(len(states),period*200)
            for step,q,head,tape in states:
                n, phase = divmod(step,period)
                if name=='growing_frontier_4':
                    expected_q = ('q0','q1','q2','q3')[phase]
                    expected_head = 2*n + (0,1,0,1)[phase]
                    expected_tape = frozenset(range(2*n + (0,1,2,2)[phase]))
                else:
                    expected_q = ('q0','q1','q2')[phase]
                    expected_head = n + (0,1,0)[phase]
                    expected_tape = frozenset({n}) if phase else frozenset()
                self.assertEqual((q,head,tape),(expected_q,expected_head,expected_tape))
            with self.assertRaises(Invalid):
                check_closed_states(machine, tuple(q for q in machine.states if q!='qH'))
            self.assertIn('no kernel-accepted',cases[name]['reference_note'])

    def test_old_datasets_still_validate_new_datasets_exclude_twelve(self):
        old = build_dataset(9,3,10,17,benchmark_suite='legacy')
        self.assertNotIn('benchmark_suite',old)
        self.assertEqual(len(old['excluded_benchmark_ids']),2)
        validate_dataset(json.loads(json.dumps(old)))
        new = build_dataset(1000,300)
        validate_dataset(json.loads(json.dumps(new)))
        self.assertEqual(len(new['excluded_benchmark_ids']),12)
        all_ids = {row['machine_id'] for split in ('train','test') for row in new[split]}
        self.assertFalse(all_ids & set(new['excluded_benchmark_ids']))

    def test_compact_prints_proof_and_saved_report_cli(self):
        rows = benchmark_rows('extended')
        results = [{**r,'attempts':2,'successes':[]} for r in rows]
        game = Game(Machine(rows[0]['machine']['rows']),budget=100)
        actions = ['choose:NONHALT','rule:exactCycle','cert:start:1','cert:period:2','cert:verify','submit']
        for action in actions: game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        results[0]['successes'] = [{'attempt':2,'certificate':game.certificate(),
                                  'actions':[{'action':a,'probability':.5} for a in actions]}]
        results[3]['checker_errors'] = {'definition reduction budget exceeded':3}
        report = make_benchmark_report(0,results,100,100,
                                      trained_ids=[rows[0]['machine_id']],suite='extended')
        text = format_compact_report(report)
        self.assertIn('VERIFIED SPECIAL PROOF',text)
        self.assertIn('cert:period:2',text)
        self.assertIn('Accepted proof tree:',text)
        self.assertIn('Observed checker resource limits',text)
        self.assertIn('not a held-out diagnostic',text)
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                write_benchmark_report(tmp,report,verbose=False)
            self.assertIn('Accepted proof tree:',output.getvalue())
            path = Path(tmp)/'special_benchmarks_epoch_00.json'
            shown = subprocess.run([sys.executable,'-m','proofgame.special_benchmarks','--report',str(path)],
                                   capture_output=True,text=True,timeout=30)
            self.assertEqual(shown.returncode,0,shown.stderr)
            self.assertIn('cert:period:2',shown.stdout)
            self.assertIn('exactCycle',shown.stdout)
            self.assertIn('Accepted proof tree:',(Path(tmp)/'special_benchmarks_epoch_00.txt').read_text())

    def test_checker_errors_counted_across_failed_attempts(self):
        from proofgame import epoch_train as et
        from types import SimpleNamespace
        def attempt(*args,diagnostic_steps=None,**kwargs):
            diagnostic_steps.append({'action':'rule:refl','error_after':'definition reduction budget exceeded'})
            return None, [], 'EXHAUSTED'
        row = benchmark_rows()[0]
        with patch.object(et,'attempt',side_effect=attempt):
            result = et.execute_machine(row,7,3,100,0,True,SimpleNamespace(hits=0,misses=0),
                                        trace_diagnostics=True)
        self.assertEqual(result['checker_errors'],{'definition reduction budget exceeded':3})

    def test_actual_success_captures_goal_and_candidate_details(self):
        from proofgame import epoch_train as et
        actions = iter(['choose:NONHALT','rule:exactCycle','cert:start:1',
                        'cert:period:2','cert:verify','submit'])
        class ScriptSampler:
            hits = misses = 0
            def probabilities(self, observation):
                action = next(actions)
                return [float(item['id']==action) for item in observation['actions']]
        rows = benchmark_rows()
        result = et.execute_machine(rows[0],7,100,100,0,True,ScriptSampler(),trace_diagnostics=True)
        self.assertEqual(result['attempts'],1)
        report = make_benchmark_report(0,[result,{**rows[1],'attempts':1,'successes':[]}],100,100)
        text = format_compact_report(report)
        self.assertIn('Goal:',text)
        self.assertIn('Candidate:',text)
        self.assertIn('Accepted proof tree:',text)
        self.assertEqual(len(report['benchmarks'][0]['sampled_proof_details']['steps']),6)

    def test_central_inference_specials_use_diagnostic_protocol(self):
        # The CPU central backend exercises the same worker/proxy protocol as
        # MPS without requiring Apple hardware.
        from proofgame.unfiltered_dataset import build_dataset as two_state_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'data.json'
            dataset.write_text(json.dumps(two_state_dataset(1,1,8,12345)))
            run = root/'run'
            cmd = [sys.executable,'-m','proofgame.epoch_train','--dataset',str(dataset),
                   '--out',str(run),'--epochs','1','--attempts','1','--workers','1',
                   '--representation','ast','--cpu-central-inference','--log-mode','compact']
            result = subprocess.run(cmd,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report = json.loads((run/'special_benchmarks_epoch_01.json').read_text())
            self.assertEqual(len(report['benchmarks']),12)
            self.assertTrue(all('checker_errors' in item for item in report['benchmarks']))
            self.assertNotIn('POLICY TRACE',result.stdout)


if __name__=='__main__': unittest.main()
