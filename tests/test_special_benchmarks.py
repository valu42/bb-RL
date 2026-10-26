"""The two special benchmarks are evaluation-only and human-readable."""
import json
import importlib.util
import subprocess
import sys
from pathlib import Path
import random
import tempfile
import unittest

from proofgame.environment import Game
from proofgame.logic import Machine
from proofgame.special_benchmarks import (
    BENCHMARKS, benchmark_rows, make_benchmark_report,
    format_benchmark_report, write_benchmark_report,
)


class SpecialBenchmarkTests(unittest.TestCase):
    def test_fixtures_and_default_dataset_nonoverlap(self):
        rows = benchmark_rows()
        self.assertEqual(len(rows),2)
        self.assertEqual(len({r['machine_id'] for r in rows}),2)
        root=Path(__file__).resolve().parents[1]/'examples'/'interesting'
        for definition,row in zip(BENCHMARKS,rows):
            fixture=json.loads((root/(definition['id']+'.json')).read_text())
            self.assertEqual(json.loads(json.dumps(row['machine'])),fixture)
            self.assertEqual(row['machine'],Machine(definition['rows']).to_json())
        # Current standard seeded unfiltered split should contain neither.
        from proofgame.unfiltered_dataset import build_dataset
        ds=build_dataset(1000,100,100,20261026)
        ids={r['machine_id'] for split in ('train','test') for r in ds[split]}
        self.assertTrue(all(row['machine_id'] not in ids for row in rows))

    def test_failure_and_verified_busy_beaver_success(self):
        parity,busy=benchmark_rows()
        machine=Machine(busy['machine']['rows'])
        game=Game(machine,budget=100)
        actions=(['choose:HALT','rule:exI']+['put:s']*6+
                 ['put:z','commit','rule:refl','submit'])
        trace=[]
        for action in actions:
            game.step(action,observe=False)
            trace.append({'action':action,'probability':1.0})
        self.assertEqual(game.status,'SOLVED')
        fail={**parity,'successes':[],'attempts':100}
        success={**busy,'successes':[{'attempt':9,'actions':trace,
                                     'certificate':game.certificate()}],
                 'attempts':9}
        report=make_benchmark_report(1,[fail,success],100,100)
        a,b=report['benchmarks']
        self.assertEqual(a['status'],'NOT_PROVED')
        self.assertEqual(b['status'],'PROVED')
        self.assertEqual(b['proved_answer'],'HALT')
        self.assertEqual(b['first_proof_attempt'],9)
        self.assertEqual(b['proof_actions'],12)
        text=format_benchmark_report(report)
        self.assertIn('Parity trap (2-state, NONHALT): NOT PROVED',text)
        self.assertIn('Busy Beaver (2-state, HALT in 6 steps): PROVED HALT',text)
        self.assertIn('attempt 9/100',text)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            write_benchmark_report(path,report)
            self.assertEqual(json.loads((path/'special_benchmarks_epoch_01.json').read_text()),
                             json.loads(json.dumps(report)))
            self.assertEqual((path/'special_benchmarks_epoch_01.txt').read_text(),text)
            self.assertEqual((path/'special_benchmarks_history.txt').read_text(),text)
            baseline=make_benchmark_report(0,[fail,{**busy,'attempts':100,'successes':[]}],100,100)
            write_benchmark_report(path,baseline)
            history=(path/'special_benchmarks_history.txt').read_text()
            self.assertLess(history.index('baseline (epoch 0)'),history.index('after epoch 1'))
            self.assertEqual(history.count('Special benchmarks'),2)

    def test_invalid_or_forged_results_are_rejected(self):
        rows=benchmark_rows()
        failed=[{**row,'attempts':1,'successes':[]} for row in rows]
        with self.assertRaises(ValueError):
            make_benchmark_report(0,failed[:1],100,100)
        with self.assertRaises(ValueError):
            make_benchmark_report(0,[{**failed[0],'machine_id':'wrong'},failed[1]],100,100)
        with self.assertRaises(ValueError):
            make_benchmark_report(0,[{**failed[0],'attempts':101},failed[1]],100,100)
        with self.assertRaises(Exception):
            make_benchmark_report(0,[failed[0],{**failed[1],
              'successes':[{'attempt':1,'actions':[],
                            'certificate':{'answer':'HALT','machine':rows[1]['machine']}}]}],100,100)


    @unittest.skipUnless(importlib.util.find_spec('torch'),
                         'requires optional training dependency')
    def test_runner_writes_baseline_and_epoch_diagnostics(self):
        from proofgame.unfiltered_dataset import build_dataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'tiny_dataset.json'
            dataset.write_text(json.dumps(build_dataset(1,1,8,12345)))
            output=root/'experiment'
            command=[
                sys.executable,'-m','proofgame.epoch_train',
                '--dataset',str(dataset),'--out',str(output),
                '--epochs','1','--attempts','1','--batch-size','1',
                '--workers','1','--seed','7','--stop-after-first-train-proof'
            ]
            finished=subprocess.run(command,capture_output=True,text=True,timeout=120)
            self.assertEqual(finished.returncode,0,
                             f"{finished.stdout}\n{finished.stderr}")
            self.assertIn('Special benchmarks — baseline (epoch 0)',finished.stdout)
            self.assertIn('Special benchmarks — after epoch 1',finished.stdout)
            self.assertIn('Epoch 1 evaluation: machines solved',finished.stdout)
            self.assertLess(finished.stdout.index('Baseline (before training):'),
                            finished.stdout.index('"phase": "train"'))
            import csv
            with (output/'evaluations.csv').open() as stream:
                evaluation_rows=list(csv.DictReader(stream))
            self.assertEqual([row['epoch'] for row in evaluation_rows],['0','1'])
            for epoch in (0,1):
                special=json.loads((output/f'special_benchmarks_epoch_{epoch:02d}.json').read_text())
                self.assertEqual(len(special['benchmarks']),2)
                normal=json.loads((output/f'epoch_{epoch:02d}_evaluation.json').read_text())
                self.assertEqual(normal['machines'],1,
                                 'two extra diagnostics must not inflate held-out metric')
                self.assertEqual(normal['machines_solved_pct'],100*normal['machine_coverage'])
                self.assertEqual(normal['attempt_success_pct'],100*normal['attempt_success_rate'])
            finished_report=json.loads((output/'report.json').read_text())
            self.assertTrue(finished_report['complete'])
            self.assertEqual(len(finished_report['special_benchmarks']),2)


if __name__=='__main__':
    unittest.main()

