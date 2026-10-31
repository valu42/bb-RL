import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from proofgame.foundation.__main__ import parser
from proofgame.foundation.schedule import pretrain_panel, rl_panel, training_config


class ScheduleTests(unittest.TestCase):
    def test_defaults_and_fixed_balanced_panels(self):
        args = parser().parse_args(['run', '--out', 'unused'])
        cfg = {**vars(args), 'budget': 512}
        dev = [{'states': n, 'machine_id': f'{n}-{i}', 'machine': {}}
               for n in (2, 3, 4, 5) for i in range(100)]
        rows, pre = pretrain_panel(cfg, dev)
        self.assertEqual((len(rows), pre['attempts_per_machine'], args.eval_every), (40, 8, 2000))
        baseline, quick = rl_panel(cfg, dev, 0)
        for epoch in (1, 2, 9, 11, 99):
            self.assertEqual(rl_panel(cfg, dev, epoch), (baseline, quick))
        self.assertEqual((len(baseline), quick['attempts_per_machine']), (100, 8))
        for n in (2, 3, 4, 5):
            self.assertEqual(sum(r['states'] == n for r in baseline), 25)
        for epoch in (10, 20, 100):
            rows, full = rl_panel(cfg, dev, epoch)
            self.assertEqual(rows, dev)
            self.assertEqual((full['kind'], full['attempts_per_machine']), ('full', 100))
        legacy = {'budget': 512, 'eval_machines': 40, 'eval_attempts': 100}
        self.assertEqual(pretrain_panel(legacy, dev)[1]['attempts_per_machine'], 100)
        self.assertEqual(rl_panel(legacy, dev, 0)[1]['kind'], 'full')

    def test_only_quick_evaluation_and_schedule_settings_are_mutable(self):
        cfg = vars(parser().parse_args(['run', '--out', 'unused']))
        for key in ('eval_every', 'eval_machines', 'pretrain_eval_attempts',
                    'rl_eval_machines', 'rl_eval_attempts', 'full_eval_every'):
            changed = {**cfg, key: cfg[key]+1}
            self.assertEqual(training_config(cfg), training_config(changed))
        for key in ('budget', 'width', 'lr', 'attempts', 'rounds', 'eval_attempts', 'seed'):
            self.assertNotEqual(training_config(cfg), training_config({**cfg, key: cfg[key]+1}))


class InlinePool:
    """Exercise phase scheduling and real persistence without slow proof search."""
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def map(self, function, tasks): return map(function, tasks)


@unittest.skipIf(torch is None, 'torch is optional')
class ScheduleResumeTests(unittest.TestCase):
    def setUp(self):
        from proofgame.foundation.compiler import compile_halt, replay
        from proofgame.foundation.data import atomic, digest, hash_file, language_hash, VERSION
        from proofgame.foundation.logic import Machine
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.data = self.root/'data'; self.data.mkdir()
        machine = Machine((('qH', 'b0', 'R'), ('qH', 'b1', 'R'))*2)
        self.row = {'machine': machine.to_json(), 'machine_id': digest(machine.to_json()), 'states': 2}
        cert = compile_halt(machine, 1)
        example = {'source': 'halt', 'answer': 'HALT', 'proof_id': digest(cert),
                   'certificate': cert, 'actions': replay(cert, budget=64)}
        (self.data/'corpus.jsonl').write_text(json.dumps({**self.row, 'examples': [example]})+'\n')
        dev = [{**self.row, 'machine_id': str(i)} for i in range(4)]
        self.machines = {'splits': {'train': dev[:2], 'dev': dev, 'test': dev, 'special': dev[:1]}}
        atomic(self.data/'machines.json', self.machines)
        atomic(self.data/'manifest.json', {'version': VERSION, 'source_hash': digest(self.machines),
               'config': {'budget': 64, 'language_hash': language_hash()},
               'corpus_sha256': hash_file(self.data/'corpus.jsonl')})

    def pre_args(self, name, *extra):
        return parser().parse_args(['pretrain', '--data', str(self.data), '--out', str(self.root/name),
            '--updates', '1', '--batch-size', '1', '--width', '16', '--layers', '1',
            '--eval-machines', '2', '--workers', '1', *extra])

    def rl_args(self, name, *extra):
        return parser().parse_args(['train', '--data', str(self.data), '--out', str(self.root/name),
            '--epochs', '2', '--batch-size', '1', '--width', '16', '--layers', '1',
            '--rl-eval-machines', '2', '--rl-eval-attempts', '2', '--full-eval-every', '2',
            '--eval-attempts', '3', '--attempts', '2', '--rounds', '1', *extra])

    def test_legacy_pending_pretraining_migrates_without_repeating_update(self):
        from proofgame.foundation.training import pretrain, load, save, init
        args = self.pre_args('pre')
        with patch('proofgame.foundation.training.autonomous', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): pretrain(args)
        path = self.root/'pre/checkpoint.pt'; state = load(path)
        # Simulate the exact old config shape and its 100-attempt score.
        state['config'].pop('pretrain_eval_attempts')
        state['config']['eval_every'] = 1000
        state['pending_evaluation'].pop('evaluation')
        state['best'] = 40
        save(path, state); save(path.with_name('best.pt'), state)
        seen = []
        def evaluate(policy, machines, budget, attempts, seed, **kwargs):
            seen.append((attempts, len(machines)))
            return {'machines_proved_by_answer': {'HALT': 1, 'NONHALT': 0}}
        args.resume = True
        with patch('proofgame.foundation.training.autonomous', side_effect=evaluate):pretrain(args)
        resumed = load(path)
        self.assertEqual(seen, [(8, 2)])
        self.assertEqual(resumed['update'], 1)
        self.assertEqual(resumed['best'], 1)
        self.assertEqual(load(path.with_name('best.pt'))['best'], 1)
        self.assertEqual(len(list(path.parent.glob('best_before_eval_change_*.pt'))), 1)
        self.assertNotIn('pending_evaluation', resumed)
        for k, v in state['state_dict'].items():self.assertTrue(torch.equal(v, resumed['state_dict'][k]))
        self.assertEqual(state['optimizer']['param_groups'], resumed['optimizer']['param_groups'])
        for key, values in state['optimizer']['state'].items():
            for name, value in values.items():self.assertTrue(torch.equal(value, resumed['optimizer']['state'][key][name]))
        for key, value in [('lr', .01), ('width', 32), ('eval_attempts', 2)]:
            changed = copy.copy(args); setattr(changed, key, value)
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):init(changed, 'pretrain')

    def test_completed_pretraining_rechecks_new_budget_not_old_best_score(self):
        from proofgame.foundation.training import pretrain, load
        def evaluate(*args, **kwargs):return {'machines_proved_by_answer': {'HALT': 0, 'NONHALT': 0}}
        args = self.pre_args('pre')
        with patch('proofgame.foundation.training.autonomous', side_effect=evaluate):pretrain(args)
        before = load(self.root/'pre/checkpoint.pt')
        args.resume = True; args.pretrain_eval_attempts = 4
        with patch('proofgame.foundation.training.autonomous', side_effect=evaluate) as mocked:pretrain(args)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[3], 4)
        after = load(self.root/'pre/checkpoint.pt')
        for k, v in before['state_dict'].items():self.assertTrue(torch.equal(v, after['state_dict'][k]))

    def test_run_resume_does_not_rewrite_consumed_pretrained_checkpoint(self):
        from proofgame.foundation.__main__ import main
        pre = self.root/'pretrain'; pre.mkdir()
        (pre/'checkpoint.pt').write_bytes(b'checkpoint placeholder')
        (pre/'best.pt').write_bytes(b'original pretrained weights')
        rl = self.root/'rl'; rl.mkdir(); (rl/'checkpoint.pt').write_bytes(b'RL checkpoint placeholder')
        argv = ['foundation', 'run', '--out', str(self.root), '--resume', '--updates', '1']
        with patch('sys.argv', argv), patch('proofgame.foundation.data.generate'), \
             patch('proofgame.foundation.training.load', return_value={'update': 1}), \
             patch('proofgame.foundation.training.pretrain', side_effect=AssertionError('must preserve consumed weights')), \
             patch('proofgame.foundation.training.train') as trained:
            main()
        self.assertEqual(trained.call_count, 1)
        self.assertTrue(trained.call_args.args[0].resume)
        self.assertEqual((pre/'best.pt').read_bytes(), b'original pretrained weights')

    def fake_job(self, task):
        snapshot, version, row, budget, seed, attempts, rounds, training = task
        self.calls.append((version, row['machine_id'], attempts, training))
        return {**row, 'successes': [], 'group': [], 'actions': 0, 'attempts': attempts,
                'statuses': {}, 'timing': {'inference_seconds': 0., 'game_seconds': 0., 'observation_seconds': 0.}}

    def run_rl(self, args, job=None):
        from proofgame.foundation.training import train
        with patch('proofgame.foundation.training.concurrent.futures.ProcessPoolExecutor', InlinePool), \
             patch('proofgame.foundation.training.machine_job', side_effect=job or self.fake_job):train(args)

    def test_rl_quick_full_special_and_final_test_budgets(self):
        self.calls = []
        args = self.rl_args('rl'); self.run_rl(args)
        root = self.root/'rl'
        for epoch, kind, count, attempts in [(0, 'quick', 2, 2), (1, 'quick', 2, 2), (2, 'full', 4, 3)]:
            report = json.loads((root/f'epoch_{epoch:02d}_evaluation.json').read_text())
            self.assertEqual((report['evaluation']['kind'], report['machines'], report['evaluation']['attempts_per_machine']), (kind, count, attempts))
        self.assertFalse((root/'special_benchmarks_epoch_00.json').exists())
        self.assertFalse((root/'special_benchmarks_epoch_01.json').exists())
        self.assertTrue((root/'special_benchmarks_epoch_02.json').exists())
        final = json.loads((root/'final_test_epoch_02.json').read_text())
        self.assertEqual(final['evaluation']['attempts_per_machine'], 3)
        self.assertEqual(final['summary']['machines'], 4)
        self.assertEqual([c[0] for c in self.calls if c[0][0]=='special'], [('special', 2)])
        self.assertEqual(len([c for c in self.calls if c[3]]), 4)
        self.assertTrue(all(c[2]==2 for c in self.calls if c[3]))
        self.calls = []; args.resume = True; self.run_rl(args)
        self.assertEqual(self.calls, [])

    def test_legacy_inflight_eval_restarts_and_training_progress_survives(self):
        from proofgame.foundation.training import init, checkpoint, save, load
        args = self.rl_args('rl', '--epochs', '0', '--skip-final-test')
        out, cfg, machines, policy, opt, _ = init(args, 'rl')
        old = {k: v for k, v in cfg.items() if k not in ('rl_eval_machines', 'rl_eval_attempts', 'full_eval_every')}
        save(out/'checkpoint.pt', checkpoint(policy, opt, old, phase_index=0, next_index=1, updates=7))
        self.calls = []; args.resume = True; self.run_rl(args)
        self.assertEqual([c[1] for c in self.calls], ['0', '1'])
        self.assertTrue(all(c[2]==2 for c in self.calls))
        self.assertEqual(load(out/'checkpoint.pt')['updates'], 7)
        # Interruption in an ordinary training batch retains its committed work.
        args = self.rl_args('partial', '--epochs', '1', '--skip-final-test')
        self.calls = []
        def interrupt(task):
            if task[1] == (1, 1):raise KeyboardInterrupt
            return self.fake_job(task)
        with self.assertRaises(KeyboardInterrupt):self.run_rl(args, interrupt)
        state = load(self.root/'partial/checkpoint.pt')
        self.assertEqual((state['phase_index'], state['next_index']), (1, 1))
        self.calls = []; args.resume = True; args.rl_eval_attempts = 1
        self.run_rl(args)
        self.assertEqual([c[1] for c in self.calls if c[3]], ['0'])  # seeded shuffled train order


if __name__ == '__main__':unittest.main()
