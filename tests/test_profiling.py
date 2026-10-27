"""Profiling is opt-in, diagnostic-only, and safe for ongoing checkpoints."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from proofgame.epoch_train import execute_machine
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank
from proofgame.profiling import make_batch_profile, summarize_epoch_profiles, save_batch_profile


class ProfilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_profiling_preserves_actions_and_probabilities(self):
        torch.manual_seed(11)
        policy=Policy(width=32,layers=1).eval()
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'profiling-fixture'}
        args=(row,42,3,10,0,False)
        plain=execute_machine(*args,FrozenSampler(policy))
        measured=execute_machine(*args,FrozenSampler(policy),profile_enabled=True)
        profile=measured.pop('profile')
        # Timings vary, but policy decisions, certificates, rewards, and
        # all saved standard result fields must be bitwise identical.
        plain.pop('seconds')
        measured.pop('seconds')
        self.assertEqual(measured,plain)
        self.assertEqual(profile['actions'],measured['actions'])
        self.assertEqual(profile['inference_calls'],measured['actions'])
        self.assertGreater(profile['inference_seconds'],0)
        self.assertGreater(profile['observation_seconds'],0)
        self.assertGreater(profile['step_seconds'],0)

    def test_parallel_worker_seconds_are_not_wall_seconds(self):
        rows=[
            {'actions':10,'attempts':2,'profile':{
                'observation_seconds':0.1,'inference_seconds':0.8,
                'step_seconds':0.05,'inference_calls':10,'actions':10}},
            {'actions':20,'attempts':4,'profile':{
                'observation_seconds':0.2,'inference_seconds':1.4,
                'step_seconds':0.08,'inference_calls':20,'actions':20}},
        ]
        stages={'rollout':1.5,'learn':0.2,'checkpoint':0.3}
        gpu_before={'inference_requests':10,'observation_cache_hits':2,
                    'gpu_context_batches':5,'inference_wall_seconds':0.5}
        gpu_after={'inference_requests':40,'observation_cache_hits':8,
                   'gpu_context_batches':9,'inference_wall_seconds':1.7}
        report=make_batch_profile(phase='train',epoch=1,batch_number=2,
                                  results=rows,stages=stages,device='mps',
                                  gpu_before=gpu_before,gpu_after=gpu_after)
        self.assertEqual(report['actions'],30)
        self.assertEqual(report['worker_counts']['inference_calls'],30)
        self.assertAlmostEqual(report['worker_sum_seconds']['inference_seconds'],2.2)
        self.assertEqual(report['stage_wall_seconds']['rollout'],1.5)
        self.assertEqual(report['central_inference_delta']['inference_requests'],30)
        with tempfile.TemporaryDirectory() as directory:
            path=save_batch_profile(directory,report)
            self.assertEqual(json.loads(path.read_text()),report)
            summary=summarize_epoch_profiles(directory,1,'train')
            self.assertEqual(summary['profiled_batches'],1)
            self.assertEqual(summary['actions'],30)

    def test_tiny_runner_profiles_baseline_and_training(self):
        from proofgame.unfiltered_dataset import build_dataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'dataset.json'
            dataset.write_text(json.dumps(build_dataset(2,1,4,12345)))
            out=root/'run'
            cmd=[sys.executable,'-m','proofgame.epoch_train',
                 '--dataset',str(dataset),'--out',str(out),
                 '--epochs','1','--attempts','2','--batch-size','1',
                 '--workers','1','--seed','7','--profile']
            completed=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
            self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)
            self.assertIn('"event": "batch_profile"',completed.stdout)
            for name in ('epoch_00_baseline_batch_0001.json',
                         'epoch_01_train_batch_0001.json',
                         'epoch_01_eval_batch_0001.json'):
                # The baseline phase folder is named eval, but profile keeps
                # the original phase name.
                path=out/'profiling'/name
                self.assertTrue(path.is_file(),str(path))
                report=json.loads(path.read_text())
                self.assertGreater(report['actions'],0)
                self.assertGreater(report['stage_wall_seconds']['rollout'],0)
                self.assertGreater(report['worker_sum_seconds']['inference_seconds'],0)
                self.assertIn('worker_model_load_seconds',report['worker_sum_seconds'])
            self.assertTrue((out/'profiling'/'epoch_01_train_summary.json').exists())
            batch=json.loads((out/'epoch_01_train'/'batch_0000.json').read_text())
            self.assertNotIn('profile',batch['machines'][0])
            self.assertNotIn('profile',json.loads((out/'config.json').read_text()))
            self.assertTrue((out/'state.pt').exists())


if __name__=='__main__':
    unittest.main()
