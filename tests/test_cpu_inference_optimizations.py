"""CPU optimizations must preserve proof semantics and invalidate stale features."""
import json
from collections import OrderedDict
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

from proofgame.epoch_train import init_worker, run_machine
from proofgame.environment import Game
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank
from proofgame.unfiltered_dataset import build_dataset


class CpuInferenceOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_action_features_identical_and_bounded(self):
        torch.manual_seed(9)
        policy=Policy(width=32,layers=1,representation='ast').eval()
        obs=Game(from_rank(8),budget=10).observation()
        features=OrderedDict()
        first=FrozenSampler(policy,capacity=0,shared_action_features=features,
                            action_cache_capacity=4)
        expected=first.probabilities(obs)
        self.assertEqual(first.action_feature_misses,len(obs['actions']))
        self.assertLessEqual(len(features),4)
        second=FrozenSampler(policy,capacity=0,shared_action_features=features,
                             action_cache_capacity=4)
        actual=second.probabilities(obs)
        self.assertEqual(expected,actual)
        self.assertGreater(second.action_feature_hits,0)
        self.assertLessEqual(len(features),4)

    def test_worker_reuses_features_only_within_version(self):
        torch.manual_seed(10)
        policy=Policy(width=32,layers=1,representation='ast').eval()
        machine=from_rank(8)
        row={'rank':8,'machine_id':'cache-control','machine':machine.to_json()}
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pt'
            init_worker(policy.config,share_action_cache=True)
            def run(version):
                torch.save({'version':version,'config':policy.config,
                            'state_dict':policy.state_dict()},path)
                return run_machine((row,123,1,8,str(path),version,False,
                                    0.,False,False,True))
            first=run(0)
            second=run(0)
            third=run(1)
            for key in ('actions','attempts','successes','statuses','first_action_counts'):
                self.assertEqual(first[key],second[key])
                self.assertEqual(first[key],third[key])
            self.assertGreater(second['profile']['action_feature_hits'],
                               first['profile']['action_feature_hits'])
            self.assertLess(second['profile']['action_feature_misses'],
                            first['profile']['action_feature_misses'])
            self.assertEqual(third['profile']['action_feature_misses'],
                             first['profile']['action_feature_misses'])
            # Reinitialize the global worker to avoid affecting other tests.
            init_worker(policy.config,share_action_cache=False)

    def test_benchmark_reports_all_three_modes(self):
        from proofgame.benchmark_cpu_inference import compare_modes
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1).eval()
        dataset=build_dataset(2,2,4,42)
        results=compare_modes(dataset,policy,machines=2,attempts=2,workers=2,
                              batch_size=2,inference_batch_size=2,seed=123)
        self.assertEqual([x['mode'] for x in results],
                         ['baseline','shared-cache','central'])
        self.assertTrue(all(x['attempts']==4 for x in results))
        self.assertTrue(all(x['actions']>0 and x['actions_per_second']>0
                            for x in results))
        self.assertEqual(results[0]['actions'],results[1]['actions'])
        self.assertEqual(results[0]['proved_machines'],results[1]['proved_machines'])
        self.assertTrue(all(x['throughput_vs_baseline']>0 for x in results))

    def test_central_batched_cpu_probs_remain_on_policy_for_gradients(self):
        from proofgame.epoch_train import learn_batch
        from proofgame.gpu_runtime import BatchedPolicyInference
        from proofgame.network_search import encode_json
        torch.manual_seed(12)
        policy=Policy(width=32,layers=1,representation='ast').eval()
        optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4)
        machine=from_rank(8)
        game=Game(machine,budget=20)
        other=Game(from_rank(9),budget=20)
        inference=BatchedPolicyInference(policy,batch_size=2)
        inference.begin_version(0)
        actions=['choose:HALT','rule:exI','put:s','put:z',
                 'commit','rule:refl','submit']
        trace=[]
        for action in actions:
            obs=game.observation()
            # Include another context to exercise genuinely batched CPU
            # numerical inference, rather than just a single-context call.
            reply=inference.infer([encode_json(obs),encode_json(other.observation())])
            self.assertEqual(reply[0][0],'ok')
            index=next(i for i,a in enumerate(obs['actions']) if a['id']==action)
            trace.append({'action':action,'probability':reply[0][1][index]})
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        row={'machine':machine.to_json(),'attempts':1,
             'successes':[{'actions':trace,'certificate':game.certificate()}]}
        update=learn_batch(policy,optimizer,[row],20)
        self.assertTrue(update['updated'])

    def test_opt_in_cpu_topology_cache_and_two_pass_cli(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'data.json'
            dataset.write_text(json.dumps(build_dataset(2, 1, 8, 777)))
            output = root / 'train'
            args = [
                sys.executable, '-m', 'proofgame.epoch_train',
                '--dataset', str(dataset), '--out', str(output),
                '--epochs', '1', '--attempts', '2', '--batch-size', '2',
                '--workers', '2', '--seed', '33',
                '--device', 'cpu', '--representation', 'ast',
                '--group-relative', '--balanced-initial-answers',
                '--cpu-shared-action-cache', '--cpu-topology-cache-mb', '2',
                '--gradient-backend', 'two-pass',
                '--gpu-gradient-batch-size', '2', '--profile',
            ]
            completed = subprocess.run(
                args, capture_output=True, text=True, timeout=180)
            self.assertEqual(
                completed.returncode, 0,
                completed.stdout + '\n' + completed.stderr)
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual(config['cpu_topology_cache_mb'], 2)
            self.assertEqual(config['gradient_execution']['backend'], 'two-pass')
            self.assertEqual(config['gradient_execution']['context_batch_size'], 2)
            profile = json.loads(
                (output / 'profiling' /
                 'epoch_01_train_batch_0001.json').read_text())
            self.assertIn('topology_cache_hits', profile['worker_counts'])
            self.assertIn('topology_cache_misses', profile['worker_counts'])
            # Performance choices are part of a reproducible run's config.
            bad = subprocess.run(
                args + ['--resume', '--cpu-topology-cache-mb', '3'],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('resume configuration changed', bad.stderr)
            invalid = subprocess.run(
                args + ['--mps-local-cpu-rollouts'],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('requires --device mps', invalid.stderr)

    def test_central_cpu_cli_runs_and_profiles(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'dataset.json'
            dataset.write_text(json.dumps(build_dataset(2,1,4,44)))
            out=root/'run'
            cmd=[sys.executable,'-m','proofgame.epoch_train',
                 '--dataset',str(dataset),'--out',str(out),
                 '--epochs','1','--attempts','2','--batch-size','2',
                 '--workers','2','--seed','7',
                 '--device','cpu','--cpu-central-inference',
                 '--gpu-inference-batch-size','2','--profile']
            completed=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)
            self.assertIn('"inference": "centralized-batched"',completed.stdout)
            self.assertIn('"event": "batch_profile"',completed.stdout)
            config=json.loads((out/'config.json').read_text())
            self.assertEqual(config['cpu_inference'],'central-batched')
            self.assertTrue((out/'epoch_01_evaluation.json').is_file())
            self.assertTrue((out/'profiling'/'epoch_01_train_summary.json').is_file())


if __name__=='__main__':
    unittest.main()
