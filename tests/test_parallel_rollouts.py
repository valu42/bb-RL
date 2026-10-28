"""Production defaults, schedule-independent sampling, and complete round groups."""
import hashlib
import json
import subprocess
import sys
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from proofgame import epoch_train as et
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler, encode_json
from proofgame.parallel_rollouts import (attempt_seed, branch_schedule,
                                        reduce_attempts, run_local_rollouts)
from proofgame.population_sample import from_rank
from proofgame.profiling import make_batch_profile


class ParallelRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def task(self, weights='unused', version=0, attempts=2, rounds=3, seed=7):
        row={'machine':from_rank(8).to_json(),'machine_id':'parallel-fixture'}
        return (row,seed,attempts,6,str(weights),version,False,0.,True,True,True,True,rounds)

    def test_completion_order_and_late_success_keep_whole_winning_round(self):
        task=self.task()
        calls=[]
        def fake(job):
            _, index, answer=job
            time.sleep(.003*(3-index%4))
            calls.append(index)
            # Only the second round has proofs. Every job in both branches
            # must finish; no third-round job should ever be dispatched.
            cert={'answer':answer} if index==5 else None
            trace=[{'action':'choose:'+answer,'probability':1.,
                    'forced_initial_answer':answer}]
            return {'certificate':cert,'trace':trace,'status':'SOLVED' if cert else 'EXHAUSTED',
                    'initial':[('choose:'+answer,{'HALT':.5,'NONHALT':.5},
                                {'HALT':float(answer=='HALT'),'NONHALT':float(answer=='NONHALT')})],
                    'detail':[],'profile':{},'hits':0,'misses':1,'seconds':.01}
        with ThreadPool(4) as pool, patch('proofgame.parallel_rollouts.run_one_attempt',side_effect=fake):
            result=run_local_rollouts(pool,[task],workers=4,split_attempts=True)[0]
        self.assertEqual(sorted(calls),list(range(8)))
        self.assertEqual(result['attempts'],8)
        self.assertEqual(result['first_proof_round'],2)
        self.assertEqual(result['rounds_completed'],2)
        self.assertEqual(len(result['group_trajectories']),2)
        self.assertEqual(result['attempts_by_branch'],{'HALT':4,'NONHALT':4})
        self.assertEqual(result['successes'][0]['attempt'],6)
        self.assertEqual(result['first_proof_answer'],branch_schedule(7,2,3)[5][1])

    def test_real_spawn_matches_serial_reference_with_independent_streams(self):
        torch.manual_seed(14)
        model=Policy(width=32,layers=1,representation='ast',encoder_backend='root-query').eval()
        task=self.task(rounds=2)
        serial=et.execute_machine(
            task[0],task[1],task[2],task[3],0,False,FrozenSampler(model),
            balanced_initial_answers=True,trace_diagnostics=True,profile_enabled=True,
            group_relative=True,branch_rounds=2,independent_attempt_seeds=True)
        with tempfile.TemporaryDirectory() as tmp:
            weights=Path(tmp)/'snapshot.pt'
            torch.save({'version':0,'state_dict':model.state_dict()},weights)
            task=self.task(weights,rounds=2)
            for workers in (1,2):
                with mp.get_context('spawn').Pool(workers,initializer=et.init_worker,
                                                initargs=(model.config,True)) as pool:
                    actual=run_local_rollouts(pool,[task],workers=workers,split_attempts=True)[0]
                # Caches/timings are schedule-dependent; action streams,
                # probabilities, certificates and groups must be identical.
                for name in ('successes','group_trajectories','statuses','actions','attempts',
                             'first_proof_round','first_proof_answer','rounds_completed',
                             'attempts_by_branch','first_action_counts','diagnostic_traces',
                             'trajectory_sha256'):
                    self.assertEqual(actual[name],serial[name],name)

    def test_snapshot_version_invalidates_all_neural_caches(self):
        model=Policy(width=32,layers=1,representation='ast',encoder_backend='root-query')
        with tempfile.TemporaryDirectory() as tmp:
            weights=Path(tmp)/'snapshot.pt'
            et.init_worker(model.config,True)
            torch.save({'version':0,'state_dict':model.state_dict()},weights)
            et.load_worker_snapshot(weights,0)
            et._worker_action_features['old']='stale'
            et._worker_samplers['old']='stale'
            et.load_worker_snapshot(weights,0)
            self.assertIn('old',et._worker_samplers)
            torch.save({'version':1,'state_dict':model.state_dict()},weights)
            et.load_worker_snapshot(weights,1)
            self.assertFalse(et._worker_samplers)
            self.assertFalse(et._worker_action_features)

    def test_errors_propagate_and_mixed_versions_are_rejected(self):
        with ThreadPool(2) as pool:
            with self.assertRaisesRegex(ValueError,'one frozen'):
                run_local_rollouts(pool,[self.task(),self.task(version=1)],workers=2)
            with patch('proofgame.parallel_rollouts.run_one_attempt',side_effect=RuntimeError('proof failure')):
                with self.assertRaisesRegex(RuntimeError,'proof failure'):
                    run_local_rollouts(pool,[self.task()],workers=2,split_attempts=True)

    def test_runtime_backend_checkpoint_and_full_signed_gradients(self):
        torch.manual_seed(42)
        full=Policy(width=32,layers=1,representation='ast').eval()
        root=Policy(**{**full.config,'encoder_backend':'root-query'}).eval()
        root.load_state_dict(full.state_dict())
        # Test the actual persisted config route used by main and workers.
        restored=Policy(**root.config)
        restored.load_state_dict(root.state_dict())
        self.assertTrue(restored.structural_encoder.use_root_query)
        texts=[encode_json({'goal':'(all Nat (eq (bound 0) (bound 0)))'}),encode_json({'id':'reset'})]
        with torch.no_grad():
            full.structural_encoder.relationship.weight.uniform_(-.5,.5)
        root.load_state_dict(full.state_dict())
        for model in (full,root):
            (model.encode(texts)*torch.tensor([1.,-.3])[:,None]).sum().backward()
        for (name,p),(_,q) in zip(full.named_parameters(),root.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=8e-5,msg=name)
        with self.assertRaises(ValueError):
            root.structural_encoder.enable_input_cache()

    def test_root_query_two_pass_group_update_matches_full_serial(self):
        from test_branch_rounds import winning_group
        from proofgame.benchmark_speed_v2 import _compare_vectors
        torch.manual_seed(17)
        full=Policy(width=32,layers=1,representation='ast').eval()
        root=Policy(**{**full.config,'encoder_backend':'root-query'}).eval()
        root.load_state_dict(full.state_dict())
        row=winning_group(full,budget=12)
        reports=[]
        for model,backend in ((full,'serial'),(root,'two-pass')):
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
            reports.append(et.learn_batch(
                model,optimizer,[row],12,group_relative=True,
                balanced_initial_answers=True,branch_rounds=6,
                length_penalty=.3,gradient_backend=backend,gpu_gradient_batch_size=8))
        self.assertTrue(all(r['updated'] for r in reports))
        self.assertEqual(reports[0]['group_negative_advantage_attempts'],7)
        error=_compare_vectors(full,root)
        self.assertLess(error['relative_gradient_l2'],1e-4)
        self.assertLess(error['max_absolute_parameter_difference'],6e-4)
        self.assertIn('timing_optimizer_seconds',reports[1])

    def test_read_only_runtime_benchmark_cli_reports_no_gradient_work(self):
        from proofgame.unfiltered_dataset import build_dataset
        model=Policy(width=32,layers=1,representation='ast',encoder_backend='root-query')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'model_epoch_02.pt'
            torch.save({'config':model.config,'state_dict':model.state_dict()},checkpoint)
            before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (root/'dataset.json').write_text(json.dumps(build_dataset(1,1,1,22)))
            (root/'config.json').write_text(json.dumps({
                'seed':7,'attempts_per_machine':2,'branch_rounds':1,
                'group_relative':True,'lr':3e-4}))
            output=root/'benchmark.json'
            cmd=[sys.executable,'-m','proofgame.benchmark_training_runtime',
                 '--run',str(root),'--machines','1','--workers','1',
                 '--repeats','1','--out',str(output)]
            completed=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['status'],'complete')
            self.assertTrue(report['trials'][0]['same_trajectories'])
            self.assertFalse(report['full_gradient_update_exercised'])
            self.assertIsNone(report['validated_speedup'])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),before)
            again=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(again.returncode,0)

    def test_profile_separates_worker_sums_and_stage_wall_percent(self):
        profile=make_batch_profile(
            phase='train',epoch=1,batch_number=1,device='cpu',workers=4,
            results=[{'attempts':2,'actions':10,'worker_task_seconds':6.,
                      'profile':{'inference_seconds':5.,'cache_hits':3,'cache_misses':7}}],
            stages={'rollout':2.,'learn':1.,'checkpoint':1.},
            learning={'two_pass_unique_actions':12,'timing_backward_seconds':.8})
        self.assertEqual(profile['elapsed_worker_occupancy'],.75)
        self.assertEqual(profile['stage_wall_percent']['rollout'],50.)
        self.assertEqual(profile['observation_cache_hit_rate'],.3)
        self.assertEqual(profile['learning']['two_pass_unique_actions'],12)
        self.assertEqual(attempt_seed(7,3),attempt_seed(7,3))
        self.assertNotEqual(attempt_seed(7,3),attempt_seed(7,4))


if __name__=='__main__':
    unittest.main()
