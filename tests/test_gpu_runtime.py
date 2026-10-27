"""Central batched inference and per-outcome reporting regression tests."""
import importlib.util
import unittest
from pathlib import Path
import tempfile
import json

if importlib.util.find_spec('torch') is not None:
    import torch
    from proofgame.environment import Game
    from proofgame.epoch_train import count_proofs_by_answer, execute_machine, learn_batch, summarize, summarize_rows
    from proofgame.gpu_runtime import BatchedPolicyInference, CentralRolloutPool
    from proofgame.model import Policy
    from proofgame.network_search import FrozenSampler, encode_json
    from proofgame.population_sample import from_rank
    from proofgame.tasks import key
else:
    torch = None


@unittest.skipUnless(torch is not None, 'optional PyTorch dependency is not installed')
class OutcomeLoggingTests(unittest.TestCase):
    def test_split_counts_proofs_and_distinct_machines(self):
        def machine(answer, proof_count, attempts):
            return {'attempts': attempts, 'actions': 10, 'statuses': {'SOLVED': proof_count},
                    'successes': [
                        {'attempt': i+1, 'certificate': {'answer': answer}}
                        for i in range(proof_count)]}
        rows=[machine('HALT', 3, 100),machine('HALT', 1, 12),
              machine('NONHALT', 2, 20),machine('HALT', 0, 100)]
        report=summarize_rows(rows)
        self.assertEqual(report['machines_proved_by_answer'],{'HALT':2,'NONHALT':1})
        self.assertEqual(report['successful_attempts_by_answer'],{'HALT':4,'NONHALT':2})
        self.assertEqual(report['machines_unproved'],1)
        self.assertEqual(report['machines_proved'],3)
        self.assertEqual(report['successful_attempts'],6)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'batch.json'
            p.write_text(json.dumps({'machines':rows}))
            self.assertEqual(summarize([p])['machines_proved_by_answer'],report['machines_proved_by_answer'])

    def test_rejects_contradictory_results(self):
        row={'successes':[{'attempt':1,'certificate':{'answer':'HALT'}},
                          {'attempt':2,'certificate':{'answer':'NONHALT'}}]}
        with self.assertRaises(ValueError):
            count_proofs_by_answer([row])


@unittest.skipUnless(torch is not None, 'optional PyTorch dependency is not installed')
class CentralGPUInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batched_probs_cache_version_and_action_order(self):
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1,representation='ast').eval()
        game=Game(from_rank(8),budget=20)
        first=game.observation()
        game.step('choose:HALT',observe=False)
        second=game.observation()
        inference=BatchedPolicyInference(policy,batch_size=2)
        inference.begin_version(0)
        answers=inference.infer([encode_json(first),encode_json(second),encode_json(first)])
        self.assertEqual([a[0] for a in answers],['ok','ok','ok'])
        self.assertEqual(answers[0][1],answers[2][1])
        reference=FrozenSampler(policy)
        for obs,(_,p) in zip((first,second),answers[:2]):
            torch.testing.assert_close(torch.tensor(p),torch.tensor(reference.probabilities(obs)),rtol=1e-3,atol=1e-4)
        self.assertEqual(inference.infer([encode_json(first)])[0],answers[0])
        self.assertGreater(inference.cache_hits,0)
        inference.begin_version(1)
        self.assertEqual(len(inference.states),0)
        self.assertEqual(len(inference.actions),0)

    def test_central_cpu_workers_match_direct_rollout(self):
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1).eval()
        machine=from_rank(8)
        row={'rank':8,'machine_id':key(machine),'machine':machine.to_json()}
        tasks=[(row,12345,2,9,0,True,0.,False,False,False),(row,54321,2,9,0,True,0.2,False,False,False)]
        expected=[execute_machine(r,seed,attempts,budget,version,stop,FrozenSampler(policy),epsilon)
                  for r,seed,attempts,budget,version,stop,epsilon,balanced,diagnostics,profiling in tasks]
        with CentralRolloutPool(policy,workers=2,inference_batch_size=2,wait_ms=1.) as pool:
            actual=pool.run(tasks)
        for left,right in zip(actual,expected):
            for field in ('attempts','actions','statuses','successes','policy_version'):
                self.assertEqual(left[field],right[field])

    def test_central_worker_profiling_is_reported(self):
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1).eval()
        machine=from_rank(8)
        row={'rank':8,'machine_id':key(machine),'machine':machine.to_json()}
        task=(row,12345,2,9,0,True,0.,False,False,True)
        with CentralRolloutPool(policy,workers=1,inference_batch_size=2,wait_ms=1.) as pool:
            result=pool.run([task])[0]
        profile=result['profile']
        self.assertEqual(profile['actions'],result['actions'])
        self.assertGreater(profile['inference_seconds'],0)
        self.assertEqual(profile['inference_calls'],result['actions'])
        self.assertIn('cache_hits',profile)
        self.assertIn('cache_misses',profile)

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(),'requires Apple Silicon Metal GPU')
    def test_mps_inference_and_policy_gradient(self):
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1,representation='ast').to('mps').eval()
        game=Game(from_rank(8),budget=20)
        initial=game.observation()
        inference=BatchedPolicyInference(policy,batch_size=2)
        inference.begin_version(0)
        statuses=inference.infer([encode_json(initial)])
        self.assertEqual(statuses[0][0],'ok')
        self.assertEqual(len(statuses[0][1]),len(initial['actions']))
        trace=[]
        policy_probs=FrozenSampler(policy)
        for action in ['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']:
            observation=game.observation()
            probs=policy_probs.probabilities(observation)
            index=next(i for i,v in enumerate(observation['actions']) if v['id']==action)
            trace.append({'action':action,'probability':probs[index]})
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
        result={'machine':from_rank(8).to_json(),'attempts':1,
                'successes':[{'actions':trace,'certificate':game.certificate()}]}
        update=learn_batch(policy,optimizer,[result],20,gpu_gradient_batch_size=2)
        self.assertTrue(update['updated'])
        self.assertEqual(update['proofs'],1)
        self.assertTrue(torch.isfinite(torch.tensor(update['loss'])))


if __name__ == '__main__':
    unittest.main()
