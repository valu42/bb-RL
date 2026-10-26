import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch dependency is not installed')
class EpochTests(unittest.TestCase):
    def test_grouped_duplicate_gradients_equal_single_trajectory(self):
        import torch
        from proofgame.epoch_train import learn_batch,task_seed
        from proofgame.environment import Game
        from proofgame.model import Policy
        from proofgame.population_sample import from_rank
        torch.set_num_threads(1);torch.manual_seed(7)
        a=Policy(width=32,layers=1).eval();b=Policy(**a.config).eval();b.load_state_dict(a.state_dict())
        oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=0.)
        ob=torch.optim.AdamW(b.parameters(),lr=3e-4,weight_decay=0.)
        machine=from_rank(8);g=Game(machine,budget=20);trace=[]
        for action in ['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']:
            obs=g.observation()
            with torch.no_grad():probabilities=a(obs)[0].softmax(-1)
            index=next(i for i,x in enumerate(obs['actions']) if x['id']==action)
            trace.append({'action':action,'probability':float(probabilities[index])});g.step(action,observe=False)
        success={'actions':trace,'certificate':g.certificate()}
        empty={'machine':machine.to_json(),'attempts':100,'successes':[]}
        initial={k:v.clone() for k,v in a.state_dict().items()}
        self.assertFalse(learn_batch(a,oa,[empty],20)['updated'])
        self.assertFalse(oa.state)
        self.assertTrue(all(torch.equal(v,initial[k]) for k,v in a.state_dict().items()))
        one={'machine':machine.to_json(),'attempts':1,'successes':[success]}
        two={'machine':machine.to_json(),'attempts':2,'successes':[success,success]}
        ra=learn_batch(a,oa,[one],20);rb=learn_batch(b,ob,[two],20)
        self.assertEqual(ra['unique_gradient_observations'],rb['unique_gradient_observations'])
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k])
        self.assertEqual(task_seed(7,'eval',1,'machine'),task_seed(7,'eval',10,'machine'))
        self.assertNotEqual(task_seed(7,'train',1,'machine'),task_seed(7,'train',2,'machine'))

    def test_first_proof_stops_training_but_not_evaluation(self):
        from unittest.mock import patch
        import proofgame.epoch_train as et
        from proofgame.population_sample import from_rank

        class FakePolicy:
            def load_state_dict(self, weights):
                pass

        machine=from_rank(8)
        selected={'rank':8,'machine_id':'test-id','machine':machine.to_json()}
        sample={'version':0,'state_dict':{}}
        def fake_attempt(sampler, machine, rng, budget):
            return {'proof':'mock'}, [{'action':'submit','probability':1.}], 'SOLVED'
        with patch.object(et,'_worker_policy',FakePolicy()), \
             patch.object(et.torch,'load',return_value=sample), \
             patch.object(et,'FrozenSampler') as sampler, \
             patch.object(et,'attempt',side_effect=fake_attempt) as run:
            sampler.return_value.misses=0
            sampler.return_value.hits=0
            task=(selected,7,100,20,'unused',0,True)
            training=et.run_machine(task)
            self.assertEqual(training['attempts'],1)
            self.assertEqual(len(training['successes']),1)
            self.assertEqual(run.call_count,1)
            run.reset_mock()
            evaluation=et.run_machine((*task[:-1],False))
            self.assertEqual(evaluation['attempts'],100)
            self.assertEqual(len(evaluation['successes']),100)
            self.assertEqual(run.call_count,100)


    def test_auto_new_run_ids_and_exact_resume_path(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from proofgame.epoch_train import choose_run_directory
        with TemporaryDirectory() as tmp:
            base=Path(tmp)/'experiment'
            first=choose_run_directory(base)
            second=choose_run_directory(base)
            third=choose_run_directory(base)
            self.assertEqual(first,base)
            self.assertEqual(second,base.with_name('experiment_run_0001'))
            self.assertEqual(third,base.with_name('experiment_run_0002'))
            self.assertTrue(all(path.is_dir() for path in (first,second,third)))
            self.assertEqual(choose_run_directory(base,resume=True),base)
            with self.assertRaises(FileNotFoundError):
                choose_run_directory(Path(tmp)/'missing',resume=True)

    def test_nonfinite_optimizer_step_is_rejected_and_restored(self):
        from unittest.mock import patch
        import torch
        from proofgame.epoch_train import learn_batch,policy_is_finite
        from proofgame.environment import Game
        from proofgame.model import Policy
        from proofgame.population_sample import from_rank
        torch.set_num_threads(1)
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
        machine=from_rank(8)
        game=Game(machine,budget=20)
        trajectory=[]
        for action in ['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']:
            obs=game.observation()
            with torch.inference_mode():
                probs=policy(obs)[0].softmax(-1)
            idx=next(i for i,v in enumerate(obs['actions']) if v['id']==action)
            trajectory.append({'action':action,'probability':float(probs[idx])})
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        result={'machine':machine.to_json(),'attempts':1,
                'successes':[{'actions':trajectory,'certificate':game.certificate()}]}
        original={name:t.detach().clone() for name,t in policy.state_dict().items()}
        real_step=optimizer.step
        def poison(*args,**kwargs):
            real_step(*args,**kwargs)
            with torch.no_grad():
                next(policy.parameters()).fill_(float('nan'))
        with patch.object(optimizer,'step',side_effect=poison):
            update=learn_batch(policy,optimizer,[result],20)
        self.assertFalse(update['updated'])
        self.assertEqual(update['rejected_update'],'nonfinite_optimizer_result')
        self.assertTrue(policy_is_finite(policy))
        self.assertFalse(optimizer.state)
        for name,value in policy.state_dict().items():
            torch.testing.assert_close(value,original[name])


if __name__=='__main__':unittest.main()
