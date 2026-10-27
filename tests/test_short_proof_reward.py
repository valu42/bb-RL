"""Length-aware REINFORCE: reward verified successes, never reward failures."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from proofgame.environment import Game
from proofgame.epoch_train import learn_batch, verified_proof_reward
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank


class ShortProofRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reward_formula_and_validation(self):
        self.assertEqual(verified_proof_reward(5,100,0),1.)
        self.assertAlmostEqual(verified_proof_reward(5,100,.3),.985)
        self.assertAlmostEqual(verified_proof_reward(50,100,.3),.85)
        self.assertAlmostEqual(verified_proof_reward(100,100,.3),.7)
        self.assertGreater(verified_proof_reward(5,100,.3),
                           verified_proof_reward(50,100,.3))
        for alpha in (-.1,1.,float('nan'),float('inf')):
            with self.subTest(alpha=alpha),self.assertRaises(ValueError):
                verified_proof_reward(5,100,alpha)
        for length in (0,101,True):
            with self.subTest(length=length),self.assertRaises(ValueError):
                verified_proof_reward(length,100,.3)

    def make_success(self,policy,extra_resets=0,budget=20):
        machine=from_rank(8)
        game=Game(machine,budget=budget)
        sampler=FrozenSampler(policy)
        trace=[]
        actions=['reset']*extra_resets + [
            'choose:HALT','rule:exI','put:s','put:z',
            'commit','rule:refl','submit']
        for action in actions:
            obs=game.observation()
            self.assertIn(action,[x['id'] for x in obs['actions']])
            probs=sampler.probabilities(obs)
            index=next(i for i,a in enumerate(obs['actions']) if a['id']==action)
            trace.append({'action':action,'probability':probs[index]})
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        return {'actions':trace,'certificate':game.certificate()}

    def test_weighted_loss_matches_independent_trajectory_sum(self):
        torch.manual_seed(7)
        model=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.)
        short=self.make_success(model,extra_resets=0)
        long=self.make_success(model,extra_resets=3)
        self.assertEqual(len(short['actions']),7)
        self.assertEqual(len(long['actions']),10)
        machine=from_rank(8)
        row={'machine':machine.to_json(),'attempts':2,
             'successes':[short,long]}
        expected={}
        for alpha in (0.,.3):
            total=0.
            for success in (short,long):
                game=Game(machine,budget=20)
                reward=verified_proof_reward(len(success['actions']),20,alpha)
                for step in success['actions']:
                    obs=game.observation()
                    logits=model(obs)[0]
                    index=next(i for i,a in enumerate(obs['actions'])
                               if a['id']==step['action'])
                    total-=reward*float(logits.log_softmax(-1)[index])/2
                    game.step(step['action'],observe=False)
            expected[alpha]=total
        zero=learn_batch(model,optimizer,[row],20,length_penalty=0.)
        shaped=learn_batch(model,optimizer,[row],20,length_penalty=.3)
        self.assertTrue(zero['updated'])
        self.assertTrue(shaped['updated'])
        self.assertAlmostEqual(zero['loss'],expected[0.],places=3)
        self.assertAlmostEqual(shaped['loss'],expected[.3],places=3)
        self.assertLess(shaped['loss'],zero['loss'])
        self.assertEqual(shaped['mean_verified_proof_actions'],8.5)
        self.assertAlmostEqual(shaped['mean_verified_proof_reward'],
                               (verified_proof_reward(7,20,.3)+
                                verified_proof_reward(10,20,.3))/2)

    def test_zero_reward_failures_and_invalid_coefficient(self):
        torch.manual_seed(7)
        model=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        empty={'machine':from_rank(8).to_json(),'attempts':20,'successes':[]}
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        result=learn_batch(model,optimizer,[empty],20,length_penalty=.3)
        self.assertFalse(result['updated'])
        self.assertIsNone(result['mean_verified_proof_reward'])
        self.assertFalse(optimizer.state)
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,initial[name]))
        with self.assertRaises(ValueError):
            learn_batch(model,optimizer,[empty],20,length_penalty=float('nan'))

    def test_faster_cpu_mode_and_reward_cli(self):
        from proofgame.unfiltered_dataset import build_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'dataset.json'
            dataset.write_text(json.dumps(build_dataset(2,1,20,20261027)))
            out=root/'run'
            cmd=[sys.executable,'-m','proofgame.epoch_train',
                 '--dataset',str(dataset),'--out',str(out),
                 '--epochs','1','--attempts','2','--batch-size','2',
                 '--workers','1','--seed','7','--representation','ast',
                 '--device','cpu','--cpu-shared-action-cache',
                 '--length-penalty','.3','--profile']
            result=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            config=json.loads((out/'config.json').read_text())
            self.assertEqual(config['length_penalty'],.3)
            self.assertEqual(config['cpu_inference'],'worker-shared-action-cache')
            self.assertTrue((out/'epoch_01_evaluation.json').is_file())
            batch=json.loads((out/'epoch_01_train'/'batch_0000.json').read_text())
            self.assertIn('mean_verified_proof_reward',batch['update'])
            self.assertIn('mean_verified_proof_actions',batch['update'])
            # A different reward must never silently resume an old run.
            bad=subprocess.run(cmd+['--resume','--length-penalty','.4'],
                               text=True,capture_output=True,timeout=30)
            self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':
    unittest.main()
