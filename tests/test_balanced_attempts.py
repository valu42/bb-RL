"""Exactly balanced initial-answer attempts and conditional REINFORCE."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from proofgame.environment import Game
from proofgame.epoch_train import execute_machine, learn_batch, summarize_initial_choices
from proofgame.exploration import answer_indices
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank


class BalancedAttemptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exactly_50_each_even_if_policy_prefers_halt(self):
        class Sampler:
            misses = 0
            hits = 0
            def probabilities(self, observation):
                return (0.999,0.0005,0.0005)
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'test'}
        first=execute_machine(row,17,100,1,0,False,Sampler(),0.,True)
        second=execute_machine(row,17,100,1,0,False,Sampler(),0.,True)
        self.assertEqual(first['attempts'],100)
        self.assertEqual(first['first_action_counts'],
                         {'choose:HALT':50,'choose:NONHALT':50})
        self.assertEqual(first['first_action_counts'],second['first_action_counts'])
        self.assertEqual(first['scheduled_initial_answer_counts'],
                         {'choose:HALT':50,'choose:NONHALT':50})
        self.assertEqual(summarize_initial_choices([first])['initial_choice_observations'],100)
        self.assertEqual(first['initial_behavior_probability_sums']['HALT'],50.)
        self.assertEqual(first['initial_behavior_probability_sums']['NONHALT'],50.)
        with self.assertRaisesRegex(ValueError,'even number'):
            execute_machine(row,17,3,1,0,False,Sampler(),0.,True)
        # Early stopping is allowed (and automatically enabled) in balanced mode.
        early=execute_machine(row,17,100,1,0,True,Sampler(),0.,True)
        self.assertEqual(early['attempts'],100)  # no successful proof in this fixture

    def test_stops_after_first_verified_proof_from_either_branch(self):
        from unittest.mock import patch
        import proofgame.epoch_train as trainer
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'test'}
        class Sampler:
            misses=0
            hits=0
        for winner in ('HALT','NONHALT'):
            with self.subTest(winner=winner):
                attempts=[]
                def fake_attempt(sampler,machine,rng,budget,**kwargs):
                    answer=kwargs['forced_initial_answer']
                    attempts.append(answer)
                    if answer==winner:
                        return {'answer':winner},[{'action':'choose:'+winner,
                                                  'probability':1.}], 'SOLVED'
                    return None,[{'action':'choose:'+answer,'probability':1.}], 'EXHAUSTED'
                with patch.object(trainer,'attempt',side_effect=fake_attempt):
                    result=execute_machine(row,17,100,20,0,False,Sampler(),0.,True)
                self.assertEqual(len(result['successes']),1)
                self.assertEqual(result['attempts'],len(attempts))
                self.assertLess(result['attempts'],100)
                self.assertEqual(attempts[-1],winner)
                self.assertEqual(sum(result['scheduled_initial_answer_counts'].values()),
                                 result['attempts'])
                self.assertEqual(result['first_action_counts']['choose:'+winner],1)

    def test_conditional_reinforce_skips_forced_action_and_only_first_proof(self):
        torch.manual_seed(7)
        model=Policy(width=32,layers=1).eval()
        other=Policy(**model.config).eval()
        other.load_state_dict(model.state_dict())
        optim=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        optim2=torch.optim.AdamW(other.parameters(),lr=3e-4,weight_decay=0.)
        machine=from_rank(8)
        game=Game(machine,budget=20)
        sampler=FrozenSampler(model)
        actions=['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']
        trace=[]
        for i,action in enumerate(actions):
            obs=game.observation()
            idx=next(j for j,a in enumerate(obs['actions']) if a['id']==action)
            prob=1. if i==0 else sampler.probabilities(obs)[idx]
            step={'action':action,'probability':prob}
            if i==0:
                step['forced_initial_answer']='HALT'
            trace.append(step)
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        success={'attempt':1,'actions':trace,'certificate':game.certificate()}
        row={'machine':machine.to_json(),'attempts':100,'successes':[success]}
        repeated={'machine':machine.to_json(),'attempts':100,'successes':[success,success]}
        result=learn_batch(model,optim,[row],20,balanced_initial_answers=True)
        result2=learn_batch(other,optim2,[repeated],20,balanced_initial_answers=True)
        self.assertTrue(result['updated'],str(result))
        self.assertEqual(result['unique_gradient_observations'],len(actions)-1)
        self.assertEqual(result2['proofs'],1)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,other.state_dict()[key])

    def test_tiny_end_to_end_run_and_eval_remains_unforced(self):
        from proofgame.unfiltered_dataset import build_dataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'data.json'
            dataset.write_text(json.dumps(build_dataset(1,1,4,12345)))
            output=root/'run'
            command=[sys.executable,'-m','proofgame.epoch_train',
                     '--dataset',str(dataset),'--out',str(output),
                     '--epochs','1','--attempts','2','--batch-size','1',
                     '--workers','1','--seed','7',
                     '--balanced-initial-answers']
            completed=subprocess.run(command,capture_output=True,text=True,timeout=150)
            self.assertEqual(completed.returncode,0,
                             completed.stdout+'\n'+completed.stderr)
            train=json.loads((output/'epoch_01_training.json').read_text())
            self.assertEqual(train['scheduled_initial_answer_counts'],
                             {'choose:HALT':1,'choose:NONHALT':1})
            self.assertEqual(train['first_action_counts']['choose:HALT'],1)
            self.assertEqual(train['first_action_counts']['choose:NONHALT'],1)
            self.assertEqual(train['attempts'],2)
            ev=json.loads((output/'epoch_01_evaluation.json').read_text())
            self.assertEqual(ev['scheduled_initial_answer_counts'],
                             {'choose:HALT':0,'choose:NONHALT':0})
            self.assertIn('Balanced initial attempts',completed.stdout)
            self.assertEqual(json.loads((output/'config.json').read_text())['balanced_initial_answers'],True)


if __name__=='__main__':
    unittest.main()
