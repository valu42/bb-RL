"""Regression tests for answer-branch exploration and on-policy learning."""
import math
import random
import unittest

import torch

from proofgame.environment import Game
from proofgame.exploration import answer_indices, behavior_log_probs, mix_probabilities
from proofgame.epoch_train import execute_machine, learn_batch, summarize_initial_choices
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixture_preserves_reset_and_guarantees_both_answers(self):
        obs=Game(from_rank(8),budget=20).observation()
        self.assertIsNotNone(answer_indices(obs))
        raw=(0.999,0.0001,0.0009)
        # Find indexes by name, not by assumed ordering.
        indexes=answer_indices(obs)
        probabilities=[0.]*len(obs['actions'])
        probabilities[indexes[0]]=0.999
        probabilities[indexes[1]]=0.0001
        for i in range(len(probabilities)):
            if i not in indexes:
                probabilities[i]=0.0009
        mixed=mix_probabilities(obs,probabilities,0.2)
        self.assertAlmostEqual(sum(mixed),1.)
        self.assertGreaterEqual(mixed[indexes[0]],0.1)
        self.assertGreaterEqual(mixed[indexes[1]],0.1)
        self.assertEqual(mix_probabilities(obs,probabilities,0.),tuple(probabilities))
        for i in range(len(probabilities)):
            if i not in indexes:
                self.assertAlmostEqual(mixed[i],0.8*probabilities[i])
        for bad in (-0.1,1.1,float('nan')):
            with self.assertRaises(ValueError):
                mix_probabilities(obs,probabilities,bad)

    def test_log_probabilities_match_sampling_and_have_correct_gradient(self):
        obs=Game(from_rank(8),budget=20).observation()
        logits=torch.tensor([3.,-2.,0.],requires_grad=True)
        epsilon=0.2
        logp=behavior_log_probs(logits,obs,epsilon)
        mixed=mix_probabilities(obs,logits.softmax(-1).tolist(),epsilon)
        torch.testing.assert_close(logp.exp(),torch.tensor(mixed),rtol=1e-6,atol=1e-7)
        i=answer_indices(obs)[1]
        logp[i].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        # Finite difference on the actual mixed log probability.
        delta=1e-3
        for k in range(len(logits)):
            up=logits.detach().clone(); down=logits.detach().clone()
            up[k]+=delta;down[k]-=delta
            finite=(behavior_log_probs(up,obs,epsilon)[i]-
                    behavior_log_probs(down,obs,epsilon)[i])/(2*delta)
            self.assertAlmostEqual(float(logits.grad[k]),float(finite),places=3)

    def test_unchanged_evaluation_and_logged_training_choices(self):
        torch.manual_seed(5)
        policy=Policy(width=32,layers=1).eval()
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'fixture'}
        train=execute_machine(row,42,5,10,0,False,FrozenSampler(policy),0.2)
        evaluate=execute_machine(row,42,5,10,0,False,FrozenSampler(policy),0.)
        self.assertEqual(train['initial_choice_observations'],5)
        self.assertEqual(evaluate['initial_choice_observations'],5)
        self.assertEqual(sum(train['first_action_counts'].values()),5)
        self.assertEqual(sum(evaluate['first_action_counts'].values()),5)
        stats=summarize_initial_choices([train])
        self.assertEqual(stats['initial_choice_observations'],5)
        for answer in ('HALT','NONHALT'):
            self.assertGreaterEqual(stats['mean_initial_behavior_probability'][answer],0.1)
        # Evaluation raw and behavior distributions must be identical.
        raw=summarize_initial_choices([evaluate])
        self.assertEqual(raw['mean_initial_policy_probability'],
                         raw['mean_initial_behavior_probability'])

    def test_verified_reinforce_uses_exploration_distribution(self):
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
        machine=from_rank(8)
        game=Game(machine,budget=20)
        sampler=FrozenSampler(policy)
        actions=['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']
        trace=[]
        for action in actions:
            obs=game.observation()
            probs=mix_probabilities(obs,sampler.probabilities(obs),0.2)
            idx=next(i for i,x in enumerate(obs['actions']) if x['id']==action)
            trace.append({'action':action,'probability':probs[idx]})
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        result={'machine':machine.to_json(),'attempts':1,
                'successes':[{'actions':trace,'certificate':game.certificate()}]}
        update=learn_batch(policy,optimizer,[result],20,exploration_epsilon=0.2)
        self.assertTrue(update['updated'],str(update))
        self.assertEqual(update['proofs'],1)


if __name__=='__main__':
    unittest.main()
