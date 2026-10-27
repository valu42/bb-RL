"""Eight-attempt group-relative learning with fixed 100-machine evaluation."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from proofgame.environment import Game
from proofgame.epoch_train import (
    execute_machine, learn_batch, prepare_group_relative_gradients,
    verified_proof_reward,
)
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank
from proofgame.unfiltered_dataset import build_dataset


def scripted_group(policy, successes=1, budget=12):
    """Four forced HALT and four forced NONHALT, with controlled successes."""
    machine=from_rank(8)
    sampler=FrozenSampler(policy)
    rows=[]
    certificates=[]
    for attempt in range(8):
        answer='HALT' if attempt<4 else 'NONHALT'
        solved=attempt<successes
        if solved and answer!='HALT':
            raise ValueError('fixture only has a HALT proof')
        actions=(['choose:HALT','rule:exI','put:s','put:z','commit',
                  'rule:refl','submit'] if solved
                 else ['choose:'+answer]+['reset']*(budget-1))
        game=Game(machine,budget=budget)
        trace=[]
        for i,action in enumerate(actions):
            obs=game.observation()
            idx=next(j for j,a in enumerate(obs['actions']) if a['id']==action)
            p=1. if i==0 else sampler.probabilities(obs)[idx]
            step={'action':action,'probability':p}
            if i==0:
                step['forced_initial_answer']=answer
            trace.append(step)
            game.step(action,observe=False)
        cert=game.certificate() if solved else None
        if solved:
            assert game.status=='SOLVED'
            certificates.append({'attempt':attempt+1,'actions':trace,
                                 'certificate':cert})
        else:
            assert game.status=='EXHAUSTED'
        rows.append({'attempt':attempt+1,'actions':trace,
                     'status':game.status,'certificate':cert})
    return {'machine':machine.to_json(),'attempts':8,
            'successes':certificates,'group_trajectories':rows}


class GroupRelativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_heldout_dataset_is_disjoint_and_deterministic(self):
        old=build_dataset(1000,100,100,20261026)
        ranks=[r['rank'] for r in old['test']]
        new=build_dataset(12500,100,100,20261026,fixed_test_ranks=ranks)
        self.assertEqual(new['test'],old['test'])
        self.assertEqual(len(new['train']),12500)
        self.assertEqual(len(set(r['rank'] for r in new['train'])),12500)
        self.assertTrue(set(r['rank'] for r in new['train']).isdisjoint(ranks))
        self.assertTrue(set(r['rank'] for r in new['train']).isdisjoint(
            new['excluded_benchmark_ranks']))
        self.assertEqual(len(new['excluded_benchmark_ranks']),2)
        self.assertEqual(new,build_dataset(12500,100,100,20261026,
                                           fixed_test_ranks=ranks))
        with self.assertRaises(ValueError):
            build_dataset(12500,100,100,20261026,fixed_test_ranks=ranks[:-1])

    def test_collects_all_eight_attempts_after_a_success(self):
        from unittest.mock import patch
        from proofgame import epoch_train as et
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'fixture'}
        class Dummy:
            hits=misses=0
        calls=[]
        def fake_attempt(sampler,machine,rng,budget,**kwargs):
            answer=kwargs['forced_initial_answer']
            calls.append(answer)
            kwargs['on_initial_choice']('choose:'+answer,
                {'HALT':.5,'NONHALT':.5},
                {'HALT':float(answer=='HALT'),'NONHALT':float(answer=='NONHALT')})
            trace=[{'action':'choose:'+answer,'probability':1.,
                    'forced_initial_answer':answer}]
            return ({'answer':answer} if len(calls)==1 else None),trace,(
                'SOLVED' if len(calls)==1 else 'EXHAUSTED')
        with patch.object(et,'attempt',side_effect=fake_attempt):
            result=execute_machine(row,11,8,10,0,False,Dummy(),
                                   balanced_initial_answers=True,
                                   group_relative=True)
        self.assertEqual(result['attempts'],8)
        self.assertEqual(len(result['group_trajectories']),8)
        self.assertEqual(len(result['successes']),1)
        self.assertEqual(calls.count('HALT'),4)
        self.assertEqual(calls.count('NONHALT'),4)

    def test_leave_one_out_advantages_and_negative_failure_gradients(self):
        torch.manual_seed(7)
        model=Policy(width=32,layers=1).eval()
        result=scripted_group(model,successes=1)
        groups,proofs,stats=prepare_group_relative_gradients(
            [result],12,.3,balanced_initial_answers=True)
        self.assertEqual(proofs,1)
        self.assertEqual(stats['groups_informative'],1)
        self.assertEqual(stats['group_positive_advantage_attempts'],1)
        self.assertEqual(stats['group_negative_advantage_attempts'],7)
        self.assertEqual(stats['groups_all_failed'],0)
        reward=verified_proof_reward(7,12,.3)
        self.assertAlmostEqual(stats['mean_group_reward'],reward/8)
        # Compute the reference loss by independently replaying each
        # trajectory and applying leave-one-out advantages to log-probs.
        rewards=[reward]+[0.]*7
        reference=0.
        for t,r in zip(result['group_trajectories'],rewards):
            advantage=(8*r-sum(rewards))/7
            game=Game(from_rank(8),budget=12)
            for step in t['actions']:
                obs=game.observation()
                if 'forced_initial_answer' not in step:
                    logits=model(obs)[0]
                    idx=next(i for i,a in enumerate(obs['actions'])
                             if a['id']==step['action'])
                    reference-=advantage*float(logits.log_softmax(-1)[idx])/8
                game.step(step['action'],observe=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.)
        update=learn_batch(model,optimizer,[result],12,
                           balanced_initial_answers=True,
                           length_penalty=.3,group_relative=True)
        self.assertTrue(update['updated'],str(update))
        self.assertAlmostEqual(update['loss'],reference,places=3)
        self.assertAlmostEqual(update['mean_verified_proof_reward'],reward)
        self.assertTrue(groups)
        self.assertTrue(any(any(v<0 for v in counts)
                            for counts,_ in groups.values()))

    def test_all_fail_and_equal_reward_groups_have_no_update(self):
        torch.manual_seed(8)
        model=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        failed=scripted_group(model,successes=0)
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        update=learn_batch(model,optimizer,[failed],12,
                           balanced_initial_answers=True,group_relative=True)
        self.assertFalse(update['updated'])
        self.assertEqual(update['groups_all_failed'],1)
        self.assertEqual(update['groups_informative'],0)
        self.assertFalse(optimizer.state)
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,original[k]))

    def test_uniform_success_rewards_produce_no_relative_gradient(self):
        torch.manual_seed(15)
        model=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        base=scripted_group(model,successes=1,budget=12)
        successful=base['group_trajectories'][0]
        # Same valid proof eight times, no forced answer balancing.
        copied=[]
        for i in range(8):
            actions=[{k:v for k,v in step.items()
                      if k!='forced_initial_answer'}
                     for step in successful['actions']]
            # The first action was forced in the original fixture; restore
            # its true on-policy probability for unforced group training.
            game=Game(from_rank(8),budget=12)
            actions[0]['probability']=FrozenSampler(model).probabilities(
                game.observation())[0]
            copied.append({'attempt':i+1,'actions':actions,
                           'status':'SOLVED',
                           'certificate':successful['certificate']})
        row={'machine':from_rank(8).to_json(),'attempts':8,
             'successes':[{'attempt':t['attempt'],'actions':t['actions'],
                           'certificate':t['certificate']} for t in copied],
             'group_trajectories':copied}
        update=learn_batch(model,optimizer,[row],12,
                           length_penalty=.3,group_relative=True)
        self.assertFalse(update['updated'])
        self.assertEqual(update['proofs'],8)
        self.assertEqual(update['groups_all_successful'],1)
        self.assertEqual(update['groups_uniform_reward'],1)
        self.assertFalse(optimizer.state)

    def test_full_end_to_end_and_fixed_eval_attempt_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=build_dataset(5,2,10,20261026)
            new=build_dataset(3,2,10,99,
                              fixed_test_ranks=[r['rank'] for r in old['test']])
            dataset=root/'data.json'
            dataset.write_text(json.dumps(new))
            out=root/'group'
            cmd=[sys.executable,'-m','proofgame.epoch_train',
                 '--dataset',str(dataset),'--out',str(out),
                 '--epochs','1','--attempts','8','--eval-attempts','3',
                 '--batch-size','2','--workers','2','--seed','7',
                 '--representation','ast','--device','cpu',
                 '--balanced-initial-answers','--group-relative',
                 '--cpu-shared-action-cache','--length-penalty','.3']
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
            self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
            config=json.loads((out/'config.json').read_text())
            self.assertEqual(config['attempts_per_machine'],8)
            self.assertEqual(config['eval_attempts_per_machine'],3)
            self.assertTrue(config['group_relative'])
            train=json.loads((out/'epoch_01_training.json').read_text())
            self.assertEqual(train['attempts'],24)
            self.assertEqual(train['scheduled_initial_answer_counts'],
                             {'choose:HALT':12,'choose:NONHALT':12})
            self.assertEqual(train['group_relative_stats']['groups_total'],3)
            eval_report=json.loads((out/'epoch_01_evaluation.json').read_text())
            self.assertEqual(eval_report['attempts'],6)
            for p in (out/'epoch_01_train').glob('batch_*.json'):
                batch=json.loads(p.read_text())
                self.assertIn('groups_total',batch['update'])
                for machine in batch['machines']:
                    self.assertNotIn('group_trajectories',machine)
            bad=subprocess.run(cmd+['--resume','--attempts','10'],
                               capture_output=True,text=True,timeout=30)
            self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':
    unittest.main()
