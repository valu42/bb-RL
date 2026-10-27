"""Branch-race training: up to 6 rounds of 8 HALT + 8 NONHALT attempts."""
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from proofgame import epoch_train as et
from proofgame.environment import Game
from proofgame.model import Policy
from proofgame.network_search import FrozenSampler
from proofgame.population_sample import from_rank
from proofgame.unfiltered_dataset import build_dataset
from tests.test_group_relative import scripted_group


class BranchRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fake_rollout(self, *, success_on=None, success_branch='HALT', group_size=8,
                     rounds=6, seed=77):
        """Mock only the proof search; exercise real round scheduling and logging."""
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'branch-round-fixture'}
        calls=[]
        class Sampler:
            hits=misses=0
        def fake_attempt(sampler,machine,rng,budget,**kwargs):
            branch=kwargs['forced_initial_answer']
            calls.append(branch)
            kwargs['on_initial_choice'](
                'choose:'+branch,{'HALT':.5,'NONHALT':.5},
                {'HALT':float(branch=='HALT'),
                 'NONHALT':float(branch=='NONHALT')})
            round_number=(len(calls)-1)//(2*group_size)+1
            succeeded=(success_on is not None and round_number==success_on and
                       branch==success_branch and
                       calls[(round_number-1)*2*group_size:].count(branch)==1)
            trace=[{'action':'choose:'+branch,'probability':1.,
                    'forced_initial_answer':branch}]
            return ({'answer':branch} if succeeded else None),trace,(
                'SOLVED' if succeeded else 'EXHAUSTED')
        with patch.object(et,'attempt',side_effect=fake_attempt):
            result=et.execute_machine(row,seed,group_size,12,0,False,Sampler(),
                                      balanced_initial_answers=True,
                                      group_relative=True,branch_rounds=rounds)
        return result,calls

    def test_first_round_winner_has_complete_group(self):
        for branch in ('HALT','NONHALT'):
            result,calls=self.fake_rollout(success_on=1,success_branch=branch)
            self.assertEqual(result['attempts'],16)
            self.assertEqual(result['rounds_completed'],1)
            self.assertEqual(result['first_proof_round'],1)
            self.assertEqual(result['first_proof_answer'],branch)
            self.assertEqual(result['attempts_by_branch'],{'HALT':8,'NONHALT':8})
            self.assertEqual(len(result['group_trajectories']),8)
            self.assertEqual([t['attempt'] for t in result['group_trajectories']],
                             list(range(1,9)))
            self.assertTrue(all(t['actions'][0]['forced_initial_answer']==branch
                                for t in result['group_trajectories']))
            self.assertEqual(Counter(calls),{'HALT':8,'NONHALT':8})

    def test_later_round_and_maximum_attempts(self):
        found,calls=self.fake_rollout(success_on=4,success_branch='HALT')
        self.assertEqual(found['attempts'],64)
        self.assertEqual(found['rounds_completed'],4)
        self.assertEqual(found['first_proof_round'],4)
        self.assertEqual(len(found['group_trajectories']),8)
        self.assertEqual(Counter(calls),{'HALT':32,'NONHALT':32})
        failed,calls=self.fake_rollout(success_on=None)
        self.assertEqual(failed['attempts'],96)
        self.assertEqual(failed['rounds_completed'],6)
        self.assertIsNone(failed['first_proof_round'])
        self.assertIsNone(failed['first_proof_answer'])
        self.assertEqual(failed['group_trajectories'],[])
        self.assertEqual(Counter(calls),{'HALT':48,'NONHALT':48})
        hist=et.summarize_first_proof_rounds([found,failed])
        self.assertEqual(hist['HALT'],{'4':1})
        self.assertEqual(hist['NONHALT'],{})
        self.assertEqual(hist['no_proof'],1)
        self.assertEqual(hist['attempts_by_branch'],{'HALT':80,'NONHALT':80})

    def test_randomized_order_is_reproducible_and_not_always_halt_first(self):
        first=self.fake_rollout(success_on=None,rounds=1,seed=4)[1]
        self.assertEqual(first,self.fake_rollout(success_on=None,rounds=1,seed=4)[1])
        starts={self.fake_rollout(success_on=None,rounds=1,seed=seed)[1][0]
                for seed in range(12)}
        self.assertEqual(starts,{'HALT','NONHALT'})

    def test_gradient_uses_only_selected_branch_group(self):
        torch.manual_seed(17)
        model=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.)
        group=scripted_group(model,successes=1,budget=12)
        winning=[dict(t,attempt=i+1) for i,t in
                 enumerate(group['group_trajectories'][:4])]
        # Add four further HALT failures with valid on-policy trajectories.
        winning += [dict(t,attempt=i+5) for i,t in
                    enumerate(group['group_trajectories'][1:5])]
        row={**group,'attempts':64,'rounds_completed':4,
             'first_proof_round':4,'first_proof_answer':'HALT',
             'group_trajectories':winning,
             'successes':[group['successes'][0]]}
        groups,proofs,stats=et.prepare_group_relative_gradients(
            [row],12,.3,balanced_initial_answers=True,branch_rounds=6)
        self.assertEqual(proofs,1)
        self.assertEqual(stats['group_attempts'],8)
        self.assertEqual(stats['exploration_attempts'],64)
        self.assertEqual(stats['groups_informative'],1)
        update=et.learn_batch(model,optimizer,[row],12,
                              balanced_initial_answers=True,length_penalty=.3,
                              group_relative=True,branch_rounds=6)
        self.assertTrue(update['updated'],str(update))
        self.assertEqual(update['groups_informative'],1)

    def test_all_failed_race_has_no_gradient(self):
        torch.manual_seed(18)
        model=Policy(width=32,layers=1).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        machine=from_rank(8)
        row={'machine':machine.to_json(),'attempts':96,
             'first_proof_round':None,'first_proof_answer':None,
             'group_trajectories':[],'successes':[]}
        update=et.learn_batch(model,optimizer,[row],12,
                              balanced_initial_answers=True,
                              group_relative=True,branch_rounds=6)
        self.assertFalse(update['updated'])
        self.assertEqual(update['groups_all_failed'],1)
        self.assertFalse(optimizer.state)

    def test_end_to_end_train_eval_and_saved_round_histograms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'dataset.json'
            dataset.write_text(json.dumps(build_dataset(2,1,12,20261027)))
            out=root/'run'
            cmd=[sys.executable,'-m','proofgame.epoch_train',
                 '--dataset',str(dataset),'--out',str(out),
                 '--epochs','1','--attempts','8','--eval-attempts','3',
                 '--branch-rounds','2','--group-relative',
                 '--balanced-initial-answers',
                 '--batch-size','2','--workers','2',
                 '--representation','ast','--device','cpu',
                 '--cpu-shared-action-cache','--length-penalty','.3']
            result=subprocess.run(cmd,text=True,capture_output=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            cfg=json.loads((out/'config.json').read_text())
            self.assertEqual(cfg['branch_rounds'],2)
            train=json.loads((out/'epoch_01_training.json').read_text())
            self.assertIn('first_proof_round_by_answer',train)
            hist=train['first_proof_round_by_answer']
            self.assertEqual(sum(map(sum,(map(int,hist[x].values())
                                      for x in ('HALT','NONHALT'))))+
                             hist['no_proof'],2)
            self.assertEqual(hist['attempts_by_branch']['HALT'],
                             hist['attempts_by_branch']['NONHALT'])
            for path in (out/'epoch_01_train').glob('batch_*.json'):
                for row in json.loads(path.read_text())['machines']:
                    self.assertNotIn('group_trajectories',row)
                    self.assertIn('first_proof_round',row)
                    self.assertEqual(row['attempts']%16,0)
            eval_report=json.loads((out/'epoch_01_evaluation.json').read_text())
            self.assertEqual(eval_report['attempts'],3)
            bad=subprocess.run(cmd+['--resume','--branch-rounds','3'],
                               capture_output=True,text=True,timeout=30)
            self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':
    unittest.main()
