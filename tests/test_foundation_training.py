import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'torch is optional')
class FoundationTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        from proofgame.foundation.logic import Machine
        self.m=Machine((('qH','b0','R'),('qH','b1','R'),('qH','b0','R'),('qH','b1','R')))
    def test_informative_group_updates_from_foundational_replay(self):
        from proofgame.model import Policy
        from proofgame.foundation.compiler import compile_halt,replay
        from proofgame.foundation.environment import Game
        from proofgame.foundation.training import learn
        p=Policy(width=16,layers=1,heads=4,representation='ast');opt=torch.optim.AdamW(p.parameters(),lr=.001)
        cert=compile_halt(self.m,1);actions=replay(cert,budget=64);g=Game(self.m,budget=64);probs=[]
        with torch.no_grad():
            for i,a in enumerate(actions):
                obs=g.observation();logits,_=p(obs);j=next(j for j,v in enumerate(obs['actions']) if v['id']==a)
                probs.append(1. if i==0 else torch.softmax(logits,0)[j].item());g.step(a,observe=False)
        success={'actions':actions,'probabilities':probs,'certificate':cert,'forced':'HALT'}
        failure={'actions':['choose:HALT'],'probabilities':[1.],'certificate':None,'forced':'HALT'}
        before={k:v.clone() for k,v in p.state_dict().items()}
        metrics=learn(p,opt,[{'machine':self.m.to_json(),'group':[success,failure]}],64,.3)
        self.assertEqual(metrics['informative_groups'],1);self.assertTrue(metrics['update_accepted'])
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in p.state_dict().items()))
    def test_sft_resume_matches_uninterrupted_and_v1_rejected(self):
        from proofgame.foundation.compiler import compile_halt,replay
        from proofgame.foundation.data import atomic,digest,hash_file,language_hash,VERSION
        from proofgame.foundation.__main__ import parser
        from proofgame.foundation.training import pretrain,load,init
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';data.mkdir();cert=compile_halt(self.m,1)
            row={'machine':self.m.to_json(),'machine_id':digest(self.m.to_json()),'family_id':'test','states':2}
            machines={'splits':{'train':[row],'dev':[row],'test':[row]}}
            example={'source':'halt','answer':'HALT','proof_id':digest(cert),'certificate':cert,'actions':replay(cert,budget=64)}
            (data/'corpus.jsonl').write_text(json.dumps({**row,'examples':[example]})+'\n')
            atomic(data/'machines.json',machines)
            atomic(data/'manifest.json',{'version':VERSION,'source_hash':digest(machines),'config':{'budget':64,'language_hash':language_hash()},'corpus_sha256':hash_file(data/'corpus.jsonl')})
            def args(out,updates,resume=False):
                argv=['pretrain','--data',str(data),'--out',str(root/out),'--updates',str(updates),'--batch-size','1','--width','16','--layers','1','--eval-every','1','--eval-machines','1','--eval-attempts','1','--curriculum-updates','0']
                if resume:argv+=['--resume']
                return parser().parse_args(argv)
            pretrain(args('full',2));pretrain(args('split',1));pretrain(args('split',2,True))
            full=load(root/'full/checkpoint.pt');split=load(root/'split/checkpoint.pt')
            for k,v in full['state_dict'].items():self.assertTrue(torch.equal(v,split['state_dict'][k]),k)
            self.assertEqual(full['update'],split['update'])
            # Interrupt after the optimizer update, during evaluation. Resume
            # must evaluate those weights without applying that update twice.
            with patch('proofgame.foundation.training.autonomous',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):pretrain(args('interrupted',1))
            pending=load(root/'interrupted/checkpoint.pt')
            self.assertEqual(pending['update'],1)
            self.assertEqual(pending['pending_evaluation']['update'],1)
            pretrain(args('interrupted',2,True))
            resumed=load(root/'interrupted/checkpoint.pt')
            self.assertNotIn('pending_evaluation',resumed)
            for k,v in full['state_dict'].items():self.assertTrue(torch.equal(v,resumed['state_dict'][k]),k)
            records=[json.loads(line) for line in (root/'interrupted/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in records if r['event']=='pretrain'],[1,2])
            self.assertTrue(any(r['event']=='evaluation_start' for r in records))
            self.assertTrue(all('evaluation_seconds' in r for r in records if r['event']=='pretrain'))
            torch.save({'language':'proofgame-0.1'},root/'old.pt')
            a=parser().parse_args(['train','--data',str(data),'--out',str(root/'bad'),'--init',str(root/'old.pt')])
            with self.assertRaises(ValueError):init(a,'rl')

    def test_parallel_evaluation_preserves_serial_search_and_reports_work(self):
        from proofgame.foundation.evaluation import evaluate_rows
        from proofgame.foundation.training import rollout,summarize
        from proofgame.model import Policy
        torch.manual_seed(31)
        policy=Policy(width=16,layers=1,heads=4,representation='ast')
        rows=[{'machine':self.m.to_json(),'machine_id':str(i),'states':2} for i in range(3)]
        # Independently reproduce the original serial evaluator's seed scheme.
        original=[]
        policy.eval()
        for i,row in enumerate(rows):
            successes=[]
            for a in range(2):
                t=rollout(policy,row,12,123+i*100000+a)
                if t['certificate']:
                    successes=[{'attempt':a+1,'actions':t['actions'],'certificate':t['certificate']}];break
            original.append({**row,'successes':successes})
        serial=evaluate_rows(policy,rows,12,2,123,workers=1,on_progress=lambda _:None)
        events=[]
        parallel=evaluate_rows(policy,rows,12,2,123,workers=2,on_progress=events.append,progress_interval=.05)
        self.assertEqual(summarize(parallel),summarize(original))
        for left,right in zip(serial,parallel):
            self.assertEqual({k:v for k,v in left.items() if k!='timing'},
                             {k:v for k,v in right.items() if k!='timing'})
        self.assertEqual([r['machine_id'] for r in parallel],['0','1','2'])
        self.assertEqual(events[0]['event'],'evaluation_start')
        self.assertTrue(any(e['event']=='evaluation_progress' for e in events))
        end=events[-1]
        self.assertEqual(end['event'],'evaluation_complete')
        self.assertEqual(end['workers'],2)
        self.assertEqual(end['machines_done'],3)
        self.assertEqual(end['actions_done'],sum(r['actions'] for r in parallel))
        self.assertEqual(end['attempts_done'],sum(r['attempts'] for r in parallel))
        self.assertEqual(end['eta_seconds'],0.)
        self.assertLessEqual(end['actions_done'],end['actions_max'])

    def test_partial_attempt_progress_and_early_success(self):
        import itertools
        from types import SimpleNamespace
        from proofgame.foundation.evaluation import _evaluate_one
        from proofgame.foundation.compiler import compile_halt,replay
        row={'machine':self.m.to_json(),'states':2}
        timing={'inference_seconds':1.,'game_seconds':.1,'observation_seconds':.1}
        def unsuccessful(policy,row,budget,seed,progress):
            progress(32,timing)
            return {'actions':['reset']*40,'status':'EXHAUSTED','certificate':None,'timing':timing}
        events=[]
        with patch('proofgame.foundation.training.rollout',side_effect=unsuccessful), \
             patch('proofgame.foundation.evaluation.time',SimpleNamespace(monotonic=itertools.count().__next__)):
            result=_evaluate_one(None,0,row,40,2,42,events.append)
        self.assertTrue(any(e['actions_done']==32 and e['attempts_done']==0 and not e['done'] for e in events))
        self.assertEqual(events[-1]['actions_done'],80)
        self.assertEqual(events[-1]['statuses'],{'EXHAUSTED':2})
        self.assertEqual(result['attempts'],2)
        cert=compile_halt(self.m,1)
        success={'actions':replay(cert,budget=64),'status':'SOLVED','certificate':cert,'timing':timing}
        with patch('proofgame.foundation.training.rollout',return_value=success) as mocked:
            result=_evaluate_one(None,2,row,64,100,42,events.append)
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[3],200042)
        self.assertEqual(result['attempts'],1)
        self.assertEqual(events[-1]['answer'],'HALT')

if __name__=='__main__':unittest.main()
