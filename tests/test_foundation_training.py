import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
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
            torch.save({'language':'proofgame-0.1'},root/'old.pt')
            a=parser().parse_args(['train','--data',str(data),'--out',str(root/'bad'),'--init',str(root/'old.pt')])
            with self.assertRaises(ValueError):init(a,'rl')

if __name__=='__main__':unittest.main()
