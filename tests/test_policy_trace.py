"""Full human-readable policy traces every tenth training/evaluation batch."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from proofgame.epoch_train import execute_machine
from proofgame.population_sample import from_rank
from proofgame.trace_report import build_trace_report,format_trace_report,write_trace_report


class ScriptedSampler:
    def __init__(self,actions):
        self.actions=iter(actions)
        self.hits=self.misses=0
    def probabilities(self,observation):
        self.misses+=1
        desired=next(self.actions)
        names=[a['id'] for a in observation['actions']]
        assert desired in names,(desired,names)
        # Include a meaningful second-best choice in the trace.
        values=[0.01]*len(names)
        values[names.index(desired)]=0.8
        total=sum(values)
        return tuple(x/total for x in values)


class TraceTests(unittest.TestCase):
    def test_success_and_failure_are_complete_and_readable(self):
        machine=from_rank(8)
        row={'machine':machine.to_json(),'machine_id':'test-machine'}
        success_actions=['choose:HALT','rule:exI','put:s','put:z',
                         'commit','rule:refl','submit']
        success=execute_machine(row,17,1,20,0,True,ScriptedSampler(success_actions),
                                trace_diagnostics=True)
        self.assertEqual(len(success['successes']),1)
        good=success['diagnostic_traces']['success']
        self.assertEqual(len(good['steps']),len(success_actions))
        self.assertEqual([x['action'] for x in good['steps']],success_actions)
        self.assertEqual(good['status'],'SOLVED')
        self.assertEqual(good['certificate']['answer'],'HALT')
        self.assertTrue(all(0 < x['probability'] <= 1 for x in good['steps']))
        self.assertTrue(all(x['alternatives'] for x in good['steps']))

        failure=execute_machine(row,19,1,4,0,False,
                                ScriptedSampler(['reset']*4),
                                trace_diagnostics=True)
        self.assertFalse(failure['successes'])
        bad=failure['diagnostic_traces']['failure']
        self.assertEqual(len(bad['steps']),4)
        self.assertEqual(bad['status'],'EXHAUSTED')
        report=build_trace_report([success,failure],'train',1,10,0)
        readable=format_trace_report(report)
        self.assertIn('SUCCESSFUL PROOF',readable)
        self.assertIn('UNSUCCESSFUL ATTEMPT',readable)
        self.assertIn('rule:exI',readable)
        self.assertIn('Top choices:',readable)
        self.assertIn('Episode status: SOLVED',readable)
        self.assertIn('Episode status: EXHAUSTED',readable)
        self.assertEqual(report['success']['trace']['certificate']['answer'],'HALT')
        with tempfile.TemporaryDirectory() as tmp:
            path=write_trace_report(tmp,report)
            self.assertEqual(path.read_text(),readable)
            self.assertEqual(json.loads(path.with_suffix('.json').read_text()),report)

    def test_empty_sample_and_fallback_are_labeled(self):
        row={'machine_id':'a','machine':{'rows':[]},'policy_version':0,
             'attempts':1,'successes':[],
             'diagnostic_traces':{'success':None,'failure':None}}
        report=build_trace_report([row],'eval',0,10,0)
        self.assertIsNone(report['success'])
        self.assertIsNone(report['failure'])
        self.assertIn('No such attempt',format_trace_report(report))

    def test_real_runner_emits_only_tenth_batch(self):
        from proofgame.unfiltered_dataset import build_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'data.json'
            dataset.write_text(json.dumps(build_dataset(10,1,4,12345)))
            out=root/'run'
            cmd=[sys.executable,'-m','proofgame.epoch_train',
                 '--dataset',str(dataset),'--out',str(out),
                 '--epochs','1','--attempts','1','--batch-size','1',
                 '--workers','1','--seed','7']
            completed=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
            self.assertEqual(completed.returncode,0,
                             completed.stdout+'\n'+completed.stderr)
            files=list((out/'policy_traces').glob('*.txt'))
            self.assertEqual(len(files),1)
            self.assertEqual(files[0].name,'epoch_01_train_batch_0010.txt')
            self.assertIn('POLICY TRACE',files[0].read_text())
            self.assertIn('policy_trace_saved',completed.stdout)
            # Traces must not contaminate training batches or gradients.
            batch=json.loads((out/'epoch_01_train'/'batch_0009.json').read_text())
            self.assertNotIn('diagnostic_traces',batch['machines'][0])


if __name__=='__main__':
    unittest.main()
