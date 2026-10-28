"""Readable, reproducible policy-decision traces for sampled rollout batches.

The traces come from the *actual frozen rollout policy*, before the batch
optimizer step. They are diagnostic only and never enter REINFORCE replay.
"""
import json
from pathlib import Path


def build_trace_report(results, phase, epoch, batch_number, policy_version):
    proved = next((r for r in results if r.get('diagnostic_traces',{}).get('success')),None)
    unproved = next((r for r in results
                     if not r['successes'] and r.get('diagnostic_traces',{}).get('failure')),None)
    fallback = False
    if unproved is None:
        unproved = next((r for r in results
                         if r.get('diagnostic_traces',{}).get('failure')),None)
        fallback = unproved is not None
    def pick(row, kind):
        if row is None:
            return None
        return {'machine_id':row['machine_id'],
                'machine':row['machine'],
                'machine_proved':bool(row['successes']),
                'policy_version':row['policy_version'],
                'attempts_used':row['attempts'],
                'trace':row['diagnostic_traces'][kind]}
    return {'phase':phase,'epoch':epoch,'batch_number':batch_number,
            'policy_version':policy_version,
            'success':pick(proved,'success'),
            'failure':pick(unproved,'failure'),
            'failure_is_from_proved_machine':fallback}


def format_trace_report(report):
    lines=[f"POLICY TRACE — {report['phase']} epoch {report['epoch']}, "
           f"batch {report['batch_number']} (frozen policy version {report['policy_version']})",
           '='*78,
           'Probabilities are sampling probabilities at that step, not chances that a theorem is true.',
           'Top alternatives are the three highest-probability actions at that state.',
           'Forced initial answers have probability 100% by design, not by model preference.',
           '']
    for key,title in (('success','SUCCESSFUL PROOF'),('failure','UNSUCCESSFUL ATTEMPT')):
        item=report[key]
        lines.append(f"--- {title} ---")
        if item is None:
            lines.append('No such attempt in this batch.')
            lines.append('')
            continue
        if key=='failure' and report['failure_is_from_proved_machine']:
            lines.append('No entirely unproved machine in this batch; showing a failed attempt on a proved machine.')
        trace=item['trace']
        lines.extend([
            f"Machine: {item['machine_id']}",
            f"Transition rows (q0/0, q0/1, q1/0, q1/1, ...): {item['machine']['rows']}",
            f"Attempt: {trace['attempt']} of {item['attempts_used']} executed",
            f"Outcome: {trace['status']} | Actions: {len(trace['steps'])}",
            ''
        ])
        for step in trace['steps']:
            chosen=step['action']
            prob=100*step['probability']
            raw=100*step['raw_policy_probability']
            mode=' [FORCED]' if step['forced'] else ''
            lines.append(f"{step['step']:>3}. {chosen} — {prob:.2f}%{mode}"
                         f" (raw model {raw:.2f}%)")
            if step['goal']:
                goal=' | '.join(step['goal'])
                lines.append('     Goal: '+goal[:240]+('…' if len(goal)>240 else ''))
            if step['draft']:
                lines.append('     Building: '+str(step['draft'].get('expression',''))[:200])
            if step.get('certificate_candidate'):
                lines.append('     Certificate candidate: '+
                             json.dumps(step['certificate_candidate'],sort_keys=True)[:240])
            top=', '.join(f"{x['action']} {100*x['probability']:.1f}%"
                          for x in step['alternatives'])
            lines.append('     Top choices: '+top)
            if step['error_after']:
                lines.append('     Checker error: '+step['error_after'])
            if step['status_after']!='ACTIVE':
                lines.append('     Episode status: '+step['status_after'])
        if trace['certificate']:
            lines.append('Verified certificate answer: '+trace['certificate']['answer'])
        lines.append('')
    return '\n'.join(lines)+'\n'


def write_trace_report(out, report, *, verbose=True):
    folder=Path(out)/'policy_traces'
    folder.mkdir(exist_ok=True)
    stem=f"epoch_{report['epoch']:02d}_{report['phase']}_batch_{report['batch_number']:04d}"
    json_path=folder/(stem+'.json')
    txt_path=folder/(stem+'.txt')
    for path,body in ((json_path,json.dumps(report,indent=2)+'\n'),
                      (txt_path,format_trace_report(report))):
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(body)
        tmp.replace(path)
    if verbose:
        print(format_trace_report(report),flush=True)
    return txt_path
