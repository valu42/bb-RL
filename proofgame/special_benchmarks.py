"""Fixed, separate diagnostic evaluations; never used for policy gradients.

These challenge machines do not join the train/test dataset or change
machine coverage. A passed diagnostic means *the model sampled a kernel-checked
certificate*, not that a hand-authored proof exists in a fixture file.
"""
from pathlib import Path
import json

from .kernel import check
from .logic import Machine
from .tasks import key


BENCHMARKS = (
    {
        'id': 'parity_trap_2',
        'name': 'Parity trap (2-state, NONHALT)',
        'expected_answer': 'NONHALT',
        'explanation': 'Two-phase recurring configurations; requires a non-halting invariant.',
        'rows': (
            ('q1', 'b1', 'R'), ('q1', 'b1', 'R'),
            ('q0', 'b0', 'L'), ('qH', 'b1', 'R'),
        ),
    },
    {
        'id': 'busy_beaver_2_6',
        'name': 'Busy Beaver (2-state, HALT in 6 steps)',
        'expected_answer': 'HALT',
        'explanation': 'Halts after six Turing-machine transitions.',
        'rows': (
            ('q1', 'b1', 'R'), ('q1', 'b1', 'L'),
            ('q0', 'b1', 'L'), ('qH', 'b1', 'R'),
        ),
    },
)


def benchmark_definitions(suite='legacy'):
    # Keep BENCHMARKS frozen: old dataset samplers depend on its two entries.
    if suite == 'legacy':
        return BENCHMARKS
    if suite == 'extended':
        from .special_cases import ADDITIONAL_BENCHMARKS
        return BENCHMARKS + ADDITIONAL_BENCHMARKS
    raise ValueError('unknown special benchmark suite')


def benchmark_rows(suite='legacy'):
    """Only machine identities and tables are supplied to evaluation workers."""
    return [
        {
            'machine_id': key(Machine(b['rows'])),
            'machine': Machine(b['rows']).to_json(),
        }
        for b in benchmark_definitions(suite)
    ]


def format_benchmark_report(report):
    header = ('Special benchmarks — baseline (epoch 0)'
              if report['epoch'] == 0
              else f"Special benchmarks — after epoch {report['epoch']}")
    lines = ['', '=' * len(header), header, '=' * len(header)]
    for item in report['benchmarks']:
        if item['status'] == 'PROVED':
            lines.append(
                f"{item['name']}: PROVED {item['proved_answer']} "
                f"(first verified proof on attempt {item['first_proof_attempt']}/"
                f"{item['attempts_limit']}; {item['proof_actions']} proof-game actions)"
            )
        elif item['status'] == 'NOT_PROVED':
            lines.append(
                f"{item['name']}: NOT PROVED "
                f"(no verified {item['expected_answer']} proof in "
                f"{item['attempts_limit']} attempts, up to "
                f"{item['action_budget']} proof-game actions each)"
            )
        else:
            lines.append(
                f"{item['name']}: UNEXPECTED VERIFIED {item['proved_answer']} PROOF "
                f"(expected {item['expected_answer']}; investigate the fixture and checker)"
            )
        lines.extend(benchmark_notes(item))
        if item.get('verified_certificate'):
            lines.append(format_special_proof(item))
    lines.append('These are separate checks and do not change the held-out test score.')
    return '\n'.join(lines) + '\n'


def make_benchmark_report(epoch, results, attempts_limit, action_budget, trained_ids=(), suite='legacy', tested_ids=()):
    definitions = benchmark_definitions(suite)
    if len(results) != len(definitions):
        raise ValueError('incomplete special benchmark results')
    if min(attempts_limit, action_budget) <= 0:
        raise ValueError('benchmark attempt and action budgets must be positive')
    benchmarks = []
    trained_ids = set(trained_ids)
    tested_ids = set(tested_ids)
    for definition, result in zip(definitions, results):
        machine = Machine(definition['rows'])
        if result['machine_id'] != key(machine) or result['machine'] != machine.to_json():
            raise ValueError('special benchmark result does not match requested machine')
        if result['attempts'] > attempts_limit or result['attempts'] < 1:
            raise ValueError('special benchmark attempts exceed budget')
        successes = result['successes']
        if len(successes) > 1:
            raise ValueError('special benchmark must stop after its first proof')
        success = successes[0] if successes else None
        if success:
            check(success['certificate'])
            if success['certificate']['machine'] != machine.to_json():
                raise ValueError('special benchmark proof belongs to another machine')
            proven_answer = success['certificate']['answer']
            status = 'PROVED' if proven_answer == definition['expected_answer'] else 'UNEXPECTED_PROOF'
        else:
            proven_answer = None
            status = 'NOT_PROVED'
        benchmarks.append({
            'id': definition['id'],
            'name': definition['name'],
            'expected_answer': definition['expected_answer'],
            'status': status,
            'proved_answer': proven_answer,
            'attempts_limit': attempts_limit,
            'attempts_used': result['attempts'],
            'action_budget': action_budget,
            'first_proof_attempt': success['attempt'] if success else None,
            'proof_actions': len(success['actions']) if success else None,
            'verified_certificate': success['certificate'] if success else None,
            'sampled_action_trace': success['actions'] if success else None,
            'machine_seen_in_training_set': result['machine_id'] in trained_ids,
            'machine_seen_in_evaluation_set': result['machine_id'] in tested_ids,
            'machine_id': result['machine_id'],
            'reference_note': definition.get('reference_note'),
            'checker_errors': result.get('checker_errors', {}),
            'sampled_proof_details': result.get('diagnostic_traces', {}).get('success'),
        })
    return {
        'epoch': epoch,
        'suite': suite,
        'evaluation_only': True,
        'uses_training_gradients': False,
        'benchmarks': benchmarks,
    }


def write_benchmark_report(output_dir, report, *, verbose=True):
    """Checkpoint-safe epoch file, plus a fully regenerated readable history."""
    from .epoch_train import atomic_json
    output_dir = Path(output_dir)
    epoch = report['epoch']
    atomic_json(output_dir / f'special_benchmarks_epoch_{epoch:02d}.json', report)
    message = format_benchmark_report(report)
    path = output_dir / f'special_benchmarks_epoch_{epoch:02d}.txt'
    tmp = path.with_suffix('.txt.tmp')
    tmp.write_text(message)
    tmp.replace(path)
    texts = []
    for file in sorted(output_dir.glob('special_benchmarks_epoch_*.json')):
        texts.append(format_benchmark_report(json.loads(file.read_text())))
    history = output_dir / 'special_benchmarks_history.txt'
    tmp = history.with_suffix('.txt.tmp')
    tmp.write_text(''.join(texts))
    tmp.replace(history)
    print(message if verbose else format_compact_report(report), flush=True)


def benchmark_notes(item):
    notes = []
    if item.get('machine_seen_in_training_set'):
        notes.append('  Overlap: this machine is in the training set; not a held-out diagnostic.')
    if item.get('machine_seen_in_evaluation_set'):
        notes.append('  Overlap: this machine also appears in the regular evaluation set.')
    if item.get('reference_note') and not item.get('verified_certificate'):
        notes.append('  Reference limitation: ' + item['reference_note'])
    limits = {error: count for error, count in item.get('checker_errors', {}).items()
              if 'limit' in error.lower() or 'budget exceeded' in error.lower()
              or 'recursion' in error.lower()}
    if limits:
        notes.append('  Observed checker resource limits: ' + json.dumps(limits, sort_keys=True))
    return notes


def format_special_proof(item):
    certificate = item['verified_certificate']
    lines = [f"--- VERIFIED SPECIAL PROOF: {item['name']} ---",
             f"Machine: {item['machine_id']}",
             'Transition rows: ' + json.dumps(certificate['machine']['rows']),
             f"Answer: {certificate['answer']} | attempt {item['first_proof_attempt']} | {item['proof_actions']} proof actions"]
    details = item.get('sampled_proof_details') or {}
    steps = details.get('steps') or item.get('sampled_action_trace') or []
    for index, step in enumerate(steps, 1):
        probability = step.get('probability')
        suffix = f" — {100*probability:.2f}%" if probability is not None else ''
        lines.append(f"{index:3}. {step['action']}{suffix}")
        if step.get('goal'):
            lines.append('     Goal: ' + ' | '.join(step['goal']))
        if step.get('draft'):
            lines.append('     Building: ' + str(step['draft'].get('expression', '')))
        if step.get('certificate_candidate'):
            lines.append('     Candidate: ' + json.dumps(step['certificate_candidate'], sort_keys=True))
        if step.get('error_after'):
            lines.append('     Checker error: ' + step['error_after'])
    lines.extend(['Accepted proof tree:', json.dumps(certificate['proof'], indent=2)])
    return '\n'.join(lines)


def format_compact_report(report):
    lines = []
    for item in report['benchmarks']:
        length = f" ({item['proof_actions']} proof actions)" if item.get('verified_certificate') else ''
        lines.append(f"Special eval {report['epoch']}: {item['name']}: {item['status']}{length}")
        lines.extend(benchmark_notes(item))
        if item.get('verified_certificate'):
            lines.append(format_special_proof(item))
    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Display saved special results and verified proofs; no model inference.')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    for item in report['benchmarks']:
        if item.get('verified_certificate'):
            check(item['verified_certificate'])
    print(format_benchmark_report(report), flush=True)


if __name__ == '__main__':
    main()
