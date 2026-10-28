"""Cumulative, machine-weighted proof lengths for one epoch phase."""
import json


class ProofProgress:
    def __init__(self):
        self.completed = 0
        self.counts = dict.fromkeys(('HALT', 'NONHALT'), 0)
        self.actions = dict.fromkeys(('HALT', 'NONHALT'), 0)
        self.by_states = {}

    def add(self, rows):
        for row in rows:
            self._add_one(row)
            if 'machine' in row:
                states = str(len(row['machine']['rows']) // 2)
                if states not in self.by_states:
                    self.by_states[states] = ProofProgress()
                self.by_states[states]._add_one(row)

    def _add_one(self, row):
        self.completed += 1
        if not row['successes']:
            return
        first = min(row['successes'], key=lambda proof: proof['attempt'])
        answer = first['certificate']['answer']
        self.counts[answer] += 1
        self.actions[answer] += len(first['actions'])

    @classmethod
    def restore(cls, folder, completed):
        result = cls()
        for path in sorted(folder.glob('batch_*.json')):
            batch = json.loads(path.read_text())
            start = batch['start_index']
            if start >= completed:
                continue  # A batch file may precede its state checkpoint.
            if start != result.completed or start + len(batch['machines']) > completed:
                raise ValueError('noncontiguous committed proof-progress batches')
            result.add(batch['machines'])
        if result.completed != completed:
            raise ValueError('missing committed proof-progress batches')
        return result

    def snapshot(self):
        return {
            'machines': self.completed,
            'machines_proved_by_answer': dict(self.counts),
            'machines_unproved': self.completed - sum(self.counts.values()),
            'mean_first_proof_actions_by_answer': {
                answer: self.actions[answer] / count if count else None
                for answer, count in self.counts.items()},
            'proof_length_scope': 'first verified proof per solved machine, current phase',
            **({'by_states': {n: item.snapshot() for n, item in sorted(self.by_states.items())}}
               if self.by_states else {}),
        }

    def text(self):
        means = self.snapshot()['mean_first_proof_actions_by_answer']
        parts = []
        for answer, count in self.counts.items():
            mean = f'{means[answer]:.1f}' if count else 'n/a'
            parts.append(f'{answer} {count} (avg {mean} proof actions)')
        parts.append(f'unproved {self.completed - sum(self.counts.values())}')
        return ' | '.join(parts)

    def state_lines(self):
        return [f'{n}-state ({item.completed} machines): {item.text()}'
                for n, item in sorted(self.by_states.items())]
