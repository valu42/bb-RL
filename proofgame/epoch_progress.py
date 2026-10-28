"""Dependency-free terminal progress and measured, phase-local ETA.

Count committed machines after rollout, learning and checkpointing. Resume
starts a new timing sample so time spent with the process stopped is excluded.
"""
import sys
import time


def duration(seconds):
    if seconds is None:
        return 'calculating'
    seconds = max(0, int(seconds + .5))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes}m {seconds:02d}s'


class EpochProgress:
    def __init__(self, epoch, phase, total, completed=0, *, stream=None, clock=None):
        self.epoch, self.phase, self.total = epoch, phase, total
        self.initial = self.completed = completed
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        self.started = self.clock()

    def snapshot(self):
        elapsed = max(0., self.clock() - self.started)
        measured = self.completed - self.initial
        remaining = self.total - self.completed
        eta = (0. if remaining == 0 else
               elapsed / measured * remaining if measured > 0 else None)
        return {'epoch':self.epoch, 'phase':self.phase,
                'completed':self.completed, 'total':self.total,
                'percent':100. * self.completed / self.total if self.total else 100.,
                'elapsed_seconds':round(elapsed, 1),
                'eta_seconds':round(eta, 1) if eta is not None else None,
                'eta_scope':'current phase, including rollout, learning and checkpoints; excludes later phases',
                'timing_sample_machines':measured}

    def show(self, completed=None, *, heartbeat=False):
        if completed is not None:
            if not self.completed <= completed <= self.total:
                raise ValueError('progress must advance within the phase total')
            self.completed = completed
        data = self.snapshot()
        # Redirected logs get one clean line per batch, without carriage
        # returns or repeated heartbeat lines. Interactive terminals refresh.
        interactive = bool(getattr(self.stream, 'isatty', lambda: False)())
        if heartbeat and not interactive:
            return data
        filled = min(24, int(data['percent'] * 24 / 100))
        bar = '█' * filled + '░' * (24 - filled)
        label = 'Baseline' if self.phase == 'baseline' else f'Epoch {self.epoch} {self.phase}'
        line = (f'{label} [{bar}] {data["percent"]:5.1f}% '
                f'{self.completed}/{self.total} machines | '
                f'elapsed {duration(data["elapsed_seconds"])} | '
                f'{self.phase} ETA ~{duration(data["eta_seconds"])}')
        # Complete lines keep stderr bars readable beside stdout JSON logs.
        print(line, file=self.stream, flush=True)
        return data
