"""Input-only structural encoding caches, optionally used in CPU rollout.

The topology cache key is derived from parent/binder links, NOT observation
values. Numeric budget changes and other lexical edits can therefore reuse
the same structural relations. No learned masks, neural features, logits, or
optimizer-dependent tensors are cached. Cache bytes are explicitly bounded.
"""
from collections import OrderedDict
from contextlib import contextmanager
import math
import time

import torch

from .structural_encoding import OTHER, cached_tree, relation_ids


class StructuralPreprocessingCache:
    def __init__(self, relation_cache_mb=128, position_cache_mb=8,
                 topology_keys=False, compact_relations=False):
        if relation_cache_mb < 0 or position_cache_mb < 0:
            raise ValueError('cache sizes must be nonnegative')
        self.topology_keys = topology_keys
        self.compact_relations = compact_relations
        # Dict keys are Python tuples and are not counted in tensor nbytes.
        # Bound entry count as well as tensor bytes to cap bookkeeping memory.
        self.relation_entry_limit = 8192 if topology_keys else None
        self.relation_limit = int(relation_cache_mb * 1024 * 1024)
        self.position_limit = int(position_cache_mb * 1024 * 1024)
        self.relations = OrderedDict()
        self.positions = OrderedDict()
        self.relation_bytes = 0
        self.position_bytes = 0
        self.hits = {'relation': 0, 'position': 0}
        self.misses = {'relation': 0, 'position': 0}
        self.timings = {'tree_seconds': 0., 'tensor_setup_seconds': 0.,
                        'relationship_seconds': 0., 'encoder_seconds': 0.}

    @staticmethod
    def _bytes(tensor):
        return tensor.numel() * tensor.element_size()

    def _get(self, entries, key, kind):
        if key in entries:
            self.hits[kind] += 1
            entries.move_to_end(key)
            return entries[key]
        self.misses[kind] += 1
        return None

    def _insert(self, entries, key, value, kind, limit):
        size = self._bytes(value)
        if size > limit:
            return
        entries[key] = value
        current = self.relation_bytes if kind == 'relation' else self.position_bytes
        current += size
        while (current > limit or
               (kind == 'relation' and self.relation_entry_limit is not None
                and len(entries) > self.relation_entry_limit)):
            _, old = entries.popitem(last=False)
            current -= self._bytes(old)
        if kind == 'relation':
            self.relation_bytes = current
        else:
            self.position_bytes = current

    def relation(self, serialized, parents, bindings):
        # Structural relation IDs depend only on tree edges and binder links,
        # not on lexical tokens, numeric values, or model weights.
        key = (parents, bindings) if self.topology_keys else serialized
        value = self._get(self.relations, key, 'relation')
        if value is None:
            value = relation_ids(parents, bindings)
            if self.compact_relations:
                # All relation IDs fit in one byte; convert back to int64
                # before nn.Embedding. Saves up to 8x cache storage.
                value = value.to(dtype=torch.uint8)
            self._insert(self.relations, key, value, 'relation',
                         self.relation_limit)
        return value

    def positional(self, length, width, device, dtype):
        key = (length, width, str(device), dtype)
        value = self._get(self.positions, key, 'position')
        if value is None:
            pos = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
            scale = torch.exp(torch.arange(0, width, 2, dtype=dtype, device=device)
                              * (-math.log(10000.) / width))
            value = torch.zeros((length, width), dtype=dtype, device=device)
            value[:, 0::2] = torch.sin(pos * scale)
            value[:, 1::2] = torch.cos(pos * scale)
            self._insert(self.positions, key, value, 'position',
                         self.position_limit)
        return value

    def forward(self, encoder, texts):
        """Numerically equivalent AST encoder, with cached CPU relation IDs."""
        from .model import ModelLimit
        started = time.perf_counter()
        trees = [cached_tree(text) for text in texts]
        size = max(len(tokens) for tokens, _, _ in trees)
        if size > encoder.max_tokens:
            raise ModelLimit(
                f'input needs {size} structural tokens; maximum is {encoder.max_tokens}')
        self.timings['tree_seconds'] += time.perf_counter() - started

        started = time.perf_counter()
        device = encoder.embedding.weight.device
        if device.type == 'mps':
            size = min(encoder.max_tokens, ((size + 15) // 16) * 16)
        ids = torch.zeros((len(trees), size), dtype=torch.long, device=device)
        relations = torch.full((len(trees), size, size), OTHER,
                               dtype=torch.long, device=device)
        for i, (text, tree) in enumerate(zip(texts, trees)):
            tokens, parents, bindings = tree
            n = len(tokens)
            ids[i, :n] = torch.tensor(tokens, dtype=torch.long, device=device)
            relations[i, :n, :n] = self.relation(text, parents, bindings).to(
                device=device, dtype=torch.long)
        self.timings['tensor_setup_seconds'] += time.perf_counter() - started

        started = time.perf_counter()
        x = encoder.embedding(ids) * math.sqrt(encoder.embedding.embedding_dim)
        pe = self.positional(size, x.shape[-1], device, x.dtype)
        biases = encoder.relationship(relations).permute(0, 3, 1, 2).contiguous()
        biases = biases.masked_fill(ids.eq(0)[:, None, None, :], -1e4)
        mask = biases.reshape(len(trees) * encoder.heads, size, size)
        self.timings['relationship_seconds'] += time.perf_counter() - started

        started = time.perf_counter()
        pooled = encoder.encoder(x + pe, mask=mask)[:, 0]
        self.timings['encoder_seconds'] += time.perf_counter() - started
        return pooled

    def stats(self):
        return {
            'relation_cache_bytes': self.relation_bytes,
            'topology_keys': self.topology_keys,
            'compact_relations': self.compact_relations,
            'relation_entry_limit': self.relation_entry_limit,
            'position_cache_bytes': self.position_bytes,
            'relation_entries': len(self.relations),
            'position_entries': len(self.positions),
            'hits': dict(self.hits),
            'misses': dict(self.misses),
            'timings': dict(self.timings),
            'timing_note': 'Substage timers are CPU wall time; on MPS they '
                           'measure dispatch time, not synchronized GPU execution.',
        }


@contextmanager
def cached_structural_forward(policy, *, relation_cache_mb=128,
                              position_cache_mb=8):
    """Temporarily substitute a structurally equivalent forward in benchmarks."""
    if policy.representation != 'ast':
        raise ValueError('structural preprocessing cache requires AST representation')
    encoder = policy.structural_encoder
    original = encoder.forward
    cache = StructuralPreprocessingCache(
        relation_cache_mb=relation_cache_mb, position_cache_mb=position_cache_mb)
    try:
        encoder.forward = lambda texts: cache.forward(encoder, texts)
        yield cache
    finally:
        encoder.forward = original
