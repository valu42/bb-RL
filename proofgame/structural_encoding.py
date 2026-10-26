"""Syntactic tree encoding with learned, non-semantic attention relations.

JSON field/list structure and parsed proof-expression trees are exposed to a
Transformer. No proof outcomes, reachability analysis, hard-coded action scores,
or mathematical oracle are supplied. Unknown lexical atoms use UTF-8 bytes,
not hash buckets, so unseen names remain representable without OOV collisions.
"""
import json
import math
import re
from collections import defaultdict
from functools import lru_cache

import torch
from torch import nn

from .logic import Invalid, parse

# Only known symbols and syntactic fields get atomic IDs; unknown names use bytes.
_ATOMS = frozenset('''
    machine rows answer status actions_left limits nodes depth focus goals path
    variables facts goal proof draft last_error actions id expression rule data
    slot sort bound children format target left right attempts state write move
    q0 q1 qH b0 b1 L R HALT NONHALT ACTIVE SOLVED EXHAUSTED MODEL_LIMIT
    Nat Bit List State Cfg false true null
    all ex and or imp eq var bound z s add mul nil cons head tail cfg
    qof next at qnext choose reset submit commit cancel focus put copy
    rule cut reduce raa casesBit casesState allI exI impI andI
    orL orR refl natI listI exact falseE apply destruct specialize
    inject discriminate rewrite forward backward
    expression actions_left max_nodes max_depth
'''.split())
_TYPES = ('PAD', 'ROOT', 'OBJECT', 'ARRAY', 'FIELD', 'TEXT', 'NUMBER', 'NONE',
          'BOOL_TRUE', 'BOOL_FALSE', 'EXPR', 'BOUND_VAR', 'FREE_VAR')
_ATOM_IDS = {word: len(_TYPES) + n for n, word in enumerate(sorted(_ATOMS))}
_BYTE_OFFSET = len(_TYPES) + len(_ATOM_IDS)
VOCAB_SIZE = _BYTE_OFFSET + 256
_TYPE_IDS = {word: i for i, word in enumerate(_TYPES)}
_LEX = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|\s+|[^\s]')

# All nine importance parameters are learned from scratch.
OTHER, SELF, PARENT, CHILD, SIBLING, ANCESTOR, DESCENDANT, BINDS, BOUND_BY = range(9)
NUM_RELATIONS = 9


def _lexical_ids(value):
    for token in _LEX.findall(value):
        if token in _ATOM_IDS:
            yield _ATOM_IDS[token]
        else:
            for byte in token.encode('utf-8'):
                yield _BYTE_OFFSET + byte


def encode_tree(value):
    """Return (tokens, parents, variable-binding edges), preserving list order."""
    tokens = []
    parents = []
    bindings = []

    def add(token, parent):
        index = len(tokens)
        tokens.append(token)
        parents.append(parent)
        return index

    root = add(_TYPE_IDS['ROOT'], -1)

    def atom(text, parent):
        for token in _lexical_ids(text):
            add(token, parent)

    def expression(node, parent, binders=()):
        if isinstance(node, tuple):
            if len(node) == 2 and node[0] in ('var', 'bound') and isinstance(node[1], int):
                idx = add(_TYPE_IDS['FREE_VAR' if node[0] == 'var' else 'BOUND_VAR'], parent)
                atom(str(node[1]), idx)
                if node[0] == 'bound' and node[1] < len(binders):
                    bindings.append((binders[node[1]], idx))
                return
            idx = add(_TYPE_IDS['EXPR'], parent)
            if node:
                atom(str(node[0]), idx)
                body_binders = (idx,) + binders if node[0] in ('all', 'ex') else binders
                for position, child in enumerate(node[1:]):
                    # Quantifier scope applies to its body, not its sort annotation.
                    scope = body_binders if node[0] in ('all', 'ex') and position == 1 else binders
                    expression(child, idx, scope)
        else:
            atom(str(node), parent)

    def walk(node, parent, field=None, within_facts=False):
        if isinstance(node, dict):
            obj = add(_TYPE_IDS['OBJECT'], parent)
            for name, child in node.items():
                key_node = add(_TYPE_IDS['FIELD'], obj)
                atom(str(name), key_node)
                walk(child, key_node, str(name), within_facts or field == 'facts')
        elif isinstance(node, (list, tuple)):
            array = add(_TYPE_IDS['ARRAY'], parent)
            for child in node:
                walk(child, array, field, within_facts)
        elif isinstance(node, str):
            if field in ('goal', 'expression') or within_facts:
                try:
                    expression(parse(node), parent)
                    return
                except (Invalid, ValueError, RecursionError):
                    pass  # Partially edited or non-formula text is still retained.
            string = add(_TYPE_IDS['TEXT'], parent)
            atom(node, string)
        elif isinstance(node, bool):
            add(_TYPE_IDS['BOOL_TRUE' if node else 'BOOL_FALSE'], parent)
        elif node is None:
            add(_TYPE_IDS['NONE'], parent)
        elif isinstance(node, (int, float)):
            number = add(_TYPE_IDS['NUMBER'], parent)
            atom(str(node), number)
        else:
            raise TypeError('unsupported syntactic value: ' + type(node).__name__)

    walk(value, root)
    return tokens, parents, bindings


@lru_cache(maxsize=16384)
def cached_tree(serialized):
    """Cache parsed syntax for repeated action descriptions and proof states."""
    tokens, parents, bindings = encode_tree(json.loads(serialized))
    return tuple(tokens), tuple(parents), tuple(bindings)


def relation_ids(parents, bindings):
    """Construct directed structural relations with indexed CPU assignments.

    Equivalent to the original scalar loop, without one torch dispatch per
    parent/ancestor/sibling edge. Binder links retain highest precedence.
    """
    count = len(parents)
    rel = torch.full((count, count), OTHER, dtype=torch.long)
    children = defaultdict(list)
    parent_edges = []
    ancestors = []
    for i, parent in enumerate(parents):
        if parent < 0:
            continue
        children[parent].append(i)
        parent_edges.append((parent, i))
        ancestor = parents[parent]
        while ancestor >= 0:
            ancestors.append((ancestor, i))
            ancestor = parents[ancestor]
    for siblings in children.values():
        if len(siblings) > 1:
            idx = torch.tensor(siblings, dtype=torch.long)
            rel[idx[:, None], idx[None, :]] = SIBLING
    if ancestors:
        pairs = torch.tensor(ancestors, dtype=torch.long)
        rel[pairs[:, 0], pairs[:, 1]] = ANCESTOR
        rel[pairs[:, 1], pairs[:, 0]] = DESCENDANT
    if parent_edges:
        pairs = torch.tensor(parent_edges, dtype=torch.long)
        rel[pairs[:, 0], pairs[:, 1]] = PARENT
        rel[pairs[:, 1], pairs[:, 0]] = CHILD
    rel.fill_diagonal_(SELF)
    if bindings:
        pairs = torch.tensor(bindings, dtype=torch.long)
        rel[pairs[:, 0], pairs[:, 1]] = BINDS
        rel[pairs[:, 1], pairs[:, 0]] = BOUND_BY
    return rel

class StructuralEncoder(nn.Module):
    """Global attention with additive, learned relation biases (zero initialized)."""
    def __init__(self, width, layers, heads, max_tokens):
        super().__init__()
        self.max_tokens = max_tokens
        self.heads = heads
        self.embedding = nn.Embedding(VOCAB_SIZE, width, padding_idx=0)
        self.relationship = nn.Embedding(NUM_RELATIONS, heads)
        nn.init.zeros_(self.relationship.weight)  # Begin without structural preference.
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        for block in self.encoder.layers:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, texts):
        from .model import ModelLimit
        trees = [cached_tree(text) for text in texts]
        size = max(len(tokens) for tokens, _, _ in trees)
        if size > self.max_tokens:
            raise ModelLimit(f'input needs {size} structural tokens; maximum is {self.max_tokens}')
        device = self.embedding.weight.device
        # Coalesce dynamic sequence lengths to reduce MPS kernel compilation.
        if device.type == 'mps':
            size = min(self.max_tokens, ((size + 15) // 16) * 16)
        batch = len(trees)
        ids = torch.zeros((batch, size), dtype=torch.long, device=device)
        relations = torch.full((batch, size, size), OTHER, dtype=torch.long, device=device)
        for i, (tokens, parents, binds) in enumerate(trees):
            n = len(tokens)
            ids[i, :n] = torch.tensor(tokens, dtype=torch.long, device=device)
            relations[i, :n, :n] = relation_ids(parents, binds).to(device)
        x = self.embedding(ids) * math.sqrt(self.embedding.embedding_dim)
        positions = torch.arange(size, dtype=x.dtype, device=device).unsqueeze(1)
        scale = torch.exp(torch.arange(0, x.shape[-1], 2, device=device, dtype=x.dtype)
                          * (-math.log(10000.) / x.shape[-1]))
        pe = torch.zeros((size, x.shape[-1]), dtype=x.dtype, device=device)
        pe[:, 0::2] = torch.sin(positions * scale)
        pe[:, 1::2] = torch.cos(positions * scale)
        biases = self.relationship(relations).permute(0, 3, 1, 2).contiguous()
        # Padded keys cannot affect valid tokens; padded queries are never pooled.
        biases = biases.masked_fill(ids.eq(0)[:, None, None, :], -1e4)
        mask = biases.reshape(batch * self.heads, size, size)
        return self.encoder(x + pe, mask=mask)[:, 0]
