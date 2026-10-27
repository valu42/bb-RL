"""Opt-in first-output-only AST encoder. No model/checkpoint format changes.

For exactly one Transformer layer, output token 0 needs all input K/V tokens,
only Q[0], and only token 0's residual/norm/feed-forward calculation. This is
an algebraic pruning of unused outputs, not sparse/approximate attention.

The learned relation bias is still applied, but only its query-0 row is needed.
All parameter gradients flow through ordinary PyTorch autograd. Reject other
architectures rather than silently applying the one-layer derivation to them.
"""
from contextlib import contextmanager
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from .structural_encoding import (
    ANCESTOR, BINDS, BOUND_BY, PARENT, SELF, OTHER, cached_tree,
)


def root_relation_ids(parents, bindings):
    """Exactly relation_ids(parents, bindings)[0], using O(N) storage.

    encode_tree emits a connected tree in parent-before-child order with root
    at zero. The root has no siblings/parent. Its remaining relations are
    PARENT to its direct children, ANCESTOR to other descendants, and SELF.
    Binding assignments retain the full implementation's final precedence.
    """
    if not parents or parents[0] != -1:
        raise ValueError('expected a nonempty AST rooted at token zero')
    row = [SELF]
    for i, parent in enumerate(parents[1:], 1):
        if not 0 <= parent < i:
            raise ValueError('expected a connected, parent-before-child AST')
        row.append(PARENT if parent == 0 else ANCESTOR)
    # Full relation_ids writes all BINDS edges, then all BOUND_BY edges.
    for binder, variable in bindings:
        if not (0 <= binder < len(row) and 0 <= variable < len(row)):
            raise ValueError('binding index outside AST')
        if binder == 0:
            row[variable] = BINDS
    for binder, variable in bindings:
        if variable == 0:
            row[binder] = BOUND_BY
    return torch.tensor(row, dtype=torch.long)


def validate_layer(layer):
    if not isinstance(layer, nn.TransformerEncoderLayer):
        raise ValueError('root-query requires a standard TransformerEncoderLayer')
    attention = layer.self_attn
    if (not attention.batch_first or attention.in_proj_weight is None
            or attention.kdim != attention.embed_dim
            or attention.vdim != attention.embed_dim
            or attention.bias_k is not None or attention.bias_v is not None
            or attention.add_zero_attn):
        raise ValueError('unsupported attention configuration')
    if attention.dropout != 0. or any(
            module.p != 0. for module in layer.modules()
            if isinstance(module, nn.Dropout)):
        raise ValueError('root-query experiment requires zero dropout')


def root_layer_forward(layer, x, bias):
    """Return layer(x, mask=... )[:, :1] for one zero-dropout layer.

    x: [batch, tokens, width]. bias: [batch, heads, 1, tokens], including
    the original additive padding mask. Pre/post-norm layouts are supported.
    """
    attention = layer.self_attn
    width = attention.embed_dim
    heads = attention.num_heads
    batch, length, _ = x.shape
    z = layer.norm1(x) if layer.norm_first else x
    weight = attention.in_proj_weight
    projection_bias = attention.in_proj_bias
    q = F.linear(z[:, :1], weight[:width],
                 None if projection_bias is None else projection_bias[:width])
    kv = F.linear(z, weight[width:],
                  None if projection_bias is None else projection_bias[width:])
    k, v = kv.split(width, dim=-1)
    q = q.reshape(batch, 1, heads, width // heads).transpose(1, 2)
    k = k.reshape(batch, length, heads, width // heads).transpose(1, 2)
    v = v.reshape(batch, length, heads, width // heads).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                      dropout_p=0., is_causal=False)
    y = y.transpose(1, 2).reshape(batch, 1, width)
    y = attention.out_proj(y)
    root = x[:, :1] + layer.dropout1(y)
    if not layer.norm_first:
        root = layer.norm1(root)
    ff_input = layer.norm2(root) if layer.norm_first else root
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
    root = root + layer.dropout2(ff)
    return root if layer.norm_first else layer.norm2(root)


def validate_encoder(encoder):
    if len(encoder.encoder.layers) != 1:
        raise ValueError('root-query experiment requires exactly ONE encoder layer')
    if getattr(encoder, '_input_cache', None) is not None:
        raise ValueError('disable the topology cache for this isolated A/B experiment')
    validate_layer(encoder.encoder.layers[0])


def root_encoder_forward(encoder, texts):
    from .model import ModelLimit
    validate_encoder(encoder)
    if not texts:
        raise ValueError('cannot encode an empty input batch')
    trees = [cached_tree(text) for text in texts]
    size = max(len(tokens) for tokens, _, _ in trees)
    if size > encoder.max_tokens:
        raise ModelLimit(f'input needs {size} structural tokens; maximum is '
                         f'{encoder.max_tokens}')
    device = encoder.embedding.weight.device
    # Match the baseline's MPS padding exactly; do not truncate inputs.
    if device.type == 'mps':
        size = min(encoder.max_tokens, ((size + 15) // 16) * 16)
    # Assemble integer input buffers on CPU and transfer each once. No NxN
    # relation/bias matrix is allocated in this implementation.
    ids = torch.zeros((len(trees), size), dtype=torch.long)
    relations = torch.full_like(ids, OTHER)
    for i, (tokens, parents, bindings) in enumerate(trees):
        n = len(tokens)
        ids[i, :n] = torch.tensor(tokens, dtype=torch.long)
        relations[i, :n] = root_relation_ids(parents, bindings)
    ids, relations = ids.to(device), relations.to(device)
    x = encoder.embedding(ids) * math.sqrt(encoder.embedding.embedding_dim)
    positions = torch.arange(size, dtype=x.dtype, device=device).unsqueeze(1)
    scale = torch.exp(torch.arange(0, x.shape[-1], 2, dtype=x.dtype, device=device)
                      * (-math.log(10000.) / x.shape[-1]))
    pe = torch.zeros((size, x.shape[-1]), dtype=x.dtype, device=device)
    pe[:, 0::2] = torch.sin(positions * scale)
    pe[:, 1::2] = torch.cos(positions * scale)
    bias = encoder.relationship(relations).permute(0, 2, 1).unsqueeze(2)
    bias = bias.masked_fill(ids.eq(0)[:, None, None, :], -1e4)
    root = root_layer_forward(encoder.encoder.layers[0], x + pe, bias)
    if encoder.encoder.norm is not None:
        root = encoder.encoder.norm(root)
    return root[:, 0]


@contextmanager
def root_query_encoder(policy):
    """Temporarily replace only AST forward; preserve parameters/state_dict.

    Restoration also runs after errors. Autograd is NOT disabled; model train/
    eval state and global PyTorch settings are untouched. No optimizer steps.
    """
    if policy.representation != 'ast':
        raise ValueError('root-query experiment requires an AST checkpoint')
    encoder = policy.structural_encoder
    validate_encoder(encoder)
    had_override = 'forward' in encoder.__dict__
    original = encoder.__dict__.get('forward')
    encoder.forward = MethodType(root_encoder_forward, encoder)
    try:
        yield policy
    finally:
        if had_override:
            encoder.forward = original
        else:
            del encoder.forward
