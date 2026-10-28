"""Randomly initialized byte/AST Transformer scoring a variable action menu.

The optional AST mode exposes formal syntax and learnable relation biases but
never adds proof demonstrations or proof-outcome features.
"""
import json
import math
import torch
from torch import nn


class ModelLimit(RuntimeError):
    pass


class Policy(nn.Module):
    def __init__(self,width=64,layers=2,heads=4,max_tokens=4096,representation="bytes",
                 encoder_backend='full'):
        super().__init__()
        if width%heads or width%2: raise ValueError("width must be even and divisible by heads")
        if representation not in ('bytes', 'ast'):
            raise ValueError('representation must be bytes or ast')
        self.config=dict(width=width,layers=layers,heads=heads,max_tokens=max_tokens)
        self.representation=representation
        if encoder_backend not in ('full', 'root-query'):
            raise ValueError('encoder backend must be full or root-query')
        if encoder_backend == 'root-query' and (representation != 'ast' or layers != 1):
            raise ValueError('root-query requires a one-layer AST policy')
        if representation=='bytes':
            self.embedding=nn.Embedding(258,width,padding_idx=0)
            layer=nn.TransformerEncoderLayer(width,heads,4*width,dropout=0.,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
            # TransformerEncoder clones initial layer weights; initialize each independently.
            for block in self.encoder.layers:
                for p in block.parameters():
                    if p.dim()>1: nn.init.xavier_uniform_(p)
        else:
            from .structural_encoding import StructuralEncoder
            self.structural_encoder=StructuralEncoder(width,layers,heads,max_tokens)
            self.config['representation']='ast'
            self.structural_encoder.use_root_query = encoder_backend == 'root-query'
        if encoder_backend != 'full':
            self.config['encoder_backend'] = encoder_backend
        self.query=nn.Linear(width,width)
        self.key=nn.Linear(width,width)
        self.action_bias=nn.Linear(width,1)
        self.value=nn.Linear(width,1)

    def encode(self,texts):
        if self.representation=='ast':
            return self.structural_encoder(texts)
        encoded=[[257]+[b+1 for b in t.encode("utf-8")] for t in texts]
        n=max(map(len,encoded))
        if n>self.config["max_tokens"]: raise ModelLimit(f"input needs {n} tokens; maximum is {self.config['max_tokens']}")
        device=self.embedding.weight.device
        # Limit unique Metal graph shapes for variable-length inputs.
        if device.type=='mps':
            n=min(self.config['max_tokens'],((n+15)//16)*16)
        ids=torch.zeros((len(texts),n),dtype=torch.long,device=device)
        for i,row in enumerate(encoded): ids[i,:len(row)]=torch.tensor(row,device=device)
        x=self.embedding(ids)*math.sqrt(self.config["width"])
        pos=torch.arange(n,device=device,dtype=x.dtype).unsqueeze(1)
        scale=torch.exp(torch.arange(0,x.shape[-1],2,device=device,dtype=x.dtype)*(-math.log(10000.)/x.shape[-1]))
        pe=torch.zeros((n,x.shape[-1]),device=device,dtype=x.dtype)
        pe[:,0::2]=torch.sin(pos*scale); pe[:,1::2]=torch.cos(pos*scale)
        return self.encoder(x+pe,src_key_padding_mask=ids.eq(0))[:,0]

    def score_many(self,observations):
        """Differentiable, shared-embedding scores for a group of observations.

        Actions are encoded only once per unique JSON description, and states
        share one encoder invocation. Useful for MPS policy-gradient updates.
        Scores have the same per-observation action ordering as forward().
        """
        if not observations:
            return []
        contexts=[{k:v for k,v in obs.items() if k!='actions'} for obs in observations]
        context_texts=[json.dumps(obj,sort_keys=True,separators=(',',':')) for obj in contexts]
        action_groups=[[json.dumps(a,sort_keys=True,separators=(',',':')) for a in obs['actions']]
                       for obs in observations]
        if any(not group for group in action_groups):
            raise ValueError('terminal observation has no actions')
        unique=list(dict.fromkeys(action for group in action_groups for action in group))
        positions={value:i for i,value in enumerate(unique)}
        context_embeddings=self.encode(context_texts)
        action_embeddings=self.encode(unique)
        keys=self.key(action_embeddings)
        biases=self.action_bias(action_embeddings).squeeze(-1)
        queries=self.query(context_embeddings)
        scores=[]
        for query,group in zip(queries,action_groups):
            indices=torch.tensor([positions[a] for a in group],dtype=torch.long,device=keys.device)
            logits=(keys.index_select(0,indices)*query).sum(-1)/math.sqrt(query.shape[-1])
            scores.append(logits+biases.index_select(0,indices))
        return scores

    def forward(self,observation):
        actions=observation["actions"]
        if not actions: raise ValueError("terminal observation has no actions")
        # The candidates are encoded separately and scored against the complete context.
        context={k:v for k,v in observation.items() if k!="actions"}
        h=self.encode([json.dumps(context,sort_keys=True,separators=(",",":"))])[0]
        a=self.encode([json.dumps(x,sort_keys=True,separators=(",",":")) for x in actions])
        logits=(self.key(a)*self.query(h)).sum(-1)/math.sqrt(h.shape[-1])+self.action_bias(a).squeeze(-1)
        return logits,self.value(h).squeeze(-1)
