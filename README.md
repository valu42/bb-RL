# bb-RL — proof-game research prototype

A small, randomly initialized, byte-level Transformer is trained with on-policy REINFORCE to build formally checked proofs about Turing-machine halting or non-halting.

The complete original source archive is available from the project owner. The repository is being populated from that archive; do not treat a partial checkout as runnable until the proof checker, environment, and training modules are added.

## Architecture
- Transformer encoder: 2 layers, width 64, 4 attention heads; approximately 125k parameters.
- UTF-8 byte tokenizer (no pretrained vocabulary).
- One state encoder and shared encoder for a variable menu of candidate actions.
- Learned scalar value baseline; reward of 1 only for accepted, submitted proofs.
- Training code uses PyTorch and on-policy REINFORCE.

## Intended commands (once all modules are present)

```bash
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
python -m proofgame demo one_step
python -m proofgame.train --episodes 1000 --states 2 3 --out runs/my_first_run
```

Results from the supplied archive are preliminary: a filtered test set is heavily skewed toward one- or two-step halting machines, so the high success rate should not be read as general non-halting proof discovery.
