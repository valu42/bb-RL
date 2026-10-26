"""Regression tests for auto-numbered run directories and AST preprocessing."""
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from proofgame.epoch_train import choose_run_directory
from proofgame.structural_encoding import (
    encode_tree, cached_tree, relation_ids,
    OTHER, SELF, PARENT, CHILD, SIBLING, ANCESTOR,
    DESCENDANT, BINDS, BOUND_BY
)
import torch


def reference_relations(parents, bindings):
    """Original scalar relation algorithm for exact equivalence."""
    n = len(parents)
    rel = torch.full((n,n), OTHER, dtype=torch.long)
    rel.fill_diagonal_(SELF)
    children=defaultdict(list)
    for i,p in enumerate(parents):
        if p>=0:
            rel[p,i]=PARENT
            rel[i,p]=CHILD
            children[p].append(i)
            a=parents[p]
            while a>=0:
                rel[a,i]=ANCESTOR
                rel[i,a]=DESCENDANT
                a=parents[a]
    for siblings in children.values():
        for i in siblings:
            for j in siblings:
                if i!=j:
                    rel[i,j]=SIBLING
    for binder,occurrence in bindings:
        rel[binder,occurrence]=BINDS
        rel[occurrence,binder]=BOUND_BY
    return rel


class RunDirectoryTests(unittest.TestCase):
    def test_numbered_runs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'experiment'
            a=choose_run_directory(root)
            (a/'state.pt').write_bytes(b'keep-me')
            b=choose_run_directory(root)
            c=choose_run_directory(root)
            self.assertEqual(a,root)
            self.assertEqual(b.name,'experiment_run_0001')
            self.assertEqual(c.name,'experiment_run_0002')
            self.assertEqual((a/'state.pt').read_bytes(),b'keep-me')
            self.assertEqual(choose_run_directory(a,resume=True),a)
            with self.assertRaises(FileNotFoundError):
                choose_run_directory(Path(tmp)/'missing',resume=True)


class StructuralPreprocessingTests(unittest.TestCase):
    def test_cached_trees_and_relation_equivalence(self):
        examples=[
            {'goals':[{'goal':'(all Nat (imp (eq bvar0 z) false))',
                       'facts':{'h0':'(ex Nat (eq bvar0 bvar0))'}}],
             'machine':{'rows':[['q1','b1','L'],['qH','b0','R']]}},
            {'actions':[{'id':'rule:exI'},{'id':'put:s'},{'id':'submit'}]},
            {'machine':{'rows':[['q0','b0','R']]*4},'answer':None,'actions_left':100},
            {'deep':{'children':[{'x':[1,2,3]},{'y':{'a':'(all Nat (eq bvar0 z))'}}]}}
        ]
        for item in examples:
            with self.subTest(item=item):
                text=json.dumps(item,sort_keys=True,separators=(',',':'))
                tokens,parents,bindings=encode_tree(item)
                cached=cached_tree(text)
                self.assertEqual(cached,(tuple(tokens),tuple(parents),tuple(bindings)))
                torch.testing.assert_close(relation_ids(parents,bindings),
                                           reference_relations(parents,bindings))
                torch.testing.assert_close(relation_ids(cached[1],cached[2]),
                                           reference_relations(parents,bindings))


if __name__=='__main__':
    unittest.main()
