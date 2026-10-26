import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch dependency is not installed')
class ASTEncodingTests(unittest.TestCase):
    def test_formal_tree_and_bound_variable_links(self):
        from proofgame.structural_encoding import (
            encode_tree, relation_ids, BINDS, BOUND_BY, VOCAB_SIZE
        )
        value={'goals':[{'goal':'(all Nat (imp (eq bvar0 z) false))',
                         'facts':{'h0':'(ex Nat (eq bvar0 bvar0))'}}],
               'machine':{'rows':[['q1','b1','L'],['qH','b0','R']]}}
        tokens, parents, bindings = encode_tree(value)
        self.assertTrue(bindings, 'the quantifier binder link must be represented')
        self.assertEqual(len(tokens),len(parents))
        self.assertTrue(all(0 <= token < VOCAB_SIZE for token in tokens))
        relations = relation_ids(parents, bindings)
        for binder, occurrence in bindings:
            self.assertEqual(relations[binder, occurrence].item(),BINDS)
            self.assertEqual(relations[occurrence, binder].item(),BOUND_BY)
        self.assertNotEqual(encode_tree({'id':'rule:exI'})[0],
                            encode_tree({'id':'rule:impI'})[0])
        self.assertNotEqual(encode_tree({'id':'unknown_identifier_13'})[0],
                            encode_tree({'id':'unknown_identifier_14'})[0])

    def test_policy_gradients_checkpoint_and_frozen_sampler(self):
        import torch
        from proofgame.model import Policy
        from proofgame.network_search import FrozenSampler
        torch.set_num_threads(1)
        torch.manual_seed(11)
        policy = Policy(width=32,layers=1,representation='ast').eval()
        self.assertEqual(policy.config['representation'],'ast')
        obs = {'machine':{'rows':[['q0','b1','R'],['qH','b0','L']]},
               'answer':None, 'goals':[{'goal':'(ex Nat (eq (qof (at bvar0 (cfg q0 nil nil))) qH))'}],
               'actions_left':100,
               'actions':[{'id':'choose:HALT'},{'id':'choose:NONHALT'},{'id':'reset'}]}
        frozen = FrozenSampler(policy)
        sampled = frozen.probabilities(obs)
        logits, value = policy(obs)
        torch.testing.assert_close(torch.tensor(sampled),logits.softmax(-1).detach(),rtol=5e-4,atol=5e-6)
        self.assertEqual(len(sampled),3)
        self.assertEqual(frozen.probabilities(obs),sampled)
        loss = -logits.log_softmax(-1)[0] + value.square() * 0.01
        loss.backward()
        grad = policy.structural_encoder.relationship.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0)
        clone=Policy(**policy.config).eval()
        clone.load_state_dict(policy.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(clone(obs)[0],policy(obs)[0])

    def test_verified_proof_can_update_ast_policy(self):
        import torch
        from proofgame.epoch_train import learn_batch
        from proofgame.environment import Game
        from proofgame.model import Policy
        from proofgame.network_search import FrozenSampler
        from proofgame.population_sample import from_rank
        torch.set_num_threads(1)
        torch.manual_seed(7)
        policy=Policy(width=32,layers=1,representation='ast').eval()
        optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0.)
        sampler=FrozenSampler(policy)
        machine=from_rank(8)
        game=Game(machine,budget=20)
        trajectory=[]
        for action in ['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']:
            observation=game.observation()
            probs=sampler.probabilities(observation)
            index=next(i for i,a in enumerate(observation['actions']) if a['id']==action)
            trajectory.append({'action':action,'probability':probs[index]})
            game.step(action,observe=False)
        self.assertEqual(game.status,'SOLVED')
        success={'actions':trajectory,'certificate':game.certificate()}
        result={'machine':machine.to_json(),'attempts':1,'successes':[success]}
        update=learn_batch(policy,optimizer,[result],20)
        self.assertTrue(update['updated'])
        self.assertEqual(update['proofs'],1)
        self.assertTrue(torch.isfinite(torch.tensor(update['loss'])))

    def test_limit_and_legacy_model_config(self):
        from proofgame.model import ModelLimit, Policy
        legacy=Policy(width=32,layers=1)
        self.assertNotIn('representation',legacy.config)
        self.assertEqual(Policy(**legacy.config).representation,'bytes')
        policy=Policy(width=32,layers=1,max_tokens=2,representation='ast')
        with self.assertRaises(ModelLimit):
            policy({'actions':[{'id':'reset'}], 'machine':{'rows':[]}})


if __name__=='__main__':
    unittest.main()
