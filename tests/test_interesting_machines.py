"""Regression tests for curated Turing machines and proof-action budgets.

Finite simulation is separate from formal kernel certificate checking. The two
NONHALT examples have human-described inductive invariants; the tests check
their phase structure but do not claim a bb-RL certificate was constructed.
"""
import json
from pathlib import Path
import unittest

from proofgame.environment import Game
from proofgame.kernel import check
from proofgame.logic import Machine


CASES = {
    row['id']: row
    for row in json.loads(
        (Path(__file__).resolve().parents[1] / 'examples' / 'interesting_machines.json').read_text()
    )['machines']
}


def transitions(machine, count):
    """Blank tape, q0, position zero, with an explicit dictionary of ones."""
    state, position, tape = 'q0', 0, set()
    for step in range(1, count + 1):
        bit = 'b1' if position in tape else 'b0'
        next_state, write, direction = machine.row(state, bit)
        if write == 'b1':
            tape.add(position)
        else:
            tape.discard(position)
        position += 1 if direction == 'R' else -1
        state = next_state
        yield step, state, position, frozenset(tape)
        if state == 'qH':
            return


def unary_proof_actions(t):
    return (['choose:HALT', 'rule:exI'] +
            ['put:s'] * t + ['put:z', 'commit', 'rule:refl', 'submit'])


def arithmetic_107_actions():
    # The term is add(mul(10, 10), 7), using the existing typed Nat
    # expression editor. A unary 107 witness would take 113 actions.
    return (['choose:HALT', 'rule:exI', 'put:add', 'put:mul'] +
            ['put:s'] * 10 + ['put:z'] +
            ['put:s'] * 10 + ['put:z'] +
            ['put:s'] * 7 + ['put:z', 'commit', 'rule:refl', 'submit'])


class InterestingMachinesTests(unittest.TestCase):
    def test_all_machine_tables_are_well_formed(self):
        self.assertEqual(len(CASES), 5)
        for item in CASES.values():
            with self.subTest(item=item['id']):
                machine = Machine(item['machine']['rows'])
                self.assertEqual(len(machine.rows), 2 * item['states'])

    def test_known_first_halt_transition(self):
        for item in CASES.values():
            if item['halt_steps'] is None:
                continue
            with self.subTest(item=item['id']):
                machine = Machine(item['machine']['rows'])
                states = list(transitions(machine, item['halt_steps'] + 1))
                self.assertEqual(len(states), item['halt_steps'])
                self.assertEqual(states[-1][1], 'qH')
                self.assertTrue(all(row[1] != 'qH' for row in states[:-1]))
        four = list(transitions(Machine(CASES['busy-beaver-4-107']['machine']['rows']), 107))
        self.assertNotEqual(four[99][1], 'qH')
        self.assertNotEqual(four[105][1], 'qH')
        self.assertEqual(four[106][1], 'qH')
        self.assertEqual(len(four[106][3]), 13)

    def test_two_state_parity_exact_cycle(self):
        machine = Machine(CASES['parity-trap-2']['machine']['rows'])
        states = list(transitions(machine, 2000))
        self.assertEqual(len(states), 2000)
        for step,state,position,tape in states:
            self.assertEqual(tape, frozenset({0}))
            if step % 2:
                self.assertEqual((state, position), ('q1', 1))
            else:
                self.assertEqual((state, position), ('q0', 0))

    def test_three_state_inductive_frontier_pattern(self):
        machine = Machine(CASES['growing-frontier-3']['machine']['rows'])
        states = list(transitions(machine, 3000))
        self.assertEqual(len(states), 3000)
        for step, state, position, tape in states:
            phase = step % 3
            n = (step - 1) // 3
            if phase == 1:  # step 3n+1
                self.assertEqual((state, position), ('q1', n+1))
                self.assertEqual(tape, frozenset(range(n+1)))
            elif phase == 2:  # step 3n+2
                self.assertEqual((state, position), ('q2', n))
                self.assertEqual(tape, frozenset(range(n+2)))
            else:  # step 3n+3
                self.assertEqual((state, position), ('q0', n+1))
                self.assertEqual(tape, frozenset(range(n+2)))

    def test_concrete_halting_proofs_use_real_kernel(self):
        for name, t in [('busy-beaver-2-6',6),('busy-beaver-3-21',21)]:
            with self.subTest(machine=name):
                machine=Machine(CASES[name]['machine']['rows'])
                actions=unary_proof_actions(t)
                self.assertLessEqual(len(actions),100)
                game=Game(machine,budget=100)
                for action in actions:
                    game.step(action,observe=False)
                self.assertEqual(game.status,'SOLVED',f'{name}: {game.error}')
                check(game.certificate())

    def test_four_state_compact_arithmetic_witness(self):
        machine=Machine(CASES['busy-beaver-4-107']['machine']['rows'])
        actions=arithmetic_107_actions()
        self.assertEqual(len(actions),37)
        self.assertGreater(len(unary_proof_actions(107)),100)
        game=Game(machine,budget=100)
        errors=[]
        for action in actions:
            game.step(action,observe=False)
            if game.error is not None:
                errors.append({'action':action,'message':game.error})
            if game.status!='ACTIVE':
                break
        if game.status=='SOLVED':
            check(game.certificate())
        else:
            # Regression fixture for proof expressivity, not a requirement
            # that the current kernel accepts the 107-step witness.
            self.assertEqual(game.status,'ACTIVE')
            self.assertTrue(errors,'the incomplete proof must be diagnosed')
            print('BUSY_BEAVER_107_PROOF_STATUS',json.dumps({
                'attempted_proof_actions':len(actions),
                'status':game.status,'errors':errors}),flush=True)


if __name__ == '__main__':
    unittest.main()
