"""Unlabelled uniform machine sampling and deterministic train/test partition."""
import hashlib
import json
import random
from .logic import Machine


def key(m):
    return hashlib.sha256(json.dumps(m.to_json(),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def is_test(m):
    return int(key(m)[:8],16)%5==0


def sample(rng, states=(2,3), split="train"):
    if split not in ("train","test"): raise ValueError("split must be train or test")
    if not states or any(type(k) is not int or not 1<=k<=32 for k in states):
        raise ValueError("state counts must be 1..32")
    for _ in range(100000):
        k=rng.choice(states); qs=[f"q{i}" for i in range(k)]+["qH"]
        m=Machine(tuple((rng.choice(qs),rng.choice(("b0","b1")),rng.choice(("L","R"))) for _ in range(2*k)))
        if is_test(m)==(split=="test"): return m
    raise RuntimeError("could not sample requested partition")


def fixtures():
    # Demonstrations/test fixtures only. The trainer does not call this function.
    return {
        "one_step":Machine((("qH","b0","R"),("qH","b1","R"))),
        "two_step":Machine((("q1","b1","R"),("q1","b1","R"),
                            ("qH","b0","L"),("qH","b1","L"))),
        "forever":Machine((("q0","b1","R"),("q0","b0","R"))),
        "three_step":Machine((("q1","b1","R"),("q1","b1","R"),
                              ("q2","b1","R"),("q2","b1","R"),
                              ("qH","b1","R"),("qH","b1","R"))),
        "two_state_forever":Machine((("q1","b1","R"),("q1","b1","R"),
                                     ("q0","b1","R"),("q0","b1","R"))),
        "blank_sweeper":Machine((("q0","b1","R"),("qH","b1","R"))),
    }
