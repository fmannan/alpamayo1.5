"""Output pairing between runs."""

import numpy as np
import pytest

from orinbench.behavior import compare, seed_spread


def rec(scene, seed, offset=0.0, tokens=(1, 2, 3), finite=True):
    xyz = np.stack([np.linspace(0, 64, 64) + offset, np.zeros(64), np.zeros(64)], 1)
    return {"scene_id": scene, "seed": seed, "pred_xyz": xyz.tolist(), "gen_token_ids": list(tokens),
            "finite": finite, "reasoning_complete": True}


def test_identical_runs():
    ref = [rec("a", 0), rec("a", 1), rec("b", 0)]
    agg = compare(ref, ref)["aggregate"]
    assert agg["n_paired_finite"] == 3 and agg["identical_reasoning_rate"] == 1.0
    assert agg["vs_ref_ade_6.4s_max"] == 0.0


def test_differences_and_unpaired_records_are_reported():
    ref = [rec("a", 0), rec("b", 0), rec("c", 0)]
    test = [rec("a", 0, offset=0.5, tokens=(1, 2, 4)), rec("b", 0, finite=False), rec("d", 0)]
    agg = compare(ref, test)["aggregate"]
    assert agg["n_paired_finite"] == 1
    assert agg["n_test_only"] == 1 and agg["n_reference_only"] == 1
    assert agg["identical_reasoning_rate"] == 0.0
    assert agg["vs_ref_ade_6.4s_mean"] == pytest.approx(0.5)


def test_duplicates_are_rejected():
    with pytest.raises(ValueError):
        compare([rec("a", 0), rec("a", 0)], [rec("a", 0)])


def test_seed_spread():
    s = seed_spread([rec("a", 0), rec("a", 1, offset=1.0), rec("b", 0)])
    assert s["pairs"] == 1 and s["seed_pair_ade_6.4s_mean"] == pytest.approx(1.0)
