"""--sampler original/novel calls a different new-edge sampler. Runs only in the sparserdiff env:

    uv run --project ../../sparserdiff python -m pytest tests/test_sampler_flag.py
"""

import pytest

sd = pytest.importorskip("sparse_diffusion.diffusion_model_sparse")

import torch  # noqa: E402

import profile_sparsediff  # noqa: E402


@pytest.mark.parametrize("novel", [False, True])
def test_flag_selects_sampler(monkeypatch, novel):
    calls = []
    for name in ("sample_non_existing_edges_novel", "sample_non_existing_edges_batched"):
        orig = getattr(sd, name)

        def spy(*a, _orig=orig, _name=name, **k):
            calls.append(_name)
            return _orig(*a, **k)

        monkeypatch.setattr(sd, name, spy)

    metrics = profile_sparsediff.run_profile_on_seed(
        10, 20, 2, "test", 2, 9, 4, torch.device("cpu"), edge_fraction=0.1, use_novel_sampling=novel
    )
    assert calls, "no new edges were sampled"
    expected = "sample_non_existing_edges_novel" if novel else "sample_non_existing_edges_batched"
    assert set(calls) == {expected}
    assert all(m["time"] == m["time"] for m in metrics.values())  # no NaN
