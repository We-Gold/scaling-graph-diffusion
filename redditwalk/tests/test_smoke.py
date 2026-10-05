"""Smoke tests on synthetic data. CPU, a few seconds. No network."""

import importlib.util
import json
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from make_synthetic import write_giant_component_csv, write_snap_tsvs
from redditwalk.dataset import RedditDataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_steps_01_02_header(tmp_path):
    write_snap_tsvs(tmp_path)
    res = tmp_path / "results"
    stats = load_script("01_giant_component.py").main(["--data-dir", str(tmp_path), "--results-dir", str(res)])

    nodes = (tmp_path / "giant_component_nodes.txt").read_text().split()
    assert nodes == ["a", "b", "c", "d"]
    assert "SOURCE_SUBREDDIT" not in nodes
    assert stats == json.loads((tmp_path / "wcc_stats.json").read_text())
    assert stats == json.loads((res / "sec6_1_wcc_stats.json").read_text())
    assert stats["tsv_rows"] == 5 and stats["directed_edges_dedup"] == 4
    assert stats["nodes"] == 6 and stats["num_wccs"] == 2
    assert stats["gwcc_nodes"] == 4 and stats["second_wcc_nodes"] == 2

    n_body, n_title, n_comb = load_script("02_filter_edges.py").main(["--data-dir", str(tmp_path)])
    assert (n_body, n_title, n_comb) == (2, 2, 4)  # combined keeps the duplicate (a, b)
    comb = pd.read_csv(tmp_path / "giant_component_combined.csv")
    assert list(comb.columns)[:2] == ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]
    assert len(comb) == 4


def test_step_08_sampler(tmp_path):
    csv = write_giant_component_csv(tmp_path / "giant_component.csv")
    out = tmp_path / "subgraphs"
    m_max = load_script("08_sample_subgraphs.py").main(
        ["--csv", str(csv), "--out-dir", str(out), "--num-subgraphs", "5", "--size", "50", "--base-seed", "3"])
    assert m_max > 0

    stem = out / "5_reddit_subgraphs_50"
    records = torch.load(f"{stem}.pt", weights_only=False)
    node_lists = json.loads(Path(f"{stem}_nodes.json").read_text())
    jac = json.loads(Path(f"{stem}_jaccard.json").read_text())
    assert len(records) == len(node_lists) == 5
    for (A, C, E), nodes in zip(records, node_lists):
        assert len(A) == len(nodes) == 50 and E.shape == (50, 50)
        assert 2 <= min(A) and max(A) <= 21
        assert set(np.unique(E)) <= {1, 2, 3}
    sets = [set(n) for n in node_lists]
    scores = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    assert jac["mean"] == pytest.approx(np.mean(scores))
    assert jac["max"] == pytest.approx(np.max(scores))
    assert jac["n_pairs"] == 10 and jac["base_seed"] == 3

    train = RedditDataset(root=str(out), dataset_file="5_reddit_subgraphs_50.pt")
    test = RedditDataset(stage="test", root=str(out), dataset_file="5_reddit_subgraphs_50.pt")
    assert (len(train), len(test)) == (4, 1)


def test_step_09_stats(tmp_path):
    csv = write_giant_component_csv(tmp_path / "giant_component.csv")
    sub = tmp_path / "subgraphs"
    step08 = load_script("08_sample_subgraphs.py")
    sets = []
    for n, size in [(6, 30), (5, 60), (4, 100)]:
        step08.main(["--csv", str(csv), "--out-dir", str(sub), "--num-subgraphs", str(n),
                     "--size", str(size), "--base-seed", "0"])
        sets.append(f"{n}_reddit_subgraphs_{size}.pt")

    res = tmp_path / "results"
    stats, table = load_script("09_dataset_stats.py").main(
        ["--subgraph-dir", str(sub), "--results-dir", str(res), "--sets", *sets,
         "--labels", "30-node", "60-node", "100-node"])
    for name in ["deg_dist", "clust_dist", "edge_density", "triangles"]:
        assert (res / f"fig12_{name}.png").stat().st_size > 0
    saved = json.loads((res / "fig12_stats.json").read_text())
    assert saved["30-node"]["num_graphs"] == 6
    t16 = pd.read_csv(res / "table16_jaccard.csv")
    assert list(t16["graph_size"]) == [30, 60, 100]
    assert list(t16["num_graphs"]) == [6, 5, 4]


class _Resp:
    def __init__(self, code, payload=None):
        self.status_code = code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.mark.parametrize("script", ["03_fetch_descriptions.py", "04_retry_rate_limited.py"])
def test_steps_03_04_status_no_network(monkeypatch, script):
    mod = load_script(script)
    replies = {
        "ok": _Resp(200, {"data": {"public_description": " hi\tthere "}}),
        "empty": _Resp(200, {"data": {"public_description": "", "description": ""}}),
        "gone": _Resp(404), "private": _Resp(403), "slow": _Resp(429), "bad": _Resp(500),
    }
    monkeypatch.setattr(mod.requests, "get", lambda url, **kw: replies[url.split("/")[4]])
    got = {k: mod.fetch_subreddit_description(k)[0] for k in replies}
    assert got == {"ok": "success_with_desc", "empty": "success_no_desc", "gone": "not_found",
                   "private": "forbidden", "slow": "rate_limited", "bad": "error_500"}


def test_step_03_04_files(tmp_path, monkeypatch):
    (tmp_path / "giant_component_nodes.txt").write_text("ok\nslow\n")
    step03 = load_script("03_fetch_descriptions.py")
    monkeypatch.setattr(step03, "DELAY_BETWEEN_REQUESTS", 0)
    monkeypatch.setattr(step03.time, "sleep", lambda s: None)
    status = {"ok": _Resp(200, {"data": {"public_description": "a\nb"}}), "slow": _Resp(429)}
    monkeypatch.setattr(step03.requests, "get", lambda url, **kw: status[url.split("/")[4]])
    step03.main(["--data-dir", str(tmp_path)])
    lines = (tmp_path / "subreddit_descriptions.tsv").read_text().splitlines()
    assert lines == ["INDEX\tSUBREDDIT\tSTATUS\tDESCRIPTION", "1\tok\tsuccess_with_desc\ta b",
                     "2\tslow\trate_limited\t"]

    step04 = load_script("04_retry_rate_limited.py")
    monkeypatch.setattr(step04.time, "sleep", lambda s: None)
    monkeypatch.setattr(step04.requests, "get", lambda url, **kw: _Resp(404))
    step04.main(["--data-dir", str(tmp_path)])
    lines = (tmp_path / "subreddit_descriptions_retry.tsv").read_text().splitlines()
    assert lines == ["INDEX\tSUBREDDIT\tSTATUS\tDESCRIPTION", "2\tslow\tnot_found\t"]


def test_walk_termination_check():
    G = nx.Graph()
    G.add_edges_from([("x", "y"), ("y", "z")])
    nx.set_node_attributes(G, 0, "topic")
    ds = RedditDataset.__new__(RedditDataset)
    with pytest.raises(ValueError, match="component of 3 nodes"):
        ds.sample_subgraph(G, max_size=10, seed=0)
    assert ds.sample_subgraph(G, max_size=3, seed=0).number_of_nodes() == 3
