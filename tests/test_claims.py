"""Jede Zahl in den Grenzen der App und im README, gegen die echten Auswertungsfunktionen (feste Seeds, plattformstabil dank random.Random; Ränder bei Zufallsgrößen)."""

import math
from dataclasses import replace
from functools import lru_cache

import pytest

import rst_algorithm as A
import rst_constants as C
import rst_evaluation as ev

BASE = ev.Settings()


@lru_cache(maxsize=None)
def _rc(**changes):
    return ev.run_config(BASE, **changes)


def _text():
    return open("app.py", encoding="utf-8").read()


def test_count_and_cost_by_size():
    assert [round(_rc(n=n)["log10_tau"], 1) for n in C.N_SWEEP] == [6.2, 9.6, 15.7, 23.9, 31.2]
    assert [round(_rc(n=n)["cost_ratio"], 2) for n in C.N_SWEEP] == [1.98, 2.05, 1.88, 1.82, 1.79]
    assert [round(_rc(k=k)["log10_tau"], 1) for k in (3, 4, 6, 10, 1000)] == [8.0, 11.0, 15.7, 20.4, 25.1]
    assert [round(_rc(k=k)["cost_ratio"], 2) for k in (3, 6, 1000)] == [1.30, 1.88, 3.37]


def test_uniform_trees_ignore_costs():
    r = _rc()
    assert r["mst_incl"] == pytest.approx(0.249, abs=0.0005) and r["other_incl"] == pytest.approx(0.259, abs=0.0005) and r["mst_incl"] < r["other_incl"]
    assert r["cost_ratio"] == pytest.approx(1.88, abs=0.005)
    _t = _text()
    for v in ("24.9 %", "25.9 %", "1.88-mal", "3.37-mal", "1.10 bei b = 4", "1.03 bei b = 8"):
        assert v in _t, v


def test_temperature_lowers_the_cost_ratio_and_raises_p_mst():
    assert [round(_rc(b=b)["cost_ratio"], 2) for b in (2.0, 4.0, 8.0)] == [1.28, 1.10, 1.03]
    assert _rc(b=4.0)["p_mst"] == pytest.approx(0.0003, abs=0.00005) and _rc(b=8.0)["p_mst"] == pytest.approx(0.027, abs=0.0005) and _rc(b=16.0)["p_mst"] == pytest.approx(0.248, abs=0.0005)
    _t = _text()
    for v in ("0.03 %", "2.7 %", "24.8 %"):
        assert v in _t, v


def test_wilson_and_aldous_steps_explode_when_cold():
    r0, r2, r4 = _rc(), _rc(b=2.0), _rc(b=4.0)
    assert (round(r0["wilson_steps"]), round(r0["aldous_steps"])) == (47, 100) and r0["steps_ratio"] == pytest.approx(0.47, abs=0.005)
    assert (round(r4["wilson_steps"]), round(r4["aldous_steps"])) == (882, 5734) and r2["wilson_steps"] < r4["wilson_steps"]
    done = {}
    for b in (8.0, 16.0):
        cmp = [ev.sampler_compare(ev.analyse(replace(BASE, b=b, seed=s)), 100) for s in C.SWEEP_SEEDS]
        done[b] = [c["wilson"]["n_samples"] == 100 for c in cmp]
        assert all(c["kruskal"]["n_samples"] == 100 for c in cmp)
    assert sum(done[8.0]) == 2 and sum(done[16.0]) == 0
    assert min(c for c in (ev.sampler_compare(ev.analyse(replace(BASE, b=8.0, seed=100000)), 100)["wilson"]["steps"],)) == pytest.approx(9058, abs=1)
    assert math.isnan(_rc(b=16.0)["wilson_steps"])
    _t = _text()
    for v in ("47 Zufallsschritte", "882 gegen 5 734", "9 058", "2 der 5", "in einer Million Schritten"):
        assert v in _t, v


def test_random_kruskal_is_not_uniform():
    u = ev.uniformity("textbook")
    assert u["kruskal"]["chi2"] == pytest.approx(189.9, abs=0.05) and u["crit"] == pytest.approx(45.4, abs=0.05) and u["wilson"]["chi2"] == pytest.approx(15.1, abs=0.05) and u["aldous"]["chi2"] == pytest.approx(17.5, abs=0.05)
    assert u["kruskal_exact_tv"] == pytest.approx(0.041, abs=0.0005)
    r = _rc()
    assert r["kruskal_dev"] == pytest.approx(0.045, abs=0.0015) and r["wilson_dev"] == pytest.approx(0.024, abs=0.0015) and r["kruskal_dev"] > 1.5 * r["wilson_dev"]
    _t = _text()
    for v in ("189.9", "45.4", "4.1 %", "(15.1)", "(17.5)", "0.045", "0.024"):
        assert v in _t, v


def test_weighted_random_kruskal_is_temperature_aware_but_not_gibbs():
    a = ev.analyse(ev.Settings(kind="textbook", b=2.0))
    w, k = ev.sample_stats(a, "wilson", 20000), ev.sample_stats(a, "kruskal", 20000)
    assert a.exp_cost == pytest.approx(16.81, abs=0.005) and abs(w["mean_cost"] - a.exp_cost) < 0.06 and k["mean_cost"] == pytest.approx(16.41, abs=0.03) and k["mean_cost"] < a.exp_cost - 0.3
    flat = ev.sample_stats(ev.analyse(ev.Settings(kind="textbook", b=0.0)), "kruskal", 20000)
    assert flat["mean_cost"] > k["mean_cost"] + 1.0
    _t = _text()
    for v in ("16.41 gegen den exakten Erwartungswert 16.81",):
        assert v in _t, v


def test_bridges_and_reliability_by_density():
    assert [(round(100 * ev.bridge_rate(ev.Settings(k=k))["share"]), ev.bridge_rate(ev.Settings(k=k))["max"]) for k in (3, 4, 6, 20)] == [(40, 2), (8, 1), (0, 0), (0, 0)]
    assert ev.bridge_rate(ev.Settings(kind="depot", n=20, k=1000))["share"] == 0.0
    med = {k: ev.reliability_curve(ev.Settings(k=k)) for k in (3, 4, 6)}
    got = {k: [round(100 * med[k][p]["median"], 1) for p in (0.2, 0.3)] for k in med}
    assert got == {3: [77.6, 49.6], 4: [97.6, 87.9], 6: [99.9, 99.3]}
    assert med[3][0.5]["median"] == pytest.approx(0.036, abs=0.001) and med[6][0.5]["median"] == pytest.approx(0.815, abs=0.0005)
    assert med[3][0.2]["tree"] == pytest.approx(0.0115, abs=0.00005) and med[3][0.3]["tree"] == pytest.approx(0.0008, abs=0.00005)
    _t = _text()
    for v in ("1.15 %", "0.08 %", "99.9 %", "99.3 %", "97.6 %", "87.9 %", "77.6 %", "49.6 %", "40 % der Instanzen", "8 %"):
        assert v in _t, v


def test_rank_correlation_between_tree_count_and_reliability():
    got = {(k, p): ev.correlation(ev.Settings(k=k), p)["spearman"] for k, p in ((4, 0.3), (3, 0.2), (6, 0.5))}
    assert [round(v, 2) for v in got.values()] == [0.71, 0.72, 0.84] and all(v < 1.0 for v in got.values())
    _t = _text()
    for v in ("0.71 (k = 4, p = 0.3)", "0.72 (k = 3, p = 0.2)", "0.84 (k = 6, p = 0.5)"):
        assert v in _t, v


def test_markov_bound_is_useless_here():
    r = _rc()
    assert r["expected_trees"] == pytest.approx(6.6e14, rel=0.01)
    k3 = ev.run_config(replace(BASE, k=3), p=0.3)
    assert k3["expected_trees"] == pytest.approx(8.7e4, rel=0.01) and k3["survive"] == pytest.approx(0.442, abs=0.0005) and k3["expected_trees"] > 1e4
    _t = _text()
    for v in ("6.6 x 10^14", "8.7 x 10^4", "44 %"):
        assert v in _t, v


def test_counting_is_exact_and_fast_at_the_upper_limit():
    a = ev.analyse(ev.Settings(n=C.N_MAX, k=1000))
    assert a.tau == (C.N_MAX + 1) ** (C.N_MAX - 1)
    assert A.count_trees(a.n, a.edges) == a.tau


def test_step_counts_by_size_and_temperature_in_the_readme():
    got = [(round(_rc(n=n)["wilson_steps"], 1), round(_rc(n=n)["aldous_steps"]), round(_rc(n=n)["steps_ratio"], 2)) for n in C.N_SWEEP]
    assert got == [(13.8, 22, 0.59), (22.8, 45, 0.49), (46.9, 100, 0.47), (93.1, 190, 0.5), (138.3, 286, 0.49)]
    assert (round(_rc(b=2.0)["wilson_steps"]), round(_rc(b=2.0)["aldous_steps"])) == (87, 439)
    a = ev.analyse(ev.Settings(b=8.0, seed=100000))
    for kind in ("aldous",):
        assert all(ev.sampler_compare(ev.analyse(ev.Settings(b=8.0, seed=s)), 100)[kind]["n_samples"] < 100 for s in C.SWEEP_SEEDS)
    assert _rc(b=16.0)["cost_ratio"] == pytest.approx(1.004, abs=0.0005) and a.tau > 0


def test_k4_uniformity_and_wilson_mean_cost_in_the_readme():
    u = ev.uniformity("k4")
    assert (round(u["wilson"]["chi2"], 1), round(u["aldous"]["chi2"], 1), round(u["kruskal"]["chi2"], 1), round(u["crit"], 1)) == (18.1, 13.5, 60.8, 37.8)
    a = ev.analyse(ev.Settings(kind="textbook", b=2.0))
    assert ev.sample_stats(a, "wilson", 20000)["mean_cost"] == pytest.approx(16.80, abs=0.02)
