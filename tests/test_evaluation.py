"""Auswertung: Analyse, Stichproben mit Schrittbudget, Sampler-Vergleich, Gleichverteilungstest, Zuverlässigkeit, Korrelation, run_config, Sweeps."""

import math
from dataclasses import replace

import pytest

import rst_algorithm as A
import rst_constants as C
import rst_evaluation as ev

FAST = ev.Settings(n=10, k=5, seed=3)


def test_analyse_is_consistent():
    a = ev.analyse(FAST)
    assert a.tau == A.count_trees(a.n, a.edges) and len(a.mst) == a.n - 1 and A.is_spanning_tree(a.n, a.edges, a.mst)
    assert sum(a.incl) == pytest.approx(a.n - 1, abs=1e-9) and sum(a.incl_uniform) == pytest.approx(a.n - 1, abs=1e-9) and a.incl == a.incl_uniform            # b = 0
    assert a.cost_ratio == pytest.approx(a.exp_cost / a.mst_cost) and a.cost_ratio > 1.0 and a.p_mst == pytest.approx(1.0 / a.tau)
    assert a.cbar == pytest.approx(a.mst_cost / (a.n - 1)) and a.beta == 0.0 and all(w == 1.0 for w in a.weights)
    assert a.log10_tau == pytest.approx(math.log10(a.tau)) and a.bridges == []


def test_temperature_changes_the_weights_and_the_expectation():
    cold = ev.analyse(replace(FAST, b=4.0))
    assert cold.beta == pytest.approx(4.0 / cold.cbar) and cold.incl != cold.incl_uniform and sum(cold.incl) == pytest.approx(cold.n - 1, abs=1e-9)
    assert cold.cost_ratio < ev.analyse(FAST).cost_ratio and cold.p_mst > ev.analyse(FAST).p_mst
    assert cold.mst_incl_mean() > cold.other_incl_mean()
    hot = ev.analyse(FAST)
    assert hot.mst_incl_mean() == pytest.approx(hot.other_incl_mean(), abs=0.1)


def test_bridges_are_found_in_thin_networks():
    a = ev.analyse(ev.Settings(n=20, k=3, seed=19))
    assert len(a.bridges) == 2 and all(a.incl_uniform[e] == pytest.approx(1.0) for e in a.bridges)


def test_sample_stats_fields_and_consistency():
    a = ev.analyse(FAST)
    st = ev.sample_stats(a, "wilson", 400)
    assert st["n_samples"] == 400 and not st["truncated"] and len(st["costs"]) == 400 and len(st["freq"]) == a.m and 0.0 <= st["mst_share"] <= 1.0
    assert st["steps_per_tree"] == pytest.approx(st["steps"] / 400) and st["ratio"] == pytest.approx(st["mean_cost"] / a.mst_cost) and 0.0 < st["leaf_share"] < 1.0 and st["max_degree"] >= 2.0
    assert st["max_dev"] == max(abs(f - q) for f, q in zip(st["freq"], a.incl)) and st["min_cost"] >= a.mst_cost - 1e-9
    assert ev.sample_stats(a, "wilson", 400) == st                                              # deterministisch


def test_step_budget_truncates_or_yields_nothing():
    a = ev.analyse(replace(FAST, n=20, b=16.0))
    st = ev.sample_stats(a, "wilson", 50, budget=5000)
    assert st["n_samples"] == 0 and st["truncated"] and st["mean_cost"] is None and st["max_dev"] is None and st["steps"] == 5000
    trees, steps = A.sample_trees("wilson", a.n, a.edges, a.weights, 1000, "budget", budget=3000)
    assert len(trees) < 1000 and steps <= 3000 or steps == 3000
    cmp = ev.sampler_compare(a, 20)
    assert cmp["kruskal"]["steps"] is not None and cmp["kruskal"]["n_samples"] == 20
    assert cmp["wilson"]["n_samples"] < 20


def test_sampler_compare_keys_and_ordering():
    cmp = ev.sampler_compare(ev.analyse(FAST), 200)
    assert set(cmp) == set(C.SAMPLERS) and cmp["wilson"]["steps"] < cmp["aldous"]["steps"] and all(cmp[k]["n_samples"] == 200 for k in cmp)
    assert all(0.0 < cmp[k]["max_dev"] < 0.3 for k in cmp)


def test_uniformity_test():
    u = ev.uniformity("textbook")
    assert u["trees"] == 21 and u["crit"] == pytest.approx(A.chi2_crit(20)) and u["wilson"]["chi2"] < u["crit"] and u["aldous"]["chi2"] < u["crit"] and u["kruskal"]["chi2"] > u["crit"]
    assert u["kruskal_exact_tv"] == pytest.approx(0.0413, abs=0.0005) and u["kruskal"]["tv"] > 0.03 > u["wilson"]["tv"]
    k4 = ev.uniformity("k4")
    assert k4["trees"] == 16 and k4["wilson"]["chi2"] < k4["crit"] and k4["kruskal"]["chi2"] > k4["crit"]


def test_reliability_table_and_curve():
    a = ev.analyse(ev.Settings(kind="textbook"))
    rows = ev.reliability_table(a, trials=4000)
    exact = A.connected_subgraph_counts(a.n, a.edges)
    for r in rows:
        assert abs(r["network"] - A.survival_exact(exact, a.m, r["p"])) < 4 * r["se"] + 1e-9
        assert r["tree"] == pytest.approx((1 - r["p"]) ** (a.n - 1)) and r["expected_trees"] == pytest.approx(21 * (1 - r["p"]) ** 4) and r["network"] >= r["tree"]
    curve = ev.reliability_curve(FAST, ps=(0.2, 0.5), seeds=range(200000, 200006), trials=300)
    assert curve[0.2]["median"] > curve[0.5]["median"] and curve[0.5]["tree"] == pytest.approx(0.5 ** 10)


def test_fail_patterns_are_deterministic():
    a = ev.analyse(FAST)
    p1, p2 = ev.fail_patterns(a, 0.3), ev.fail_patterns(a, 0.3)
    assert p1 == p2 and len(p1) == C.FAIL_RUNS and any(p[2] for p in p1)
    assert ev.fail_patterns(a, 0.3) != ev.fail_patterns(replace(a, settings=replace(a.settings, seed=4)), 0.3)


def test_correlation_returns_a_rank_correlation():
    c = ev.correlation(replace(FAST, k=4), 0.3, seeds=range(200000, 200012), trials=300)
    assert c["n_runs"] == 12 and -1.0 <= c["spearman"] <= 1.0 and c["tau_min"] < c["tau_max"]


def test_run_config_and_sweeps():
    out = ev.run_config(FAST)
    assert out["n_runs"] == 5
    for key in ("log10_tau", "cost_ratio", "p_mst", "wilson_steps", "aldous_steps", "steps_ratio", "kruskal_dev", "wilson_dev", "survive", "tree_survive", "expected_trees"):
        assert f"{key}_lo" in out and f"{key}_hi" in out and out[f"{key}_lo"] <= out[key] + 1e-12 <= out[f"{key}_hi"] + 2e-12
    for param, values in (("n", (6, 10)), ("k", (3, 1000)), ("terrain", (0.0, 1.0)), ("b", (0.0, 2.0)), ("p", (0.1, 0.5))):
        rows = ev.sweep(param, FAST, values)
        assert [r["value"] for r in rows] == list(values)
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    cold = ev.run_config(replace(FAST, n=20), b=16.0)
    assert cold["wilson_steps"] != cold["wilson_steps"]                                         # Budget: keine Zahl


def test_count_curve_and_textbook():
    rows = ev.count_curve(FAST, ns=(8, 12))
    assert [r["n"] for r in rows] == [9, 13] and all(r["log10_tau"] < r["log10_cayley"] for r in rows) and rows[1]["log10_tau"] > rows[0]["log10_tau"]
    a = ev.analyse(ev.Settings(kind="textbook"))
    assert a.tau == 21 and a.p_mst == pytest.approx(1 / 21) and a.exp_cost == pytest.approx(20.0)
