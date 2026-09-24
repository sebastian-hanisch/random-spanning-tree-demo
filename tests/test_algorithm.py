"""Korrektheitskette (vor jeder Messung): Matrix-Baum-Satz gegen Aufzählung und Formeln, Einschlusswahrscheinlichkeiten (Kirchhoff, Foster), Sampler gegen die exakte Verteilung, Zufalls-Kruskal verzerrt,
Temperatur (Erwartungswert, P(MST)), Zuverlässigkeit (exakt, Formeln, Monte-Carlo, erwartete Baumzahl), Sonderfälle."""

import itertools
import math
import random

import pytest

import rst_algorithm as A
import rst_scenario as S


def rand_graph(rng, n, extra):
    """Zusammenhängender einfacher Zufallsgraph: (u, v, Kosten)."""
    pairs = set()
    for v in range(1, n):
        pairs.add((rng.randrange(0, v), v))
    for _ in range(extra):
        u, v = sorted(rng.sample(range(n), 2))
        pairs.add((u, v))
    return [(u, v, float(rng.randint(1, 6))) for u, v in sorted(pairs)]


def k4():
    return 4, [(u, v, 1.0) for u in range(4) for v in range(u + 1, 4)]


def counts_of(trees):
    out = {}
    for t in trees:
        out[t] = out.get(t, 0) + 1
    return out


# --- 1. Matrix-Baum-Satz ---------------------------------------------------------------------------------------------------------------------------


def test_matrix_tree_theorem_against_enumeration():
    rng = random.Random(1)
    checked = 0
    for _ in range(120):
        n = rng.randint(2, 7)
        edges = rand_graph(rng, n, rng.randint(0, 6))
        if len(edges) > 12:
            continue
        assert A.count_trees(n, edges) == len(A.enumerate_trees(n, edges))
        checked += 1
    assert checked > 90


def test_known_counts():
    for n in range(2, 10):
        complete = [(u, v, 1.0) for u in range(n) for v in range(u + 1, n)]
        assert A.count_trees(n, complete) == n ** (n - 2)                                    # Cayley
    assert A.count_trees(6, [(i, i + 1, 1.0) for i in range(5)]) == 1                        # Baum
    for n in range(3, 9):
        assert A.count_trees(n, [(i, (i + 1) % n, 1.0) for i in range(n)]) == n              # Kreis
    for a, b in ((2, 3), (3, 3), (2, 5), (3, 4)):
        kab = [(u, a + v, 1.0) for u in range(a) for v in range(b)]
        assert A.count_trees(a + b, kab) == a ** (b - 1) * b ** (a - 1)
    assert A.count_trees(4, [(0, 1, 1.0), (2, 3, 1.0)]) == 0                                 # unzusammenhängend
    assert A.count_trees(1, []) == 1 and A.count_trees(2, [(0, 1, 1.0)]) == 1


def test_weighted_partition_function_is_the_sum_over_trees():
    rng = random.Random(2)
    for _ in range(40):
        n = rng.randint(3, 7)
        edges = rand_graph(rng, n, rng.randint(0, 5))
        if len(edges) > 11:
            continue
        w = [rng.uniform(0.2, 3.0) for _ in edges]
        total = sum(A.tree_weight(w, t) for t in A.enumerate_trees(n, edges))
        assert math.exp(A.log_partition(n, edges, w)) == pytest.approx(total, rel=1e-9)
    assert A.log_partition(4, [(0, 1, 1.0), (2, 3, 1.0)], [1.0, 1.0]) == -math.inf


# --- 2. Einschlusswahrscheinlichkeiten -----------------------------------------------------------------------------------------------------------


def test_inclusion_probabilities_against_enumeration_and_foster():
    rng = random.Random(3)
    for it in range(60):
        n = rng.randint(3, 7)
        edges = rand_graph(rng, n, rng.randint(0, 5))
        if len(edges) > 11:
            continue
        w = None if it % 2 else [rng.uniform(0.2, 3.0) for _ in edges]
        p = A.inclusion_probabilities(n, edges, w)
        probs = A.exact_tree_probs(n, edges, [1.0] * len(edges) if w is None else w)
        for e in range(len(edges)):
            assert p[e] == pytest.approx(sum(q for t, q in probs.items() if e in t), abs=1e-9)
        assert sum(p) == pytest.approx(n - 1, abs=1e-9)                                     # Foster


def test_bridges_and_series_parallel_by_hand():
    tri = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    assert A.inclusion_probabilities(3, tri) == pytest.approx([2 / 3] * 3)
    bridge = tri + [(2, 3, 1.0)]
    p = A.inclusion_probabilities(4, bridge)
    assert p[3] == pytest.approx(1.0) and p[:3] == pytest.approx([2 / 3] * 3)
    heavy = A.inclusion_probabilities(3, tri, [1.0, 1.0, 100.0])
    assert heavy[2] > 0.98 and heavy[0] == pytest.approx(heavy[1])


# --- 3. Sampler gegen die exakte Verteilung -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["wilson", "aldous"])
def test_samplers_are_uniform_on_small_graphs(kind):
    graphs = {"K4": k4(), "Lehrbuch": (5, list(S.textbook_instance().edges)), "Brücke": (5, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (2, 4, 1.0), (1, 3, 1.0)])}
    for name, (n, edges) in graphs.items():
        probs = A.exact_tree_probs(n, edges, [1.0] * len(edges))
        trees, _ = A.sample_trees(kind, n, edges, None, 20000, f"u-{name}")
        assert all(A.is_spanning_tree(n, edges, t) for t in trees)
        c = counts_of(trees)
        assert A.chi2_stat(c, probs) < A.chi2_crit(len(probs) - 1), (kind, name)
        assert A.tv_distance(c, probs) < 0.02


@pytest.mark.parametrize("kind", ["wilson", "aldous"])
def test_weighted_samplers_follow_the_weight_product(kind):
    n, edges = 5, list(S.textbook_instance().edges)
    w = [0.3, 2.0, 1.0, 0.5, 1.7, 0.9, 1.2]
    probs = A.exact_tree_probs(n, edges, w)
    trees, _ = A.sample_trees(kind, n, edges, w, 30000, "weighted")
    assert A.chi2_stat(counts_of(trees), probs) < A.chi2_crit(len(probs) - 1)


def test_every_root_gives_the_same_distribution():
    n, edges = k4()
    probs = A.exact_tree_probs(n, edges, [1.0] * 6)
    walker = A.Walker(n, edges)
    for root in range(4):
        rng = A.make_rng("root", root)
        c = counts_of([A.wilson(walker, rng, root)[0] for _ in range(16000)])
        assert A.chi2_stat(c, probs) < A.chi2_crit(15), root
        c = counts_of([A.aldous_broder(walker, rng, root)[0] for _ in range(16000)])
        assert A.chi2_stat(c, probs) < A.chi2_crit(15), root


def test_random_kruskal_matches_its_exact_distribution_but_is_not_uniform():
    n, edges = 5, list(S.textbook_instance().edges)
    exact = A.random_kruskal_distribution(n, edges)
    assert sum(exact.values()) == pytest.approx(1.0) and set(exact) == set(A.enumerate_trees(n, edges))
    trees, _ = A.sample_trees("kruskal", n, edges, None, 30000, "rk")
    c = counts_of(trees)
    assert A.chi2_stat(c, exact) < A.chi2_crit(len(exact) - 1)                              # der Sampler trifft SEINE Verteilung
    uniform = {t: 1.0 / len(exact) for t in exact}
    assert A.tv_distance(c, uniform) > 0.03 and A.chi2_stat(c, uniform) > A.chi2_crit(len(exact) - 1)   # aber nicht die Gleichverteilung
    assert max(exact.values()) / min(exact.values()) > 1.3


# --- 4. Gültigkeit, Wiedergabe, Buchführung -----------------------------------------------------------------------------------------------------


def test_wilson_walks_reproduce_the_tree_and_count_steps():
    inst = S.generate(15, 6, 0.3, 4)
    edges = list(inst.edges)
    walker = A.Walker(inst.n, edges)
    lookup = {(min(u, v), max(u, v)): e for e, (u, v, _w) in enumerate(edges)}
    for seed in range(20):
        rng = A.make_rng("rec", seed)
        tree, steps, walks = A.wilson(walker, rng, 0, record=True)
        assert A.is_spanning_tree(inst.n, edges, tree)
        assert steps == sum(len(w["walk"]) - 1 for w in walks)
        from_paths = sorted({lookup[(min(a, b), max(a, b))] for w in walks for a, b in zip(w["path"], w["path"][1:])})
        assert tuple(from_paths) == tree
        assert all(w["erased"] == (len(w["walk"]) - 1) - (len(w["path"]) - 1) >= 0 for w in walks)
        rng2 = A.make_rng("rec", seed)
        assert A.wilson(walker, rng2, 0)[:2] == (tree, steps)                                # Aufzeichnung ändert den Zufallsstrom nicht


def test_samples_are_valid_trees_and_deterministic():
    inst = S.generate(12, 5, 0.3, 2)
    edges = list(inst.edges)
    for kind in ("wilson", "aldous", "kruskal"):
        a, sa = A.sample_trees(kind, inst.n, edges, None, 50, "det")
        b, sb = A.sample_trees(kind, inst.n, edges, None, 50, "det")
        assert a == b and sa == sb and all(A.is_spanning_tree(inst.n, edges, t) for t in a) and sa >= 50 * (inst.n - 1)
        assert a != A.sample_trees(kind, inst.n, edges, None, 50, "other")[0]
    with pytest.raises(ValueError):
        A.sample_trees("nope", inst.n, edges, None, 1, "x")


# --- 5. Temperatur ---------------------------------------------------------------------------------------------------------------------------------


def test_expected_cost_and_mst_probability_against_enumeration():
    rng = random.Random(5)
    for _ in range(25):
        n = rng.randint(4, 7)
        edges = rand_graph(rng, n, rng.randint(1, 5))
        if len(edges) > 11:
            continue
        costs = [e[2] + rng.random() for e in edges]
        for beta in (0.0, 0.3, 1.0, 3.0):
            w = A.edge_weights(costs, beta)
            probs = A.exact_tree_probs(n, edges, w)
            exp = sum(q * sum(costs[e] for e in t) for t, q in probs.items())
            assert A.expected_cost(n, edges, costs, beta) == pytest.approx(exp, rel=1e-9)
            mst = tuple(sorted(A.mst_edges(n, edges, costs)))
            assert A.p_mst(n, edges, costs, beta) == pytest.approx(probs[mst], rel=1e-9)


def test_expected_cost_is_minus_the_derivative_of_log_z():
    n, edges = 5, list(S.textbook_instance().edges)
    costs = [e[2] for e in edges]

    def lnz(beta):
        return A.log_partition(n, edges, [math.exp(-beta * c) for c in costs])

    h = 1e-5
    for beta in (0.0, 0.4, 1.2):
        deriv = (lnz(beta + h) - lnz(beta - h)) / (2 * h)
        assert -deriv == pytest.approx(A.expected_cost(n, edges, costs, beta), rel=1e-6)


def test_temperature_limits_and_monotonicity():
    inst = S.generate(12, 6, 0.3, 3)
    edges = list(inst.edges)
    raw = [e[2] for e in edges]
    mst = A.mst_edges(inst.n, edges, raw)
    costs = [c / (sum(raw[e] for e in mst) / len(mst)) for c in raw]                        # mittlere MST-Kante = 1: b = beta
    mst_cost = sum(costs[e] for e in mst)
    bs = (0.0, 0.5, 2.0, 8.0, 16.0)
    exp = [A.expected_cost(inst.n, edges, costs, b) for b in bs]
    assert exp == sorted(exp, reverse=True) and exp[-1] == pytest.approx(mst_cost, rel=0.01) and exp[0] > 2.0 * mst_cost
    pm = [A.p_mst(inst.n, edges, costs, b) for b in bs]
    assert pm == sorted(pm) and pm[0] == pytest.approx(1.0 / A.count_trees(inst.n, edges)) and pm[-1] > 0.5


def test_wilson_mean_cost_matches_the_expectation():
    inst = S.generate(10, 6, 0.3, 6)
    edges = list(inst.edges)
    costs = [e[2] for e in edges]
    for beta in (0.0, 0.05):
        w = A.edge_weights(costs, beta)
        trees, _ = A.sample_trees("wilson", inst.n, edges, w, 4000, f"temp{beta}")
        vals = [sum(costs[e] for e in t) for t in trees]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
        assert abs(mean - A.expected_cost(inst.n, edges, costs, beta)) < 4 * sd


# --- 6. Zuverlässigkeit ----------------------------------------------------------------------------------------------------------------------------


def test_exact_reliability_against_closed_forms():
    tree = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    c = A.connected_subgraph_counts(4, tree)
    cyc = [(i, (i + 1) % 5, 1.0) for i in range(5)]
    cc = A.connected_subgraph_counts(5, cyc)
    for p in (0.0, 0.05, 0.3, 0.7, 1.0):
        assert A.survival_exact(c, 3, p) == pytest.approx((1 - p) ** 3)
        assert A.survival_exact(cc, 5, p) == pytest.approx((1 - p) ** 5 + 5 * p * (1 - p) ** 4)
    n, edges = k4()
    ck = A.connected_subgraph_counts(n, edges)
    vals = [A.survival_exact(ck, 6, p) for p in (0.0, 0.1, 0.2, 0.4, 0.8, 1.0)]
    assert vals[0] == 1.0 and vals[-1] == 0.0 and vals == sorted(vals, reverse=True)
    assert sum(ck[3:4]) == 16 and ck[6] == 1 and ck[2] == 0                              # 16 Bäume mit 3 Kanten, 1 Graph mit allen, keiner mit 2


def test_monte_carlo_reliability_matches_exact():
    n, edges = 5, list(S.textbook_instance().edges)
    counts = A.connected_subgraph_counts(n, edges)
    for p in (0.1, 0.3):
        est, se = A.survival_mc(n, edges, p, 20000, "mc")
        assert abs(est - A.survival_exact(counts, len(edges), p)) < 4 * se
    assert A.survival_mc(n, edges, 0.2, 500, "x") == A.survival_mc(n, edges, 0.2, 500, "x")


def test_expected_surviving_trees_and_markov_bound():
    n, edges = 5, list(S.textbook_instance().edges)
    m = len(edges)
    for p in (0.1, 0.4):
        exp_enum = 0.0
        for mask in range(1 << m):
            kept = [edges[e] for e in range(m) if mask >> e & 1]
            k = len(kept)
            exp_enum += (1 - p) ** k * p ** (m - k) * (A.count_trees(n, kept) if len(kept) >= n - 1 else 0)
        assert A.expected_surviving_trees(n, edges, p) == pytest.approx(exp_enum, rel=1e-9)
        assert A.survival_exact(A.connected_subgraph_counts(n, edges), m, p) <= A.expected_surviving_trees(n, edges, p) + 1e-12


def test_failure_pattern_components():
    n, edges = 5, list(S.textbook_instance().edges)
    rng = A.make_rng("fp")
    seen = set()
    for _ in range(200):
        failed, comp, ok = A.failure_pattern(n, edges, 0.4, rng)
        assert ok == (len(set(comp)) == 1) and all(0 <= e < len(edges) for e in failed)
        seen.add(ok)
    assert seen == {True, False}


# --- 7. Statistik-Hilfen und Sonderfälle -----------------------------------------------------------------------------------------------------------


def test_statistics_helpers():
    assert A.chi2_crit(15) == pytest.approx(37.70, abs=0.3) and A.chi2_crit(5) == pytest.approx(20.52, abs=0.4)
    assert A.ranks([10, 20, 20, 5]) == [2.0, 3.5, 3.5, 1.0]
    assert A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0) and A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert A.spearman([1, 1, 1], [1, 2, 3]) == 0.0
    assert A.tree_stats(4, [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)], (0, 1, 2)) == (3, 3)


def test_special_cases():
    assert A.wilson(A.Walker(2, [(0, 1, 1.0)]), A.make_rng("a"))[:2] == ((0,), 1)
    assert A.aldous_broder(A.Walker(2, [(0, 1, 1.0)]), A.make_rng("a")) == ((0,), 1)
    assert A.inclusion_probabilities(2, [(0, 1, 1.0)]) == pytest.approx([1.0])
    assert A.random_kruskal(3, [(0, 1, 1.0), (1, 2, 1.0)], A.make_rng("b"))[0] == (0, 1)
    assert list(itertools.islice(A.enumerate_trees(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]), 5)) == [(0, 1), (0, 2), (1, 2)]
    w = A.edge_weights([5.0, 7.0], 100.0)
    assert w[0] == 1.0 and 0.0 <= w[1] < 1e-80


def test_weighted_random_kruskal_follows_the_temperature_but_is_not_gibbs():
    n, edges = 5, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0), (2, 3, 1.5), (3, 4, 1.0), (1, 3, 4.0), (2, 4, 2.5)]
    costs = [e[2] for e in edges]
    weights = A.edge_weights(costs, 1.0)
    unw, _ = A.sample_trees("kruskal", n, edges, None, 20000, "wk")
    wtd, _ = A.sample_trees("kruskal", n, edges, weights, 20000, "wk")
    gibbs = A.exact_tree_probs(n, edges, weights)
    mean = lambda ts: sum(sum(costs[e] for e in t) for t in ts) / len(ts)
    exact = sum(q * sum(costs[e] for e in t) for t, q in gibbs.items())
    assert mean(wtd) < mean(unw) - 0.5 and all(A.is_spanning_tree(n, edges, t) for t in wtd[:200])
    counts = {}
    for t in wtd:
        counts[t] = counts.get(t, 0) + 1
    assert A.tv_distance(counts, gibbs) > 0.02 and mean(wtd) < exact - 0.05                  # zu billig gegenüber der Gibbs-Verteilung
    assert A.random_kruskal(n, edges, A.make_rng("x"), [1.0] * 7)[0] == A.random_kruskal(n, edges, A.make_rng("x"))[0]
