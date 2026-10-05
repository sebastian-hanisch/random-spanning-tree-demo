"""Unabhängiges Orakel für Zählen und Einschlusswahrscheinlichkeiten: exakte Bruchrechnung (fractions) statt Gleitkomma, Löschen-Zusammenziehen (tau(G) = tau(G - e) + tau(G / e)),
Satz von Foster (Summe der Einschlusswahrscheinlichkeiten = n - 1) und der Erwartungswert der Baumlänge >= MST auf Netzen mit 40 bis 60 Knoten und Gewichten über viele Zehnerpotenzen,
auf denen `numpy.linalg.pinv` bzw. `slogdet` jede Stelle verloren (pinv: Foster-Summe 39.79 statt 40 schon bei Temperatur 0 und 40 Knoten; Kostenverhältnis 0.97 < 1 bei b = 16)."""

import math
from fractions import Fraction

import pytest

import rst_algorithm as A
import rst_evaluation as ev


def exact_incl_and_log_partition(n, edges, weights):
    """Gauss-Jordan auf dem Laplace-Minor in exakten Brüchen: (w_e * R_eff(e), ln der gewichteten Baumzahl)."""
    w = [Fraction(x) for x in weights]
    m = n - 1
    lap = [[Fraction(0)] * m for _ in range(m)]
    for (u, v, *_), x in zip(edges, w):
        for a, b in ((u, v), (v, u)):
            if a < m:
                lap[a][a] += x
                if b < m:
                    lap[a][b] -= x
    aug = [row[:] + [Fraction(int(i == j)) for j in range(m)] for i, row in enumerate(lap)]
    log_det = 0.0
    for i in range(m):
        piv = aug[i][i]
        log_det += math.log(piv.numerator) - math.log(piv.denominator)
        aug[i] = [x / piv for x in aug[i]]
        for r in range(m):
            if r != i and aug[r][i] != 0:
                f = aug[r][i]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[i])]
    inv = [row[m:] for row in aug]
    g = lambda a, b: inv[a][b] if a < m and b < m else Fraction(0)
    return [float(x * (g(u, u) + g(v, v) - 2 * g(u, v))) for (u, v, *_), x in zip(edges, w)], log_det


def contract_count(n, pairs):
    """Zahl der Spannbäume eines Multigraphen durch Löschen und Zusammenziehen der ersten Kante."""
    pairs = [(u, v) for u, v in pairs if u != v]
    if n == 1:
        return 1
    if not pairs:
        return 0
    (u, v), rest = pairs[0], pairs[1:]
    keep = [x for x in range(n) if x != v]
    new = {x: i for i, x in enumerate(keep)}
    merged = [(new[u if a == v else a], new[u if b == v else b]) for a, b in rest]
    return contract_count(n, rest) + contract_count(n - 1, merged)


@pytest.mark.parametrize("n, k, seed, b", [(40, 3, 5, 0.0), (40, 3, 5, 8.0), (30, 4, 3, 16.0), (24, 6, 2, 16.0), (30, 1000, 17334, 4.0)])
def test_inclusion_probabilities_and_partition_function_against_exact_arithmetic(n, k, seed, b):
    a = ev.analyse(ev.Settings(kind="depot", n=n, k=k, terrain=0.3, seed=seed, b=b))
    exact, log_z = exact_incl_and_log_partition(a.n, a.edges, a.weights)
    assert max(abs(x - y) for x, y in zip(exact, a.incl)) < 1e-9
    assert abs(sum(a.incl) - (a.n - 1)) < 1e-9
    assert abs(A.log_partition(a.n, a.edges, a.weights) - log_z) < 1e-9


@pytest.mark.parametrize("n, k, seed", [(60, 6, 11), (60, 3, 7), (40, 3, 5)])
def test_expected_cost_is_never_below_the_mst_and_the_mst_probability_is_a_probability(n, k, seed):
    prev = None
    for b in (0.0, 2.0, 8.0, 16.0):
        a = ev.analyse(ev.Settings(kind="depot", n=n, k=k, terrain=0.3, seed=seed, b=b))
        assert abs(sum(a.incl) - (a.n - 1)) < 1e-6
        assert a.cost_ratio >= 1.0 - 1e-12 and 0.0 <= a.p_mst <= 1.0
        assert prev is None or a.cost_ratio <= prev + 1e-12                       # kälter = näher am MST
        prev = a.cost_ratio


def test_extreme_weights_on_a_small_graph_by_hand_arithmetic():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 1.0)]
    w = [1.0, 1e-30, 1e-60, 1e-90]
    exact, log_z = exact_incl_and_log_partition(4, edges, w)
    got = A.inclusion_probabilities(4, edges, w)
    assert exact[3] == pytest.approx(1.0) and got[3] == pytest.approx(1.0, abs=1e-12)           # Brücke: in jedem Baum
    assert got == pytest.approx(exact, abs=1e-12)
    assert A.log_partition(4, edges, w) == pytest.approx(log_z, abs=1e-9)
    assert A.log_partition(4, edges, w) == pytest.approx(math.log(1.0 * 1e-30 * 1e-90 + 1.0 * 1e-60 * 1e-90 + 1e-30 * 1e-60 * 1e-90), abs=1e-6)


@pytest.mark.parametrize("seed", range(12))
def test_tree_count_against_deletion_contraction(seed):
    a = ev.analyse(ev.Settings(kind="depot", n=7, k=3, terrain=0.3, seed=seed))
    assert a.tau == A.count_trees(a.n, a.edges) == contract_count(a.n, [(u, v) for u, v, *_ in a.edges])


def test_chi2_critical_value_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 4, 10, 15, 20, 50, 100, 500):
        for p in (0.5, 0.9, 0.95, 0.99, 0.999):
            assert A.chi2_crit(df, p) == pytest.approx(stats.chi2.ppf(p, df), abs=1e-9, rel=1e-12), (df, p)


def test_chi2_critical_value_against_table_values():
    # Tabellenwerte der Chi-Quadrat-Verteilung (hier gegen scipy gerechnet und mit gängigen Tafeln verglichen); unabhängig von scipy im Lauf
    assert A.chi2_crit(1, 0.95) == pytest.approx(3.841458820694124, abs=1e-9)
    assert A.chi2_crit(10, 0.95) == pytest.approx(18.307038053275146, abs=1e-9)
    assert A.chi2_crit(2, 0.95) == pytest.approx(2 * math.log(20), abs=1e-9)        # df = 2: Exponentialverteilung, Quantil = -2 ln(1 - p)
    assert round(A.chi2_crit(20), 1) == 45.3 and round(A.chi2_crit(15), 1) == 37.7
