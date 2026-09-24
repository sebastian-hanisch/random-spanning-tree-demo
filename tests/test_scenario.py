"""Instanz: Kopie treu zur Kruskal-Demo, Aufbau, Dichte, Determinismus, Lehrbuchbeispiel, Konstanten."""

import pytest

import rst_algorithm as A
import rst_constants as C
import rst_scenario as S


def connected(n, edges):
    uf = A.UnionFind(n, "full")
    for u, v, *_ in edges:
        uf.union(u, v)
    return uf.components == 1


def test_copy_matches_the_kruskal_demo():
    """Die Kopie ist treu: derselbe Plan wie in kruskal-demo (Seed 35, 30 Filialen, k = 6, Zuschlag 0.3): 113 Kanten, Baumkosten 466.63 (Zahlen aus kruskal-demo)."""
    inst = S.generate(30, 6, 0.3, 35)
    edges = list(inst.edges)
    assert inst.m == 113 and inst.n == 31
    assert sum(edges[e][2] for e in A.mst_edges(inst.n, edges, [e[2] for e in edges])) == pytest.approx(466.63, abs=0.01)
    assert S.generate(30, 3, 0.3, 35).m == 59 and S.generate(30, 1000, 0.3, 35).m == 465


@pytest.mark.parametrize("k", [3, 6, 20, 1000])
def test_instance_structure(k):
    inst = S.generate(25, k, 0.4, 3)
    assert inst.n == 26 and inst.depot == 0 and inst.kind == "depot"
    assert all(u < v for u, v, _w in inst.edges) and list(inst.edges) == sorted(inst.edges, key=lambda e: (e[0], e[1]))
    assert all(w > 0 for _u, _v, w in inst.edges) and connected(inst.n, inst.edges) and len({(u, v) for u, v, _w in inst.edges}) == inst.m
    if k >= 1000:
        assert inst.m == 26 * 25 // 2


def test_determinism_and_seed_dependence():
    assert S.generate(20, 6, 0.3, 5).edges == S.generate(20, 6, 0.3, 5).edges
    assert S.generate(20, 6, 0.3, 5).edges != S.generate(20, 6, 0.3, 6).edges


def test_textbook_instance():
    inst = S.textbook_instance()
    assert inst.n == 5 and inst.m == 7 and inst.labels == ("A", "B", "C", "D", "E") and inst.kind == "textbook"
    assert A.count_trees(inst.n, inst.edges) == 21


def test_constants_are_consistent():
    assert C.DEFAULT_K in C.K_OPTIONS and C.DEFAULT_B in C.B_OPTIONS and C.DEFAULT_P in C.P_OPTIONS and C.DEFAULT_SAMPLES in C.SAMPLE_OPTIONS and C.DEFAULT_TERRAIN in C.TERRAIN_OPTIONS
    assert C.N_MIN <= C.DEFAULT_N <= C.N_MAX and C.DEFAULT_SAMPLER in C.SAMPLERS and list(C.STEPS) == [1, 2, 3, 4] and C.B_OPTIONS[0] == 0.0 and list(C.B_OPTIONS) == sorted(C.B_OPTIONS)
    assert set(C.SAMPLER_LABELS) == set(C.SAMPLERS)
