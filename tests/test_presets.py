"""Presets: gültige Werte, Bänder (Median über die 5 festen Instanzen) und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen."""

import pytest

import rst_algorithm as A
import rst_constants as C
import rst_evaluation as ev
from rst_presets import PRESET_KEYS, SETTING_SPECS, STEP_SLIDERS


def _settings(p):
    return ev.Settings(p["kind"], p["n"], p["k"], p["terrain"], p["seed"], p["b"], p["sampler"], p["p"], p["samples"])


def _analysis(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def _has(name, *values):
    text = C.PRESET_HELP[name]
    for v in values:
        assert v in text, (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 9
    for name, p in C.PRESETS.items():
        assert set(p) == set(PRESET_KEYS), name
        for key, state_key in PRESET_KEYS.items():
            if state_key in STEP_SLIDERS:
                assert isinstance(p[key], int) and p[key] >= 0
                continue
            spec = SETTING_SPECS[state_key]
            assert spec.caster(p[key]) == p[key], (name, key)
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip(), name
        if p["walk_i"]:
            assert p["step"] == 2
        if p["fail_i"]:
            assert p["step"] == 4 and p["fail_i"] < C.FAIL_RUNS


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_preset_bands_over_the_five_fixed_instances(name):
    metric, lo, hi = C.PRESET_EXPECTED_BANDS[name]
    assert lo <= ev.run_config(_settings(C.PRESETS[name]))[metric] <= hi


def test_help_standard():
    a = _analysis("Standardfall (Voreinstellung)")
    assert (a.m, len(a.bridges)) == (75, 0) and a.log10_tau == pytest.approx(15.28, abs=0.005) and a.tau == pytest.approx(1.9e15, rel=0.03)
    assert (round(100 * min(a.incl)), round(100 * max(a.incl))) == (20, 37)
    assert a.cost_ratio == pytest.approx(1.87, abs=0.005) and a.exp_cost == pytest.approx(666.46, abs=0.005) and a.mst_cost == pytest.approx(357.32, abs=0.005) and a.p_mst == pytest.approx(5.3e-16, rel=0.03)
    _has("Standardfall (Voreinstellung)", "75 Kandidatenkanten", "1.9 x 10^15", "20 bis 37 %", "1.87-mal", "666.46 gegen 357.32", "5.3 x 10^-16")


def test_help_textbook():
    a = _analysis("Lehrbuchbeispiel")
    assert (a.tau, a.m) == (21, 7) and (round(100 * min(a.incl)), round(100 * max(a.incl))) == (48, 62) and a.exp_cost == pytest.approx(20.0) and a.mst_cost == 15.0 and a.cost_ratio == pytest.approx(4 / 3, abs=0.005)
    _has("Lehrbuchbeispiel", "21 Spannbäume", "48 bis 62 %", "20.00", "1.33-fach", "1/21 = 4.8 %")


def test_help_thin_network():
    a = _analysis("Dünnes Netz (Brücken)")
    assert (a.m, len(a.bridges), a.tau) == (37, 2, 12_720_000) and a.cost_ratio == pytest.approx(1.26, abs=0.005)
    _has("Dünnes Netz (Brücken)", "37 Kanten", "12 720 000", "2 Brücken", "1.26-mal")


def test_help_complete_graph():
    a = _analysis("Vollständiger Graph (Cayley)")
    assert a.n == 10 and a.tau == 10 ** 8 and all(q == pytest.approx(0.2) for q in a.incl)
    assert a.cost_ratio == pytest.approx(2.28, abs=0.005) and a.exp_cost == pytest.approx(473.06, abs=0.005) and a.mst_cost == pytest.approx(207.70, abs=0.005)
    _has("Vollständiger Graph (Cayley)", "10^8 = 100 000 000", "2/n", "20 %", "2.28-mal", "473.06 gegen 207.70")


def test_help_wilson_in_action():
    name = "Wilson in Aktion"
    p = C.PRESETS[name]
    a = _analysis(name)
    walker = A.Walker(a.n, a.edges, a.weights)
    tree, steps, walks = A.wilson(walker, A.make_rng(a.settings.seed, a.settings.kind, "play"), 0, record=True)
    w = walks[p["walk_i"] - 1]
    assert (len(walks), len(w["walk"]) - 1, w["erased"], len(w["path"]) - 1, steps) == (7, 31, 27, 4, 39)
    assert sum(a.costs[e] for e in tree) == pytest.approx(506.15, abs=0.005) and a.mst_cost == pytest.approx(248.04, abs=0.005)
    cmp = ev.sampler_compare(a, 100)
    assert cmp["wilson"]["steps"] == pytest.approx(25.4, abs=0.05) and cmp["aldous"]["steps"] == pytest.approx(46.1, abs=0.05)
    _has(name, "Weg 1 von 7", "31 Zufallsschritte", "27", "4 Kanten", "39 Schritte", "506.15 gegen MST 248.04", "25.4", "46.1")


def test_help_cold():
    name = "Kalt: der MST kommt heraus"
    a = _analysis(name)
    st = ev.sample_stats(a)
    assert a.p_mst == pytest.approx(0.748, abs=0.0005) and st["mst_share"] == pytest.approx(0.722, abs=0.0005) and st["n_samples"] == 500
    assert st["mean_cost"] == pytest.approx(193.99, abs=0.005) and a.mst_cost == pytest.approx(190.97, abs=0.005) and a.exp_cost == pytest.approx(193.94, abs=0.005) and st["steps_per_tree"] == pytest.approx(321, abs=0.5)
    _has(name, "74.8 %", "72.2 %", "500", "193.99 gegen 190.97", "193.94", "321 Zufallsschritte")


def test_help_kruskal():
    name = "Kruskal mit Zufallsordnung"
    u = ev.uniformity("textbook")
    assert (u["wilson"]["chi2"], u["aldous"]["chi2"], u["kruskal"]["chi2"], u["crit"]) == pytest.approx((15.1, 17.5, 189.9, 45.4), abs=0.05) and u["kruskal_exact_tv"] == pytest.approx(0.041, abs=0.0005)
    _has(name, "2000", "20 000", "15.1", "17.5", "45.4", "189.9", "4.1 %")


def test_help_high_failure():
    name = "Hohe Ausfallquote"
    p = C.PRESETS[name]
    a = _analysis(name)
    pat = ev.fail_patterns(a, a.settings.p)[p["fail_i"]]
    est, _se = ev.survive(a, a.settings.p, 1000)
    row = next(r for r in ev.reliability_table(a, ps=(0.3,), trials=1000))
    assert a.m == 51 and a.tau == pytest.approx(1.6e11, rel=0.03) and len(pat[0]) == 16 and len(set(pat[1])) == 2 and est == pytest.approx(0.848, abs=0.0005)
    assert row["tree"] == pytest.approx(0.0008, abs=5e-5) and row["expected_trees"] == pytest.approx(1.3e8, rel=0.03)
    _has(name, "51 Kanten", "1.6 x 10^11", "16 Kanten", "2 Teile", "84.8 %", "0.08 %", "1.3 x 10^8")


def test_help_fragile():
    name = "Dünn und zerbrechlich"
    p = C.PRESETS[name]
    a = _analysis(name)
    pat = ev.fail_patterns(a, a.settings.p)[p["fail_i"]]
    est, _se = ev.survive(a, a.settings.p, 1000)
    row = ev.reliability_table(a, ps=(0.2,), trials=1000)[0]
    assert a.m == 37 and a.tau == pytest.approx(2.0e7, rel=0.03) and len(a.bridges) == 1 and len(pat[0]) == 7 and len(set(pat[1])) == 2 and est == pytest.approx(0.620, abs=0.0005)
    assert row["tree"] == pytest.approx(0.0115, abs=0.00005) and row["expected_trees"] == pytest.approx(2.3e5, rel=0.03)
    _has(name, "37 Kanten", "2.0 x 10^7", "1 Brücke", "7 Kanten", "2 Teile", "62.0 %", "1.15 %", "2.3 x 10^5")
