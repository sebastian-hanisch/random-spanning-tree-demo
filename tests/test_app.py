"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jede Position der Schritt-Regler, beide Instanzen, Randwerte, Budgetwarnungen bei niedriger Temperatur, Würfel-Knopf, Permalink-Grenzen,
Instanzwechsel, Experimente und Sweeps auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rst_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="rst_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"Spannbäume", "Zufallsbaum / MST", "P(MST)", "Netz verbunden"} <= {m.label for m in at.metric}
    assert _metric(at, "Zufallsbaum").value == "1.87 x" and _metric(at, "P(MST)").value == "5.3e-16" and _metric(at, "Spannbäume").value == "10^15.3" and _metric(at, "Spannbäume").delta == "75 Kanten"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["n_slider"], ss["k_select"], ss["terrain_select"], ss["seed_input"], ss["b_select"], ss["sampler_select"], ss["p_select"], ss["samples_select"], ss["rst_step"]) == (
        p["kind"], p["n"], p["k"], p["terrain"], p["seed"], p["b"], p["sampler"], p["p"], p["samples"], p["step"])
    if p["walk_i"]:
        assert ss["walk_i"] == p["walk_i"]
    if p["fail_i"]:
        assert ss["fail_i"] == p["fail_i"]
    assert at.metric and at.get("plotly_chart")


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_every_step_runs_for_every_kind(step):
    for kind in C.KINDS:
        for k in (3, 1000):
            at = _run(kind_select=kind, k_select=k, n_slider=15, rst_step=step)
            _ok(at)
            assert at.get("plotly_chart") and at.session_state["rst_step"] == step


def test_walk_slider_walks_through_all_paths():
    at = _run(step=2, n_slider=12)
    _ok(at)
    wmax = int(at.slider(key="walk_i").max)
    assert 4 <= wmax <= 11
    for i in (0, 1, wmax):
        at.slider(key="walk_i").set_value(i).run()
        _ok(at)
        assert any(x.value.startswith(f"**Weg {i} von {wmax}") for x in at.markdown)
    assert any("Baum fertig" in x.value for x in at.markdown)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["walk_i"] <= int(at.slider(key="walk_i").max) <= 4


def test_fail_slider_walks_through_all_patterns():
    at = _run(step=4, k_select=4, seed_input=1, p_select=0.3)
    _ok(at)
    fmax = int(at.slider(key="fail_i").max)
    assert fmax == C.FAIL_RUNS - 1
    for j in (0, 10, fmax):
        at.slider(key="fail_i").set_value(j).run()
        _ok(at)
        assert any(x.value.startswith(f"**Muster {j + 1} von {C.FAIL_RUNS}") for x in at.markdown)


@pytest.mark.parametrize("kw", [
    dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(n_slider=C.N_MAX, k_select=1000), dict(n_slider=C.N_MIN, k_select=3), dict(kind_select="textbook"), dict(terrain_select=1.0), dict(terrain_select=0.0),
    dict(p_select=C.P_OPTIONS[0]), dict(p_select=C.P_OPTIONS[-1]), dict(samples_select=C.SAMPLE_OPTIONS[0]), dict(samples_select=C.SAMPLE_OPTIONS[-1], sampler_select="aldous"),
    dict(b_select=C.B_OPTIONS[1]), dict(b_select=C.B_OPTIONS[-1], n_slider=C.N_MIN), dict(sampler_select="kruskal", b_select=4.0),
])
def test_extreme_settings_run(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


def test_cold_temperature_shows_budget_warnings_instead_of_crashing():
    at = _run(step=3, n_slider=40, b_select=16.0)
    _ok(at)
    assert any("nicht einmal ein Baum" in w.value for w in at.warning)
    at2 = _run(step=2, n_slider=40, b_select=16.0)
    _ok(at2)
    assert any("mehr als" in w.value and "Zufallsschritte" in w.value for w in at2.warning)
    at3 = _run(step=3, n_slider=40, b_select=16.0, sampler_select="kruskal")
    _ok(at3)
    assert not any("nicht einmal ein Baum" in w.value for w in at3.warning) and at3.get("plotly_chart")


def test_truncated_sample_is_flagged():
    at = _run(step=3, n_slider=10, b_select=16.0)
    _ok(at)
    assert any("Schrittbudget war aufgebraucht" in m.value for m in at.markdown) or any("nicht einmal ein Baum" in w.value for w in at.warning)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(n="999", k="7", terrain="0.15", b="3.0", p="0.33", samples="33", sampler="nope", step="9", kind="nope").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["n_slider"], ss["k_select"], ss["terrain_select"], ss["b_select"], ss["p_select"], ss["samples_select"], ss["sampler_select"], ss["rst_step"], ss["kind_select"]) == (
        C.N_MAX, C.DEFAULT_K, C.DEFAULT_TERRAIN, C.DEFAULT_B, C.DEFAULT_P, C.DEFAULT_SAMPLES, C.DEFAULT_SAMPLER, 1, "depot")


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="depot", n="20", k="1000", terrain="0.6", seed="7", b="2.0", sampler="aldous", p="0.2", samples="200", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["n_slider"], ss["k_select"], ss["terrain_select"], ss["seed_input"], ss["b_select"], ss["sampler_select"], ss["p_select"], ss["samples_select"], ss["rst_step"]) == (
        20, 1000, 0.6, 7, 2.0, "aldous", 0.2, 200, 3)


def test_sidebar_shows_only_the_controls_that_matter():
    plain = _run()
    assert any(w.key == "n_widget" for w in plain.slider) and any(w.key == "k_widget" for w in plain.select_slider) and any(w.key == "terrain_widget" for w in plain.select_slider)
    assert any(n.key == "seed_widget" for n in plain.number_input) and any(w.key == "b_select" for w in plain.select_slider) and any(r.key == "sampler_select" for r in plain.radio)
    tb = _run(kind_select="textbook")
    assert not any(w.key == "n_widget" for w in tb.slider) and not any(w.key in ("k_widget", "terrain_widget") for w in tb.select_slider) and not any(n.key == "seed_widget" for n in tb.number_input)
    assert any(w.key == "b_select" for w in tb.select_slider) and any(w.key == "p_select" for w in tb.select_slider)


def test_textbook_numbers_and_hand_calculation():
    at = _run(kind_select="textbook")
    _ok(at)
    assert _metric(at, "P(MST)").value == "4.8 %" and _metric(at, "Zufallsbaum").value == "1.33 x"
    assert any("Von Hand" in x.value and "= Zahl der Spannbäume" in x.value and "**21**" in x.value for x in at.markdown)


def test_changing_the_instance_while_on_later_steps_does_not_crash():
    for step in (2, 3, 4):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="textbook"), dict(kind_select="depot", n_slider=8), dict(k_select=3), dict(b_select=8.0), dict(sampler_select="kruskal")):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_correlation_experiment_runs_on_demand():
    at = _run(n_slider=10)
    next(b for b in at.button if b.key == "corr_start").click().run()
    _ok(at)
    assert {"Rangkorrelation", "Baumzahl (Zehnerpotenz)", "Verbunden"} <= {m.label for m in at.metric}


@pytest.mark.parametrize("param", ["n", "k", "terrain", "b", "p"])
@pytest.mark.parametrize("metric", ["count", "cost", "steps", "reliability", "bias"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(n_slider=10, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_the_textbook_has_no_experiments():
    tb = _run(kind_select="textbook")
    assert not any(b.key in ("corr_start", "sweep_start") for b in tb.button)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any(all(w in m.value for w in ("Kirchhoff, G. (1847)", "Wilson, D. B. (1996)", "Broder, A. (1989)", "Aldous, D. J. (1990)", "Provan, J. S., & Ball, M. O. (1983)")) for m in at.markdown)
    assert any("Karger, D. R. (2001)" in m.value for m in at.markdown)
