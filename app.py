"""Zufällige Spannbäume und Kirchhoff – zählen, würfeln, Zuverlässigkeit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes und letztes Stück der Spannbaum-Reihe der "Konzepte"-Reihe: bisher wurde optimiert. Hier wird gezählt und gewürfelt: wie viele Spannbäume hat ein Netz (Matrix-Baum-Satz von Kirchhoff), wie würfelt man einen
gleichverteilt (Wilson, Aldous-Broder - und warum "Kruskal mit Zufallsordnung" das nicht tut), wie teuer ist ein Zufallsbaum gegenüber dem billigsten (Temperatur), und wie zuverlässig ist ein Netz, dessen Kanten
ausfallen können (ein Spannbaum überlebt).

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import pandas as pd
import streamlit as st

import rst_algorithm as A
import rst_constants as C
import rst_evaluation as ev
from rst_evaluation import SWEEP_LABELS, SWEEP_TICKS, Settings, analyse
from rst_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from rst_visualization import (
    METHOD_LABELS,
    build_cost_hist,
    build_count_curve,
    build_failure_run,
    build_freq_scatter,
    build_inclusion_map,
    build_reliability,
    build_sweep,
    build_uniformity,
    build_wilson_step,
    fmt_pow10,
)

st.set_page_config(page_title="Zufällige Spannbäume – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _stats(settings):
    return ev.sample_stats(analyse(settings))


WALK_BUDGET = 300_000


@st.cache_data(show_spinner=False)
def _walks(settings):
    a = analyse(settings)
    walker = A.Walker(a.n, a.edges, a.weights)
    try:
        return A.wilson(walker, A.make_rng(settings.seed, settings.kind, "play"), 0, record=True, max_steps=WALK_BUDGET)
    except A.BudgetExceeded:
        return None


@st.cache_data(show_spinner=False)
def _compare(settings):
    return ev.sampler_compare(analyse(settings), 100)


@st.cache_data(show_spinner=False)
def _reliability(settings):
    return ev.reliability_table(analyse(settings), trials=1000)


@st.cache_data(show_spinner=False)
def _patterns(settings):
    return ev.fail_patterns(analyse(settings), settings.p)


@st.cache_data(show_spinner=False)
def _count_curve(base):
    return ev.count_curve(base)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return ev.sweep(param, base)


@st.cache_data(show_spinner=False)
def _correlation(base):
    return ev.correlation(base, base.p)


def pct(x):
    return f"{100.0 * x:.1f} %"


def num(x):
    return f"{x:.2f}"


def prob(x):
    return f"{100.0 * x:.1f} %" if x >= 0.001 else f"{x:.1e}"


st.title("🎲 Zufällige Spannbäume – zählen und würfeln statt optimieren")
st.markdown(
    """
**Elftes und letztes Stück der Spannbaum-Reihe.** Bisher wurde optimiert: der *billigste* Baum. Ein Netz hat aber unvorstellbar viele Spannbäume, und ihre Gesamtheit hat Struktur. Vier Fragen, alle gemessen:
**(1) Zählen** - wie viele Spannbäume gibt es (Matrix-Baum-Satz, Kirchhoff 1847), und wie wahrscheinlich liegt eine Kante in einem zufälligen Baum? **(2) Würfeln** - wie zieht man einen Baum gleichverteilt
(Wilson, Aldous-Broder), und warum funktioniert "Kruskal mit Zufallsordnung" *nicht*? **(3) Vom Zufall zum MST** - wie teuer ist ein Zufallsbaum, und wie weit muss man die "Temperatur" absenken, bis der billigste Baum
herauskommt? **(4) Zuverlässigkeit** - fällt jede Kante mit Wahrscheinlichkeit p aus, bleibt das Netz genau dann verbunden, wenn *ein* Spannbaum überlebt.
"""
)
st.caption(
    "Setzt auf [kruskal-demo](https://github.com/sebastian-hanisch/kruskal-demo) und [mst-sensitivity-demo](https://github.com/sebastian-hanisch/mst-sensitivity-demo) auf (Karte, Kruskal, Union-Find, Rauschen). "
    "Letztes Stück der Reihe: die Spannbaum-Reihe ist damit vollständig."
)

with st.expander("So funktionieren die Verfahren", expanded=True):
    st.markdown(
        """
1. **Zählen (Kirchhoff):** Laplace-Matrix des Netzes (Grad auf der Diagonale, −1 für jede Kante), eine Zeile und Spalte streichen, Determinante bilden: das ist die **Zahl der Spannbäume**. Mit Kantengewichten
   ist es die Summe über alle Bäume des Produkts der Gewichte. Die Wahrscheinlichkeit, dass ein zufälliger Baum eine Kante enthält, ist ihr Gewicht mal ihr **effektiver Widerstand** (Foster: die Summe über alle Kanten ist n − 1;
   Brücken liegen in jedem Baum).
2. **Wilson:** der Baum startet an einem Knoten; von jedem Knoten außerhalb läuft ein **Zufallsweg**, bis er den Baum trifft; **Schleifen werden gelöscht** (der zuletzt gewählte Ausgang zählt), der schleifenfreie Weg
   kommt in den Baum. **Aldous-Broder:** ein Zufallsweg, bis jeder Knoten besucht wurde; die Kante des ersten Besuchs jedes Knotens gehört zum Baum. Beide liefern jeden Baum mit gleicher Wahrscheinlichkeit.
   **Kruskal mit Zufallsordnung** (Kanten mischen, Kruskal) sieht plausibel aus, bevorzugt aber manche Bäume.
3. **Temperatur:** mit dem Kantengewicht exp(−β · Länge) wird das Gewicht eines Baums zu exp(−β · Baumlänge): eine Gibbs-Verteilung über alle Bäume. **b = 0**: gleichverteilt; **b groß**: fast nur billige Bäume, im Grenzfall der
   billigste (MST). b ist β mal die mittlere Kantenlänge des MST.
4. **Zuverlässigkeit:** jede Kante fällt mit Wahrscheinlichkeit p aus; das Netz bleibt verbunden genau dann, wenn ein Spannbaum überlebt. Ein **fester** Baum überlebt nur mit (1 − p)^(n−1); die **erwartete Zahl**
   überlebender Bäume ist τ · (1 − p)^(n−1) (nur eine obere Schranke für die Wahrscheinlichkeit, wenn sie unter 1 liegt).
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:3], preset_names[3:6], preset_names[6:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.radio("Instanz", options=list(C.KINDS), format_func=lambda v: C.KIND_LABELS[v], key="kind_select", help="Depot und Filialen: Karte mit Kandidatengraph. Lehrbuchbeispiel: 5 Knoten, von Hand nachzurechnen.")
    if kind != "textbook":
        n = st.slider("Filialen n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Anzahl der Filialen (das Depot kommt dazu).")
        k = st.select_slider("Kandidaten k (nächste Nachbarn)", options=list(C.K_OPTIONS), value=int(ss["k_select"]), key="k_widget", on_change=store_from_widget, args=("k_select",),
                             format_func=lambda v: "vollständig" if v >= 1000 else str(v), help="Je Knoten die k nächsten Nachbarn als Kandidatenkanten; bei kleinem k gibt es Brücken, im vollständigen Graphen n^(n-2) Bäume (Cayley).")
        terrain = st.select_slider("Geländezuschlag", options=list(C.TERRAIN_OPTIONS), value=float(ss["terrain_select"]), key="terrain_widget", on_change=store_from_widget, args=("terrain_select",),
                                   help="Kosten = Länge x Faktor aus [1, 1 + Zuschlag].")
    else:
        n, k, terrain = C.DEFAULT_N, C.DEFAULT_K, C.DEFAULT_TERRAIN
    b = st.select_slider("Temperatur b (0 = gleichverteilt)", options=list(C.B_OPTIONS), key="b_select", format_func=lambda v: "0 (gleichverteilt)" if v == 0 else f"{v:g}",
                         help="Gewicht exp(-beta * Länge) je Kante mit beta = b / mittlere MST-Kantenlänge. Je größer b, desto billiger die gewürfelten Bäume; b = 16 kommt dem MST nahe.")
    sampler = st.radio("Würfelverfahren", options=list(C.SAMPLERS), format_func=lambda v: C.SAMPLER_LABELS[v], key="sampler_select",
                       help="Für Schritt 3 (Verteilung): Wilson und Aldous-Broder würfeln gleichverteilt bzw. nach dem Gewichtsprodukt; Kruskal mit Zufallsordnung ordnet die Kanten nach exponentiellen Uhren mit Rate exp(-β · Länge): es folgt der Temperatur, ist aber verzerrt.")
    samples = st.select_slider("Stichprobe (Bäume)", options=list(C.SAMPLE_OPTIONS), key="samples_select", help="Zahl der gewürfelten Bäume in Schritt 3.")
    p = st.select_slider("Kantenausfall p", options=list(C.P_OPTIONS), key="p_select", format_func=lambda v: f"{v:g}", help="Schritt 4: jede Kante fällt unabhängig mit dieser Wahrscheinlichkeit aus.")
    if kind != "textbook":
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED

sync_query_params({"kind_select": kind, "n_slider": int(ss["n_slider"]), "k_select": int(ss["k_select"]), "terrain_select": float(ss["terrain_select"]), "seed_input": int(ss["seed_input"]),
                   "b_select": float(b), "sampler_select": sampler, "p_select": float(p), "samples_select": int(samples), "rst_step": int(ss["rst_step"])})

settings = Settings(kind, int(n), int(k), float(terrain), int(seed), float(b), sampler, float(p), int(samples))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
edges = a.edges
labels = inst.labels


def ename(e):
    u, v, *_ = edges[e]
    return f"{labels[u]}–{labels[v]}" if labels is not None else f"{u}–{v}"


# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Zählen, würfeln, zuverlässig")
step = st.select_slider("Schritt", options=list(C.STEPS), key="rst_step", format_func=lambda s: C.STEPS[s])

if step == 1:
    st.markdown(f"**{inst.n} Knoten, {a.m} Kandidatenkanten:** der Matrix-Baum-Satz zählt **{a.tau:,} Spannbäume**".replace(",", ".") + f" (≈ {fmt_pow10(a.log10_tau)}) - exakt, ohne einen einzigen aufzuzählen. "
                f"Die Farbe einer Kante zeigt, in wie viel Prozent aller Bäume sie liegt" + (" (bei Temperatur b = " + f"{b:g}" + ": gewichtet nach exp(−β · Länge))" if b > 0 else " (alle Bäume gleich wahrscheinlich)") +
                f"; die Summe über alle Kanten ist genau n − 1 = {inst.n - 1} (Satz von Foster): {sum(a.incl):.2f}. **{len(a.bridges)}** Brücke(n) liegen in jedem Baum.")
    st.plotly_chart(build_inclusion_map(inst, edges, a.incl, a.mst), width="stretch", key="s1_map")
    st.caption("Dünne schwarze Linie = billigster Baum (MST). Ohne Temperatur (b = 0) sind MST-Kanten nicht wahrscheinlicher als andere: die Gleichverteilung kennt keine Kosten.")
    if kind == "textbook":
        lap = [[0] * inst.n for _ in range(inst.n)]
        for u, v, *_ in edges:
            lap[u][u] += 1
            lap[v][v] += 1
            lap[u][v] -= 1
            lap[v][u] -= 1
        names = list(labels)
        st.markdown(f"**Von Hand:** Laplace-Matrix (Grad auf der Diagonale, −1 je Kante), die letzte Zeile und Spalte ({names[-1]}) gestrichen; ihre Determinante ist **{a.tau}** = Zahl der Spannbäume.")
        st.dataframe(pd.DataFrame([row[:-1] for row in lap[:-1]], index=names[:-1], columns=names[:-1]), width="stretch")
    else:
        rows = []
        for e in sorted(range(a.m), key=lambda e: -a.incl[e])[:6]:
            rows.append({"Kante": ename(e), "Länge": round(a.costs[e], 1), "in Bäumen (%)": round(100 * a.incl[e], 1), "im MST": "ja" if e in set(a.mst) else "nein"})
        st.markdown("**Die sechs wahrscheinlichsten Kanten:**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.plotly_chart(build_count_curve(_count_curve(replace(settings, seed=0))), width="stretch", key="count_curve")
        st.caption("Zahl der Spannbäume (Zehnerpotenz) über die Knotenzahl: der Kandidatengraph (Median über 5 feste Instanzen) gegen den vollständigen Graphen n^(n−2). Weniger Kanten, weit weniger Bäume.")
elif step == 2:
    played = _walks(settings)
    if played is None:
        st.warning(f"Bei Temperatur b = {b:g} braucht Wilson für einen einzigen Baum mehr als {WALK_BUDGET:,} Zufallsschritte".replace(",", ".") + " - die Wiedergabe wird abgebrochen. Je kälter, desto länger laufen die Wege "
                   "(vermutlich fangen die Kanten mit hohem Gewicht den Zufallsweg ein); stellen Sie die Temperatur niedriger oder verkleinern Sie das Netz.")
        played = ((), 0, [])
    tree_w, steps_w, walks = played
    wmax = len(walks)
    if "walk_i" in ss:
        ss["walk_i"] = min(max(0, int(ss["walk_i"])), wmax)
    i = st.slider("Zufallsweg", 0, wmax, key="walk_i", help="0 = nur die Wurzel; jeder weitere Schritt ist ein Zufallsweg von einem noch nicht angeschlossenen Knoten bis zum Baum.") if wmax > 0 else 0
    if i == 0:
        st.markdown(f"**Weg 0 von {wmax}:** der Baum besteht nur aus der Wurzel (Depot). Von jedem Knoten außerhalb läuft gleich ein Zufallsweg bis zum Baum.")
    else:
        w = walks[i - 1]
        st.markdown(f"**Weg {i} von {wmax}:** Start bei Knoten {w['start']}, der Zufallsweg braucht **{len(w['walk']) - 1} Schritte**, davon werden **{w['erased']}** als Schleifen gelöscht; der schleifenfreie Weg hat "
                    f"**{len(w['path']) - 1}** Kante(n) und kommt in den Baum." + (f" **Baum fertig:** {steps_w} Schritte insgesamt, Länge {num(sum(a.costs[e] for e in tree_w))} (MST {num(a.mst_cost)})." if i == wmax else ""))
    st.plotly_chart(build_wilson_step(inst, edges, walks, i), width="stretch", key=f"s2_map_{i}")
    st.caption("Grün = Baum bisher, orange gepunktet = Zufallsweg mit Schleifen, orange dick = schleifenfreier Weg, roter Knoten = Start des laufenden Wegs. Wilson gewichtet die Schritte mit exp(−β · Länge): bei höherer Temperatur laufen die Wege länger.")
    cmp = _compare(settings)
    rows = [{"Verfahren": METHOD_LABELS[kd], "Bäume gewürfelt": cmp[kd]["n_samples"], "Schritte je Baum": None if cmp[kd]["steps"] is None else round(cmp[kd]["steps"], 1),
             "Abweichung von den exakten Wahrscheinlichkeiten": None if cmp[kd]["max_dev"] is None else round(cmp[kd]["max_dev"], 3)} for kd in C.SAMPLERS]
    st.markdown("**Schritte je Baum** (Mittel über bis zu 100 Bäume; bei Kruskal = betrachtete Kanten; leer = Schrittbudget schon beim ersten Baum aufgebraucht) und größte Abweichung der Kantenhäufigkeit von der exakten Wahrscheinlichkeit (Rauschen einer Stichprobe von 100: etwa 0.1):")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
elif step == 3:
    stt = _stats(settings)
    if stt["n_samples"] == 0:
        st.warning(f"Mit {C.SAMPLER_LABELS[sampler]} bei Temperatur b = {b:g} wird nicht einmal ein Baum innerhalb des Schrittbudgets ({ev.BUDGET:,} Zufallsschritte) fertig".replace(",", ".") + ": bei niedriger Temperatur laufen die Zufallswege sehr lange. "
                   "Stellen Sie die Temperatur niedriger, verkleinern Sie das Netz oder wählen Sie \"Kruskal mit Zufallsordnung\" (das ohne Zufallswege auskommt, aber verzerrt ist).")
    else:
        st.markdown(f"**{stt['n_samples']} Bäume** mit {C.SAMPLER_LABELS[sampler]}, Temperatur b = {b:g}: mittlere Baumlänge **{num(stt['mean_cost'])}** (exakter Erwartungswert {num(a.exp_cost)}, MST {num(a.mst_cost)}); der billigste gewürfelte "
                    f"Baum {num(stt['min_cost'])}; **{stt['mst_share'] * 100:.1f} %** der Würfe sind der MST (exakt {prob(a.p_mst)}). Blattanteil {stt['leaf_share'] * 100:.0f} %, größter Grad im Mittel {stt['max_degree']:.1f}. "
                    f"Größte Abweichung einer Kantenhäufigkeit von ihrer exakten Wahrscheinlichkeit: **{stt['max_dev']:.3f}**." + (f" ⚠️ Nur {stt['n_samples']} von {stt['wanted']} Bäumen: das Schrittbudget war aufgebraucht ({stt['steps_per_tree']:,.0f} Schritte je Baum).".replace(",", ".") if stt["truncated"] else ""))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(build_cost_hist(stt["costs"], a.mst_cost, a.exp_cost, stt["mean_cost"]), width="stretch", key="s3_hist")
        with c2:
            st.plotly_chart(build_freq_scatter(a.incl, stt["freq"], a.mst), width="stretch", key="s3_scatter")
        st.caption("Links: Verteilung der Baumlängen; rechts: jede Kante als Punkt (Häufigkeit gegen exakte Wahrscheinlichkeit) - auf der Diagonalen würfelt das Verfahren richtig. Kruskal mit Zufallsordnung weicht systematisch ab.")
    u = ev.uniformity("textbook")
    st.markdown(f"**Gleichverteilungstest** auf dem Lehrbuch-Graphen ({u['trees']} Spannbäume, alle aufgezählt, je 20 000 Würfe): Chi-Quadrat gegen den kritischen Wert {u['crit']:.1f} (99.9 %).")
    st.plotly_chart(build_uniformity(u), width="stretch", key="s3_uniform")
    st.caption(f"Wilson ({u['wilson']['chi2']:.0f}) und Aldous-Broder ({u['aldous']['chi2']:.0f}) liegen unter dem kritischen Wert, Kruskal mit Zufallsordnung ({u['kruskal']['chi2']:.0f}) weit darüber; seine exakte Abweichung von der "
               f"Gleichverteilung (halber Betrag der Wahrscheinlichkeitsunterschiede) beträgt {u['kruskal_exact_tv'] * 100:.1f} %.")
else:
    rows_r = _reliability(settings)
    cur = next(r for r in rows_r if r["p"] == float(p))
    pats = _patterns(settings)
    if "fail_i" in ss:
        ss["fail_i"] = min(max(0, int(ss["fail_i"])), len(pats) - 1)
    j = st.slider("Ausfallmuster", 0, len(pats) - 1, key="fail_i", help="Jedes Muster ist eine Zufallsauswahl ausgefallener Kanten bei der eingestellten Ausfallwahrscheinlichkeit.")
    failed, comp, ok = pats[j]
    ncomp = len(set(comp))
    st.markdown(f"**Muster {j + 1} von {len(pats)} bei p = {p:g}:** {len(failed)} von {a.m} Kanten fallen aus - das Netz " + ("**bleibt verbunden** (ein Spannbaum überlebt)." if ok else f"**zerfällt** in {ncomp} Teile."))
    st.plotly_chart(build_failure_run(inst, edges, pats[j]), width="stretch", key=f"s4_map_{j}")
    st.caption("Rot gestrichelt = ausgefallen, grau = überlebt; Knotenfarben = Komponenten (die größte grün).")
    st.plotly_chart(build_reliability(rows_r), width="stretch", key="s4_curve")
    st.caption(f"Bei p = {p:g}: das Netz bleibt in **{pct(cur['network'])}** der Fälle verbunden (Monte-Carlo, ±{100 * 2 * cur['se']:.1f} %), ein fester Baum nur in **{pct(cur['tree'])}**. Erwartete Zahl überlebender Bäume: "
               f"**{cur['expected_trees']:.3g}** (τ · (1 − p)^(n−1)) - {'größer als 1, die Schranke sagt nichts' if cur['expected_trees'] >= 1 else 'kleiner als 1, eine obere Schranke für die Verbundenheitswahrscheinlichkeit'}.")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Wie viele, wie teuer, wie zuverlässig?")
rel = ev.survive(a, float(p), 1000)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spannbäume", f"10^{a.log10_tau:.1f}", delta=f"{a.m} Kanten", delta_color="off")
m2.metric("Zufallsbaum / MST", f"{a.cost_ratio:.2f} x", delta=f"b = {b:g}", delta_color="off")
m3.metric("P(MST)", prob(a.p_mst), delta="exakt", delta_color="off")
m4.metric("Netz verbunden", pct(rel[0]), delta=f"Baum {pct((1 - p) ** (a.n - 1))}", delta_color="off")
st.caption(f"{a.tau:,} Spannbäume".replace(",", ".") + f" (exakt, Kirchhoff), {len(a.bridges)} Brücke(n); P(MST) = Wahrscheinlichkeit, dass ein Wurf der billigste Baum ist; \"Baum\" = ein fester Baum bleibt bei Kantenausfall intakt. Erwartete Baumlänge bei Temperatur b = {b:g}: {num(a.exp_cost)} gegen {num(a.mst_cost)} für den MST ({a.cost_ratio:.2f}-fach). Verbundenheit bei Kantenausfall p = {p:g} per Monte-Carlo (1000 Läufe, ±{100 * 2 * rel[1]:.1f} %).")

st.markdown("---")

# --- Experimente auf Abruf ---------------------------------------------------------------------------------------------------------------------

base = replace(settings, seed=0)
if kind != "textbook":
    st.subheader("🔗 Hängt die Zuverlässigkeit an der Zahl der Spannbäume?")
    st.caption("50 Instanzen mit den Einstellungen der Seitenleiste (nur der Seed wechselt): Rangkorrelation zwischen der Zahl der Spannbäume und der Wahrscheinlichkeit, verbunden zu bleiben (Monte-Carlo, 1000 Läufe je Instanz).")
    if st.button("Korrelations-Experiment über 50 Instanzen (dauert einige Sekunden)", key="corr_start"):
        ss["corr_done"] = ss.get("corr_done", set()) | {base}
    if base in ss.get("corr_done", set()):
        with st.spinner("Rechne..."):
            r = _correlation(base)
        q1, q2, q3 = st.columns(3)
        q1.metric("Rangkorrelation", f"{r['spearman']:.2f}", delta="Baumzahl gegen Verbundenheit", delta_color="off")
        q2.metric("Baumzahl (Zehnerpotenz)", f"{r['tau_min']:.1f} bis {r['tau_max']:.1f}", delta="kleinste bis größte", delta_color="off")
        q3.metric("Verbunden", f"{100 * r['rel_min']:.0f} bis {100 * r['rel_max']:.0f} %", delta=f"bei p = {p:g}", delta_color="off")
        st.caption("Mehr Spannbäume gehen meist mit höherer Zuverlässigkeit einher, aber nicht streng: die Zahl der Bäume sagt nichts darüber, wo die dünnen Stellen des Netzes liegen.")
    st.markdown("---")

    st.subheader("📐 Sweeps")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
    metric_opts = {"count": "Zahl der Spannbäume und Brücken", "cost": "Kosten eines Zufallsbaums, Wahrscheinlichkeit des MST", "steps": "Schritte der Würfelverfahren", "reliability": "Zuverlässigkeit", "bias": "Abweichung der Verfahren"}
    if ss.get("sweep_metric") not in metric_opts:
        ss.pop("sweep_metric", None)
    metric = st.radio("Kennzahl", options=list(metric_opts), format_func=lambda v: metric_opts[v], key="sweep_metric", horizontal=True)
    if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
        ss["sweep_done"] = ss.get("sweep_done", set()) | {(sweep_param, base)}
    if (sweep_param, base) in ss.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
            rows_w = _sweep(sweep_param, base)
        series = {
            "count": ([("log10_tau", "Spannbäume (Zehnerpotenz)", "#2F6B65"), ("bridge_share", "Brücken (% der Kanten)", "#1f4e9c")], "Zehnerpotenz bzw. Prozent"),
            "cost": ([("cost_ratio", "erwartete Baumlänge / MST", "#e8a13a")], "Verhältnis"),
            "steps": ([("wilson_steps", "Wilson", "#2F6B65"), ("aldous_steps", "Aldous-Broder", "#4c78a8")], "Schritte je Baum"),
            "reliability": ([("survive", "Netz bleibt verbunden", "#2F6B65"), ("tree_survive", "fester Baum überlebt", "#d62728")], "Wahrscheinlichkeit"),
            "bias": ([("wilson_dev", "Wilson", "#2F6B65"), ("kruskal_dev", "Kruskal mit Zufallsordnung", "#d62728")], "größte Abweichung der Kantenhäufigkeit (2000 Bäume)"),
        }[metric]
        st.plotly_chart(build_sweep(rows_w, SWEEP_LABELS[sweep_param], series[0], series[1], tick=SWEEP_TICKS.get(sweep_param), log_y=metric == "steps"), width="stretch", key="sweep_chart")
        st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil. Die übrigen Regler stehen wie in der Seitenleiste.")
    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein zufälliger Spannbaum ist ein billiger Baum** | Nein: die Gleichverteilung kennt keine Kosten. MST-Kanten liegen im Mittel in 24.9 % der Bäume, alle übrigen in 25.9 %; ein Zufallsbaum ist 1.88-mal so lang wie der MST (20 Filialen, k = 6), im vollständigen Graphen 3.37-mal. Erst die Temperatur drückt das Verhältnis: 1.10 bei b = 4, 1.03 bei b = 8. Der MST selbst kommt (Median) bei b = 4 mit 0.03 %, bei b = 8 mit 2.7 % und bei b = 16 mit 24.8 % heraus. | Gibbs-Verteilungen über Bäume |
| **Wilson braucht immer etwa gleich viele Schritte** | Nur bei b = 0: dort 47 Zufallsschritte je Baum (Aldous-Broder 100, also gut das Doppelte). Kalt explodieren beide: bei b = 4 882 gegen 5 734, bei b = 8 gelingen Wilson 100 Bäume nur auf 2 der 5 Instanzen (im besten Fall 9 058 Schritte je Baum), bei b = 16 in einer Million Schritten auf keiner einzigen. Vermutlich, weil der Zufallsweg fast nur auf billigen Kanten bleibt und die teuren, die zum Baum führen, selten nimmt (Ursache nicht isoliert). | Schnellere Verfahren (Schild 2018, nicht gebaut) |
| **Kruskal mit Zufallsordnung ist "gut genug"** | Nein: auf dem Lehrbuch-Graphen ist es mit Chi-Quadrat 189.9 (kritisch 45.4) klar nicht gleichverteilt, seine exakte Abweichung beträgt 4.1 %, die von Wilson (15.1) und Aldous-Broder (17.5) liegt im Rauschen. Auf der Karte weicht die größte Kantenhäufigkeit um 0.045 ab (Wilson 0.024 = Stichprobenrauschen). Mit Temperatur (Kanten nach exponentiellen Uhren mit Rate exp(−β · Länge) geordnet) wird es billiger, aber zu billig: im Lehrbuch-Graphen bei b = 2 im Mittel 16.41 gegen den exakten Erwartungswert 16.81. | - |
| **Viele Spannbäume heißt zuverlässig** | Meist, aber nicht streng: die Rangkorrelation zwischen Baumzahl und Verbundenheit beträgt über 50 Instanzen 0.71 (k = 4, p = 0.3), 0.72 (k = 3, p = 0.2) und 0.84 (k = 6, p = 0.5). Die Baumzahl sagt nicht, wo die dünnen Stellen liegen: bei k = 3 hat 40 % der Instanzen mindestens eine Brücke (bei k = 4 8 %, ab k = 6 keine). | Netzausbau, Redundanz |
| **Ein fester Baum genügt** | Nein: er überlebt nur mit (1 − p)^(n−1), bei 20 Filialen und p = 0.2 in 1.15 %, bei p = 0.3 in 0.08 % der Fälle. Das Netz bleibt dabei (Median über 50 Instanzen) bei k = 6 in 99.9 % bzw. 99.3 % der Fälle verbunden, bei k = 4 in 97.6 % bzw. 87.9 %, bei k = 3 nur in 77.6 % bzw. 49.6 %. | Redundante Trassen |
| **Die Markov-Schranke sagt etwas** | Hier nicht: die erwartete Zahl überlebender Bäume τ · (1 − p)^(n−1) liegt weit über 1 (6.6 x 10^14 bei k = 6, p = 0.1; noch 8.7 x 10^4 bei k = 3, p = 0.3), obwohl das Netz bei k = 3, p = 0.3 nur in 44 % der Fälle verbunden bleibt. Die Schranke wäre nur unter 1 informativ. | - |
| **Die Verbundenheitswahrscheinlichkeit ist so leicht zu berechnen wie die Baumzahl** | Nein: die Zahl der Spannbäume ist eine Determinante (exakt, Bareiss, auch bei 60 Filialen sofort), die Wahrscheinlichkeit, verbunden zu bleiben, ist #P-schwer (Provan und Ball 1983); hier Monte-Carlo mit 1000 Läufen (±2-3 %), exakt nur auf dem Lehrbuch-Graphen. Das Näherungsschema von Karger (2001) ist nicht gebaut. | Karger 2001 (nicht gebaut) |
| **Zufallsschritte sind Laufzeit** | Nein: gezählt werden Schritte des Zufallswegs (bzw. betrachtete Kanten), kein Sekundenmaß; Wilson ist nur in der Grundform gebaut, das Kantengewicht exp(−β · Länge) ist eine Modellwahl, keine Physik des Netzes. | - |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Matrix-Baum-Satz (Kirchhoff 1847).** Für die Laplace-Matrix $L = D - A$ eines zusammenhängenden Graphen ist die Zahl der Spannbäume $\tau(G) = \det L^{(r)}$ (eine Zeile und Spalte $r$ gestrichen). Mit Kantengewichten $w_e$:
$\det L_w^{(r)} = \sum_T \prod_{e \in T} w_e = Z$. Für den vollständigen Graphen $K_n$ ist $\tau = n^{n-2}$ (Cayley).

**Einschlusswahrscheinlichkeit.** Für einen nach $\prod w_e$ gewürfelten Baum gilt $P(e \in T) = w_e \, R_{\mathrm{eff}}(e)$ mit dem effektiven Widerstand $R_{\mathrm{eff}}$ (Pseudoinverse von $L_w$); $\sum_e P(e \in T) = n - 1$ (Foster).

**Temperatur.** Mit $w_e = e^{-\beta c_e}$ wird $\prod_{e \in T} w_e = e^{-\beta c(T)}$: $P(T) = e^{-\beta c(T)} / Z(\beta)$. Erwartete Baumlänge $\mathbb{E}[c(T)] = \sum_e c_e P(e \in T) = -\partial_\beta \ln Z$; $\beta = 0$ gleichverteilt, $\beta \to \infty$ der MST.

**Wilson.** Zufallsweg mit Übergängen $\propto w_e$, Schleifen löschen, schleifenfreien Weg anhängen: jeder Baum $T$ mit Wahrscheinlichkeit $\prod w_e / Z$ (Wilson 1996). Erwartete Zahl der Schritte: mittlere Trefferzeit; Aldous-Broder: Überdeckungszeit.

**Zuverlässigkeit.** Fällt jede Kante unabhängig mit Wahrscheinlichkeit $p$ aus, bleibt das Netz verbunden genau dann, wenn ein Spannbaum überlebt; $\mathbb{E}[\#\text{überlebende Bäume}] = \tau (1-p)^{n-1}$. Die exakte Berechnung ist #P-schwer (Provan und Ball 1983).

**Literatur.** Kirchhoff, G. (1847). *Ueber die Auflösung der Gleichungen, auf welche man bei der Untersuchung der linearen Vertheilung galvanischer Ströme geführt wird.* Annalen der Physik und Chemie 72, 497-508. Wilson, D. B. (1996).
*Generating random spanning trees more quickly than the cover time.* STOC 1996. Broder, A. (1989). *Generating random spanning trees.* FOCS 1989, 442-447. Aldous, D. J. (1990). *The random walk construction of uniform spanning trees and uniform labelled trees.*
SIAM Journal on Discrete Mathematics 3(4), 450-465. Provan, J. S., & Ball, M. O. (1983). *The complexity of counting cuts and of computing the probability that a graph is connected.* SIAM Journal on Computing 12(4), 777-788.
Karger, D. R. (2001). *A randomized fully polynomial time approximation scheme for the all-terminal network reliability problem.* SIAM Review 43(3), 499-522 (nur genannt, nicht gebaut).

Implementiert in `rst_algorithm.py` (Zählen, Würfeln, Zuverlässigkeit), `rst_scenario.py` (Instanzen), `rst_evaluation.py` (Kennzahlen, Experimente, Sweeps).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
