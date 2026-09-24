"""Plotly-Abbildungen: Karte mit Einschlusswahrscheinlichkeiten, Wilson Weg für Weg, Kostenverteilung, Häufigkeit gegen Wahrscheinlichkeit, Gleichverteilungstest, Zuverlässigkeit, Ausfallmuster, Baumzahl, Sweeps.
Karten ohne feste Achsenbereiche (Plotly friert sie beim ersten Zeichnen ein); Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import math

import plotly.graph_objects as go

TREE_COLOR = "#2F6B65"
DEPOT_COLOR = "#2ca02c"
NODE_COLOR = "#4c78a8"
CAND_COLOR = "rgba(150,150,150,0.28)"
RED, ORANGE, BLUE = "#d62728", "#ff7f0e", "#1f4e9c"
PALETTE = ["#4c78a8", "#e8a13a", "#7b3fbf", "#d95f9b", "#2ca02c", "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
BINS = ((0.10, "unter 10 %", "rgba(160,160,160,0.55)", 1.6), (0.20, "10 bis 20 %", "#9ecae1", 2.4), (0.35, "20 bis 35 %", "#4c9bd0", 3.2), (0.60, "35 bis 60 %", "#e8a13a", 4.0),
        (0.999, "60 % und mehr", "#d95f0e", 4.8))
METHOD_COLORS = {"wilson": "#2F6B65", "aldous": "#4c78a8", "kruskal": "#d62728"}
METHOD_LABELS = {"wilson": "Wilson", "aldous": "Aldous-Broder", "kruskal": "Kruskal mit Zufallsordnung"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_axes(fig, height=480):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed")
    return _base(fig, height)


def _height(inst):
    return 340 if inst.kind == "textbook" else 480


def _lines(fig, inst, pairs, color, width=2.6, dash="solid", name="", showlegend=False):
    pairs = list(pairs)
    if not pairs:
        return
    xs, ys = [], []
    for u, v in pairs:
        xs += [inst.xy[u][0], inst.xy[v][0], None]
        ys += [inst.xy[u][1], inst.xy[v][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend))


def _nodes(fig, inst, colors=None):
    n = inst.n
    label = inst.labels
    cols = colors if colors is not None else [NODE_COLOR] * n
    rest = [v for v in range(n) if v != inst.depot]
    text = [label[v] for v in rest] if label is not None else None
    fig.add_trace(go.Scatter(x=[inst.xy[v][0] for v in rest], y=[inst.xy[v][1] for v in rest], mode="markers+text" if text else "markers", text=text, textposition="top center",
                             marker=dict(size=9 if inst.kind == "depot" else 15, color=[cols[v] for v in rest], line=dict(width=1, color="white")), name="Filiale", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[inst.xy[inst.depot][0]], y=[inst.xy[inst.depot][1]], mode="markers+text" if label is not None else "markers", text=[label[inst.depot]] if label is not None else None,
                             textposition="top center", marker=dict(size=16, symbol="star", color=cols[inst.depot] if colors is not None else DEPOT_COLOR, line=dict(width=1, color="white")),
                             name="Depot", hoverinfo="skip", showlegend=True))


def _edge_labels(fig, inst, texts):
    """Kantenbeschriftungen als Annotationen mit Hinterlegung (nur Lehrbuchbeispiel): {(u, v): Text}."""
    for (u, v), t in texts.items():
        fig.add_annotation(x=(inst.xy[u][0] + inst.xy[v][0]) / 2, y=(inst.xy[u][1] + inst.xy[v][1]) / 2, text=t, showarrow=False, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11))


def build_inclusion_map(inst, edges, incl, mst=None):
    """Karte: Kanten nach der Wahrscheinlichkeit gefärbt und dicker gezeichnet, in einem zufälligen Spannbaum zu liegen (Kirchhoff); Brücken (Wahrscheinlichkeit 1) blau gestrichelt; der MST (falls angegeben) als schwarze
    dünne Linie darüber."""
    fig = go.Figure()
    lo = 0.0
    for hi, name, color, width in BINS:
        sel = [e for e, q in enumerate(incl) if lo <= q < hi]
        _lines(fig, inst, [tuple(edges[e][:2]) for e in sel], color, width, name=f"Kante in {name} der Bäume", showlegend=bool(sel))
        lo = hi
    br = [e for e, q in enumerate(incl) if q >= 0.999]
    _lines(fig, inst, [tuple(edges[e][:2]) for e in br], BLUE, 5.0, "dash", "Brücke (in jedem Baum)", bool(br))
    if mst is not None:
        _lines(fig, inst, [tuple(edges[e][:2]) for e in mst], "rgba(0,0,0,0.85)", 1.0, "solid", "billigster Baum (MST)", True)
    _nodes(fig, inst)
    if inst.kind == "textbook":
        _edge_labels(fig, inst, {tuple(edges[e][:2]): f"{incl[e]:.2f}" for e in range(len(edges))})
    return _map_axes(fig, _height(inst))


def build_wilson_step(inst, edges, walks, i):
    """Wilson nach `i` Wegen: der bisherige Baum grün, der laufende Zufallsweg (mit Schleifen) orange gepunktet, sein schleifenfreier Weg orange dick, die Startkante rot; Weg 0 = nur die Wurzel."""
    fig = go.Figure()

    def pairs(seq):
        return [(min(a, b), max(a, b)) for a, b in zip(seq, seq[1:])]

    done = set()
    for w in walks[:max(0, i - 1) if i > 0 else 0]:
        done.update(pairs(w["path"]))
    cur = walks[i - 1] if i > 0 else None
    _lines(fig, inst, [tuple(e[:2]) for e in edges], CAND_COLOR, 0.9)
    _lines(fig, inst, sorted(done), TREE_COLOR, 3.6, name="Baum bisher", showlegend=bool(done))
    if cur is not None:
        _lines(fig, inst, pairs(cur["walk"]), "rgba(255,127,14,0.55)", 2.0, "dot", "Zufallsweg (mit Schleifen)", True)
        _lines(fig, inst, pairs(cur["path"]), ORANGE, 5.0, name="schleifenfreier Weg wird angehängt", showlegend=True)
    cols = [NODE_COLOR] * inst.n
    if cur is not None:
        cols[cur["start"]] = RED
    _nodes(fig, inst, cols)
    return _map_axes(fig, _height(inst))


def build_cost_hist(costs, mst_cost, exp_cost, mean_cost):
    """Kostenverteilung der gewürfelten Bäume mit Linien für den MST (grün), den exakten Erwartungswert (orange gestrichelt) und den Stichprobenmittelwert (schwarz gepunktet)."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=costs, nbinsx=30, marker_color="rgba(76,120,168,0.65)", name="gewürfelte Bäume"))
    for x, name, color, dash in ((mst_cost, "billigster Baum (MST)", TREE_COLOR, "solid"), (exp_cost, "Erwartungswert (exakt)", ORANGE, "dash"), (mean_cost, "Mittel der Stichprobe", "#222222", "dot")):
        fig.add_trace(go.Scatter(x=[x, x], y=[0, 1], yaxis="y2", mode="lines", line=dict(color=color, width=3, dash=dash), name=name))
    fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 1], visible=False), barmode="overlay")
    fig.update_xaxes(title_text="Baumlänge")
    fig.update_yaxes(title_text="Bäume")
    return _base(fig, 340, legend_y=-0.35)


def build_freq_scatter(incl, freq, mst=None):
    """Häufigkeit einer Kante in der Stichprobe gegen ihre exakte Einschlusswahrscheinlichkeit; auf der Diagonalen: das Verfahren würfelt richtig; Kanten des MST orange."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999999", dash="dash"), name="Diagonale", hoverinfo="skip"))
    mset = set(mst) if mst is not None else set()
    rest = [e for e in range(len(incl)) if e not in mset]
    fig.add_trace(go.Scatter(x=[incl[e] for e in rest], y=[freq[e] for e in rest], mode="markers", marker=dict(size=7, color="rgba(76,120,168,0.7)"), name="Kante"))
    if mset:
        fig.add_trace(go.Scatter(x=[incl[e] for e in sorted(mset)], y=[freq[e] for e in sorted(mset)], mode="markers", marker=dict(size=8, color=ORANGE), name="Kante des MST"))
    fig.update_xaxes(title_text="exakte Wahrscheinlichkeit", range=[0, 1.02])
    fig.update_yaxes(title_text="Häufigkeit in der Stichprobe", range=[0, 1.02])
    return _base(fig, 340, legend_y=-0.3)


def build_uniformity(u):
    """Chi-Quadrat je Würfelverfahren auf einem kleinen Graphen gegen den kritischen Wert (99.9 %): Balken über der Linie = nicht gleichverteilt."""
    kinds = ["wilson", "aldous", "kruskal"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[METHOD_LABELS[k] for k in kinds], y=[u[k]["chi2"] for k in kinds], marker_color=[METHOD_COLORS[k] for k in kinds], text=[f"{u[k]['chi2']:.0f}" for k in kinds], textposition="outside", name="Chi-Quadrat"))
    fig.add_hline(y=u["crit"], line=dict(color="#222222", dash="dash"), annotation_text="kritischer Wert (99.9 %)", annotation_position="top left")
    fig.update_yaxes(title_text="Chi-Quadrat")
    return _base(fig, 320)


def build_reliability(rows):
    """Wahrscheinlichkeit, verbunden zu bleiben, über die Kantenausfallquote p: das Netz (Monte-Carlo, Band = 2 Standardfehler) gegen einen FESTEN Baum (1-p)^(n-1)."""
    xs = [f"{r['p']:g}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["network"] for r in rows], mode="lines+markers", line=dict(color=TREE_COLOR, width=3), name="Netz bleibt verbunden (ein Baum überlebt)",
                             error_y=dict(type="data", array=[2 * r["se"] for r in rows], visible=True)))
    fig.add_trace(go.Scatter(x=xs, y=[r["tree"] for r in rows], mode="lines+markers", line=dict(color=RED, width=2.6, dash="dash"), name="ein fester Baum überlebt"))
    fig.update_xaxes(title_text="Ausfallwahrscheinlichkeit je Kante p", type="category")
    fig.update_yaxes(title_text="Wahrscheinlichkeit", range=[0, 1.02])
    return _base(fig, 340, legend_y=-0.35)


def build_failure_run(inst, edges, pattern):
    """Ein Ausfallmuster: ausgefallene Kanten rot gestrichelt, überlebende grau, Knoten nach Komponente gefärbt (die größte Komponente grün)."""
    failed, comp, ok = pattern
    fs = set(failed)
    fig = go.Figure()
    _lines(fig, inst, [tuple(edges[e][:2]) for e in range(len(edges)) if e not in fs], "rgba(90,90,90,0.55)", 1.6, name="überlebt", showlegend=True)
    _lines(fig, inst, [tuple(edges[e][:2]) for e in failed], RED, 2.6, "dash", "ausgefallen", bool(failed))
    sizes = {}
    for c in comp:
        sizes[c] = sizes.get(c, 0) + 1
    order = sorted(sizes, key=lambda c: (-sizes[c], c))
    color_of = {c: (TREE_COLOR if i == 0 else PALETTE[(i - 1) % len(PALETTE)]) for i, c in enumerate(order)}
    _nodes(fig, inst, [color_of[c] for c in comp])
    return _map_axes(fig, _height(inst))


def build_count_curve(rows):
    """Zehnerlogarithmus der Baumzahl über die Knotenzahl (Median über 5 feste Instanzen) gegen den vollständigen Graphen (Cayley n^(n-2))."""
    xs = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["log10_cayley"] for r in rows], mode="lines+markers", line=dict(color="#8c8c8c", width=2.4, dash="dash"), name="vollständiger Graph (Cayley)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["log10_tau"] for r in rows], mode="lines+markers", line=dict(color=TREE_COLOR, width=3), name="dieser Kandidatengraph"))
    fig.update_xaxes(title_text="Knoten (Depot + Filialen)", type="category")
    fig.update_yaxes(title_text="Zahl der Spannbäume (Zehnerpotenz)")
    return _base(fig, 320, legend_y=-0.3)


def build_sweep(rows, param_label, series, y_label, tick=None, log_y=False):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [tick(r["value"]) if tick else str(r["value"]) for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [None if r[key] != r[key] else r[key] for r in rows]
        lo = [None if r.get(f"{key}_lo", r[key]) != r.get(f"{key}_lo", r[key]) else r.get(f"{key}_lo", r[key]) for r in rows]
        hi = [None if r.get(f"{key}_hi", r[key]) != r.get(f"{key}_hi", r[key]) else r.get(f"{key}_hi", r[key]) for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name, connectgaps=False))
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    return _base(fig, 360, legend_y=-0.3)


def fmt_pow10(x):
    """Zehnerpotenz als Text: 15.3 -> "2.0 x 10^15"."""
    e = math.floor(x)
    return f"{10 ** (x - e):.1f} x 10^{e}"
