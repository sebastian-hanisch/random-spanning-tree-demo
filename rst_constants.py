"""Konstanten der Demo zufällige Spannbäume und Kirchhoff: Instanz-Geometrie, Regler, gemessene Werte, Presets."""
AREA = 100.0                     # Kantenlänge des Gebiets in km
DEPOT_XY = (15.0, 50.0)          # Lage des Depots (Werk) am linken Rand, Filialen zufällig im Gebiet
N_MIN, N_MAX, DEFAULT_N = 6, 60, 20                             # Filialen (ohne Depot)
K_OPTIONS = (3, 4, 5, 6, 8, 10, 15, 20, 1000)                   # nächste Nachbarn je Knoten; 1000 = vollständiger Graph
DEFAULT_K = 6
TERRAIN_OPTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
DEFAULT_TERRAIN = 0.3
SEED_MAX = 999999
DEFAULT_SEED = 35
KINDS = ("depot", "textbook")
KIND_LABELS = {"depot": "Depot und Filialen (Karte)", "textbook": "Lehrbuchbeispiel (5 Knoten)"}
B_OPTIONS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)                # Temperatur: beta mal mittlere MST-Kantenlänge; 0 = gleichverteilt
DEFAULT_B = 0.0
SAMPLERS = ("wilson", "aldous", "kruskal")
SAMPLER_LABELS = {"wilson": "Wilson (schleifenfreie Zufallswege)", "aldous": "Aldous-Broder (Zufallsweg bis alle besucht)", "kruskal": "Kruskal mit Zufallsordnung (nicht gleichverteilt)"}
DEFAULT_SAMPLER = "wilson"
P_OPTIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)                    # Ausfallwahrscheinlichkeit je Kante
DEFAULT_P = 0.1
SAMPLE_OPTIONS = (200, 500, 1000, 2000)
DEFAULT_SAMPLES = 500
STEPS = {1: "1 · Zählen", 2: "2 · Würfeln", 3: "3 · Verteilung", 4: "4 · Zuverlässigkeit"}
FAIL_RUNS = 20
SWEEP_SEEDS = tuple(range(100000, 100005))
FEAS_SEEDS = tuple(range(200000, 200050))
N_SWEEP = (8, 12, 20, 30, 40)
_BASE = {"kind": "depot", "n": 20, "k": 6, "terrain": 0.3, "seed": 35, "b": 0.0, "sampler": "wilson", "p": 0.1, "samples": 500, "step": 1, "walk_i": 0, "fail_i": 0}
PRESETS = {
    "Standardfall (Voreinstellung)": dict(_BASE),
    "Lehrbuchbeispiel": {**_BASE, "kind": "textbook"},
    "Dünnes Netz (Brücken)": {**_BASE, "k": 3, "seed": 19},
    "Vollständiger Graph (Cayley)": {**_BASE, "n": 9, "k": 1000},
    "Wilson in Aktion": {**_BASE, "n": 12, "step": 2, "walk_i": 1},
    "Kalt: der MST kommt heraus": {**_BASE, "n": 6, "b": 8.0, "step": 3},
    "Kruskal mit Zufallsordnung": {**_BASE, "kind": "textbook", "sampler": "kruskal", "samples": 2000, "step": 3},
    "Hohe Ausfallquote": {**_BASE, "k": 4, "seed": 1, "p": 0.3, "step": 4},
    "Dünn und zerbrechlich": {**_BASE, "k": 3, "seed": 3, "p": 0.2, "step": 4},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "20 Filialen + Depot, k = 6 Nachbarn, Seed 35: 75 Kandidatenkanten und 1.9 x 10^15 Spannbäume (exakt gezählt, ohne einen aufzuzählen). Jede Kante liegt in 20 bis 37 % der Bäume, keine Brücke. Ein gleichverteilter Zufallsbaum ist im Erwartungswert 1.87-mal so lang wie der MST (666.46 gegen 357.32); der MST selbst kommt mit Wahrscheinlichkeit 5.3 x 10^-16 heraus.",
    "Lehrbuchbeispiel": "5 Knoten, 7 Kanten: 21 Spannbäume (Determinante des Laplace-Minors, von Hand nachzurechnen). Jede Kante liegt in 48 bis 62 % der Bäume; ein Zufallsbaum ist im Erwartungswert 20.00 lang gegen 15 für den MST (1.33-fach), der MST kommt mit Wahrscheinlichkeit 1/21 = 4.8 % heraus.",
    "Dünnes Netz (Brücken)": "Nur die 3 nächsten Nachbarn als Kandidaten, Seed 19: 37 Kanten, 12 720 000 Spannbäume und 2 Brücken (Wahrscheinlichkeit 1: sie liegen in jedem Baum). Ein Zufallsbaum ist im Erwartungswert 1.26-mal so lang wie der MST.",
    "Vollständiger Graph (Cayley)": "10 Knoten, vollständig: 10^8 = 100 000 000 Spannbäume (Cayley: n^(n-2)); jede Kante liegt in genau 20 % der Bäume (2/n). Ein Zufallsbaum ist im Erwartungswert 2.28-mal so lang wie der MST (473.06 gegen 207.70).",
    "Wilson in Aktion": "12 Filialen: Weg 1 von 7 braucht 31 Zufallsschritte, davon werden 27 als Schleifen gelöscht, übrig bleibt ein Weg mit 4 Kanten; der ganze Baum kostet 39 Schritte (Länge 506.15 gegen MST 248.04). Im Mittel über 100 Bäume: Wilson 25.4 Schritte je Baum, Aldous-Broder 46.1.",
    "Kalt: der MST kommt heraus": "6 Filialen, Temperatur b = 8: der MST kommt mit Wahrscheinlichkeit 74.8 % heraus (exakt), in der Stichprobe von 500 Bäumen sind es 72.2 %; mittlere Baumlänge 193.99 gegen 190.97 für den MST (Erwartungswert 193.94). Kälter heißt teurer zu würfeln: 321 Zufallsschritte je Baum.",
    "Kruskal mit Zufallsordnung": "Lehrbuch-Graph, 2000 Bäume mit Kruskal und zufällig gemischten Kanten: nicht gleichverteilt. Im Test mit je 20 000 Würfen liegen Wilson (Chi-Quadrat 15.1) und Aldous-Broder (17.5) unter dem kritischen Wert 45.4, Kruskal mit Zufallsordnung (189.9) weit darüber; seine exakte Abweichung von der Gleichverteilung beträgt 4.1 %.",
    "Hohe Ausfallquote": "k = 4, Seed 1, p = 0.3: 51 Kanten, 1.6 x 10^11 Spannbäume; Muster 1: 16 Kanten fallen aus, das Netz zerfällt in 2 Teile. Insgesamt bleibt es in 84.8 % der Fälle verbunden (Monte-Carlo), ein fester Baum überlebt nur mit 0.08 %; die erwartete Zahl überlebender Bäume ist 1.3 x 10^8.",
    "Dünn und zerbrechlich": "k = 3, Seed 3, p = 0.2: 37 Kanten, 2.0 x 10^7 Spannbäume, 1 Brücke; Muster 1 zerfällt (7 Kanten fallen aus, 2 Teile). Verbunden bleibt das Netz nur in 62.0 % der Fälle (ein fester Baum überlebt in 1.15 %), obwohl die erwartete Zahl überlebender Bäume 2.3 x 10^5 beträgt: viele Bäume, aber eine Brücke genügt.",
}
# Beobachtete Spannweite der Kennzahl (MEDIAN über die 5 festen Instanzen Seeds 100000-100004) je Preset, mit Sicherheitsabstand: (Kennzahl, untere, obere Grenze).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": ("log10_tau", 14.5, 17.0),
    "Dünnes Netz (Brücken)": ("log10_tau", 7.5, 8.6),
    "Vollständiger Graph (Cayley)": ("log10_tau", 7.99, 8.01),
    "Wilson in Aktion": ("log10_tau", 9.0, 10.2),
    "Kalt: der MST kommt heraus": ("log10_tau", 3.8, 4.7),
    "Hohe Ausfallquote": ("log10_tau", 10.3, 11.6),
    "Dünn und zerbrechlich": ("log10_tau", 7.5, 8.6),
}
