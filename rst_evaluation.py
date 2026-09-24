"""Auswertung: wie viele Spannbäume gibt es, wie gut würfeln die Verfahren, wie teuer ist ein Zufallsbaum und wie zuverlässig ist das Netz?

- **Anzahl** (`log10_tau`): Zehnerlogarithmus der Zahl der Spannbäume (Matrix-Baum-Satz).
- **Einschluss** (`incl`): Wahrscheinlichkeit je Kante, in einem gewürfelten Baum zu liegen (bei Temperatur b: Gewichte exp(-beta * c), beta = b / mittlere MST-Kantenlänge).
- **Kostenverhältnis** (`cost_ratio`): erwartete Baumlänge eines gewürfelten Baums / Länge des MST (exakt aus den Einschlusswahrscheinlichkeiten).
- **Abweichung** (`max_dev`): größte Abweichung der Kantenhäufigkeit einer Stichprobe von der exakten Einschlusswahrscheinlichkeit.
- **Zuverlässigkeit** (`survive`): Wahrscheinlichkeit, dass das Netz bei Kantenausfall p verbunden bleibt (Monte-Carlo); ein FESTER Baum überlebt mit (1-p)^(n-1).
Alles deterministisch (feste Seeds, plattformstabiler Zufallsstrom); Kennzahlen über 5 feste Instanzen (Seeds 100000-100004) bzw. 50 Instanzen (200000-200049)."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import rst_algorithm as A
import rst_constants as C
import rst_scenario as S

INF = math.inf
DEV_SAMPLES = 2000                    # Stichprobe für die Abweichung der Kantenhäufigkeiten in run_config (mit 100 Bäumen wäre sie reines Rauschen)
BUDGET = 2_000_000                     # Zufallsweg-Schritte je Stichprobe (etwa 2 Sekunden): bei niedriger Temperatur laufen Wilson und Aldous-Broder sehr lange


@dataclass(frozen=True)
class Settings:
    kind: str = "depot"
    n: int = C.DEFAULT_N
    k: int = C.DEFAULT_K
    terrain: float = C.DEFAULT_TERRAIN
    seed: int = C.DEFAULT_SEED
    b: float = C.DEFAULT_B
    sampler: str = C.DEFAULT_SAMPLER
    p: float = C.DEFAULT_P
    samples: int = C.DEFAULT_SAMPLES


@lru_cache(maxsize=256)
def instance_of(settings):
    if settings.kind == "textbook":
        return S.textbook_instance()
    return S.generate(settings.n, settings.k, settings.terrain, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    edges: list
    costs: list
    tau: int                           # Zahl der Spannbäume (exakt)
    mst: tuple                         # Kantennummern des billigsten Baums
    mst_cost: float
    cbar: float                        # mittlere Kantenlänge des MST (Skala der Temperatur)
    beta: float
    weights: list
    incl: list                         # Einschlusswahrscheinlichkeit je Kante bei Temperatur b
    incl_uniform: list                 # bei b = 0 (gleichverteilt)
    exp_cost: float                    # erwartete Baumlänge bei Temperatur b
    p_mst: float                       # Wahrscheinlichkeit des MST bei Temperatur b

    @property
    def n(self):
        return self.inst.n

    @property
    def m(self):
        return len(self.edges)

    @property
    def log10_tau(self):
        return math.log10(self.tau) if self.tau > 0 else -INF

    @property
    def cost_ratio(self):
        return self.exp_cost / self.mst_cost

    @property
    def bridges(self):
        return [e for e, q in enumerate(self.incl_uniform) if q > 1.0 - 1e-9]

    def mst_incl_mean(self):
        return float(np.mean([self.incl_uniform[e] for e in self.mst]))

    def other_incl_mean(self):
        rest = [q for e, q in enumerate(self.incl_uniform) if e not in set(self.mst)]
        return float(np.mean(rest)) if rest else float("nan")


def analyse(settings):
    inst = instance_of(settings)
    edges = list(inst.edges)
    costs = [e[2] for e in edges]
    mst = tuple(sorted(A.mst_edges(inst.n, edges, costs)))
    cbar = sum(costs[e] for e in mst) / len(mst)
    beta = settings.b / cbar
    weights = A.edge_weights(costs, beta)
    incl = A.inclusion_probabilities(inst.n, edges, weights)
    incl0 = incl if settings.b == 0 else A.inclusion_probabilities(inst.n, edges)
    exp_cost = float(sum(c * q for c, q in zip(costs, incl)))
    return Analysis(settings, inst, edges, costs, A.count_trees(inst.n, edges), mst, sum(costs[e] for e in mst), cbar, beta, weights, incl, incl0, exp_cost, A.p_mst(inst.n, edges, costs, beta))


def _median(values):
    values = [v for v in values if v is not None and v == v and v != INF]
    return float(np.median(values)) if values else float("nan")


# --- Würfeln --------------------------------------------------------------------------------------------------------------------------------------


def draw(a, sampler=None, samples=None, tag="draw", budget=BUDGET):
    """Bis zu `samples` Bäume mit dem Verfahren `sampler` bei der Temperatur der Analyse (Schrittbudget `budget`); gibt (Bäume, Gesamtschritte); weniger Bäume als verlangt = Budget aufgebraucht."""
    s = a.settings
    return A.sample_trees(sampler or s.sampler, a.n, a.edges, a.weights, samples or s.samples, f"{s.seed}-{s.kind}-{tag}", budget=budget)


def sample_stats(a, sampler=None, samples=None, budget=BUDGET):
    """Kostenverteilung, Kantenhäufigkeiten gegen die exakte Einschlusswahrscheinlichkeit, Blätter, Grad, Anteil MST, Schritte je Baum; `truncated`: das Schrittbudget hat die Stichprobe verkürzt, `n_samples` = 0:
    kein einziger Baum fertig geworden (alle Kennzahlen dann None)."""
    wanted = samples or a.settings.samples
    trees, steps = draw(a, sampler, samples, budget=budget)
    n = len(trees)
    out = {"n_samples": n, "wanted": wanted, "truncated": n < wanted, "steps": steps}
    if n == 0:
        out.update({"costs": [], "mean_cost": None, "sd_cost": None, "min_cost": None, "freq": [0.0] * a.m, "max_dev": None, "leaf_share": None, "max_degree": None, "mst_share": None, "steps_per_tree": None, "ratio": None})
        return out
    costs = [sum(a.costs[e] for e in t) for t in trees]
    freq = [0] * a.m
    for t in trees:
        for e in t:
            freq[e] += 1
    freq = [f / n for f in freq]
    leaves, maxdeg = zip(*[A.tree_stats(a.n, a.edges, t) for t in trees])
    out.update({"costs": costs, "mean_cost": float(np.mean(costs)), "sd_cost": float(np.std(costs, ddof=1)) if n > 1 else 0.0, "min_cost": min(costs), "freq": freq,
                "max_dev": max(abs(f - q) for f, q in zip(freq, a.incl)), "leaf_share": float(np.mean(leaves)) / a.n, "max_degree": float(np.mean(maxdeg)),
                "mst_share": sum(tuple(t) == a.mst for t in trees) / n, "steps_per_tree": steps / n, "ratio": float(np.mean(costs)) / a.mst_cost})
    return out


def sampler_compare(a, samples=200):
    """Die drei Würfelverfahren auf derselben Instanz: Schritte je Baum und größte Abweichung der Kantenhäufigkeit von der exakten Einschlusswahrscheinlichkeit."""
    out = {}
    for kind in C.SAMPLERS:
        st = sample_stats(a, kind, samples, budget=BUDGET // 2)
        out[kind] = {"steps": st["steps_per_tree"], "max_dev": st["max_dev"], "ratio": st["ratio"], "n_samples": st["n_samples"]}
    return out


@lru_cache(maxsize=32)
def uniformity(name="textbook", samples=20000):
    """Gleichverteilungstest auf einem kleinen Graphen (alle Bäume aufgezählt): je Verfahren Chi-Quadrat, kritischer Wert (99.9 %) und Abstand (TV) zur Gleichverteilung."""
    graphs = {"textbook": (5, list(S.textbook_instance().edges)), "k4": (4, [(u, v, 1.0) for u in range(4) for v in range(u + 1, 4)])}
    n, edges = graphs[name]
    probs = A.exact_tree_probs(n, edges, [1.0] * len(edges))
    out = {"trees": len(probs), "crit": A.chi2_crit(len(probs) - 1)}
    for kind in C.SAMPLERS:
        trees, _ = A.sample_trees(kind, n, edges, None, samples, f"uniformity-{name}")
        cnt = {}
        for t in trees:
            cnt[t] = cnt.get(t, 0) + 1
        out[kind] = {"chi2": A.chi2_stat(cnt, probs), "tv": A.tv_distance(cnt, probs)}
    exact_rk = A.random_kruskal_distribution(n, edges)
    out["kruskal_exact_tv"] = 0.5 * sum(abs(exact_rk.get(t, 0.0) - q) for t, q in probs.items())
    return out


# --- Zuverlässigkeit ------------------------------------------------------------------------------------------------------------------------------


def survive(a, p, trials=2000):
    """Wahrscheinlichkeit, dass das Netz bei Kantenausfall p verbunden bleibt (Monte-Carlo): (Schätzwert, Standardfehler)."""
    s = a.settings
    return A.survival_mc(a.n, a.edges, p, trials, f"{s.seed}-{s.kind}")


def reliability_table(a, ps=C.P_OPTIONS, trials=2000):
    """Je Ausfallwahrscheinlichkeit: Netz (Monte-Carlo), fester MST (1-p)^(n-1), erwartete Zahl überlebender Bäume tau (1-p)^(n-1)."""
    rows = []
    for p in ps:
        est, se = survive(a, p, trials)
        rows.append({"p": p, "network": est, "se": se, "tree": (1.0 - p) ** (a.n - 1), "expected_trees": A.expected_surviving_trees(a.n, a.edges, p)})
    return rows


def fail_patterns(a, p, runs=C.FAIL_RUNS):
    """`runs` feste Ausfallmuster (für die Wiedergabe): Liste (ausgefallene Kanten, Komponente je Knoten, verbunden?)."""
    rng = A.make_rng(a.settings.seed, a.settings.kind, "fail", p)
    return [A.failure_pattern(a.n, a.edges, p, rng) for _ in range(runs)]


def correlation(base, p=0.1, seeds=C.FEAS_SEEDS, trials=1000):
    """Rangkorrelation (Spearman) zwischen log10 der Baumzahl und der Verbundenheitswahrscheinlichkeit bei Ausfall p über `seeds` Instanzen mit den Einstellungen von `base`."""
    an = [analyse(replace(base, seed=s, b=0.0)) for s in seeds]
    rel = [survive(a, p, trials)[0] for a in an]
    logt = [a.log10_tau for a in an]
    return {"n_runs": len(an), "spearman": A.spearman(logt, rel), "tau_min": min(logt), "tau_max": max(logt), "rel_min": min(rel), "rel_max": max(rel)}


# --- Kennzahlen über feste Instanzen ------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if v is not None and v == v and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        a = analyse(replace(s0, seed=seed))
        cmp = sampler_compare(a, 100)
        dev = {kind: sample_stats(a, kind, DEV_SAMPLES, budget=BUDGET // 4) for kind in ("wilson", "kruskal")}
        est, _se = survive(a, s0.p, 1000)
        rows.append({"m": a.m, "log10_tau": a.log10_tau, "cost_ratio": a.cost_ratio, "p_mst": a.p_mst, "mst_incl": a.mst_incl_mean(), "other_incl": a.other_incl_mean(), "bridge_share": 100.0 * len(a.bridges) / a.m,
                     "wilson_steps": cmp["wilson"]["steps"], "aldous_steps": cmp["aldous"]["steps"], "steps_ratio": (cmp["wilson"]["steps"] / cmp["aldous"]["steps"]) if cmp["wilson"]["steps"] and cmp["aldous"]["steps"] else None,
                     "kruskal_dev": dev["kruskal"]["max_dev"] if dev["kruskal"]["n_samples"] >= DEV_SAMPLES // 4 else None,
                     "wilson_dev": dev["wilson"]["max_dev"] if dev["wilson"]["n_samples"] >= DEV_SAMPLES // 4 else None, "survive": est, "tree_survive": (1.0 - s0.p) ** (a.n - 1), "expected_trees": A.expected_surviving_trees(a.n, a.edges, s0.p)})
    out = {"n_runs": len(rows)}
    for key in rows[0]:
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats([r[key] for r in rows])
    return out


SWEEP_VALUES = {"n": C.N_SWEEP, "k": (3, 4, 6, 10, 20, 1000), "terrain": (0.0, 0.2, 0.4, 0.6, 1.0), "b": C.B_OPTIONS, "p": C.P_OPTIONS}
SWEEP_LABELS = {"n": "Filialen n", "k": "Nachbarn k (1000 = vollständig)", "terrain": "Geländezuschlag", "b": "Temperatur b (0 = gleichverteilt)", "p": "Kantenausfall p"}
SWEEP_TICKS = {}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def count_curve(base, ns=None):
    """log10 der Baumzahl über n (5 feste Instanzen, Median) für die Einstellungen von `base`, dazu der vollständige Graph n^(n-2)."""
    ns = ns if ns is not None else C.N_SWEEP
    rows = []
    for n in ns:
        an = [analyse(replace(base, n=n, seed=s)) for s in C.SWEEP_SEEDS]
        rows.append({"n": n + 1, "log10_tau": _median([a.log10_tau for a in an]), "log10_cayley": (n - 1) * math.log10(n + 1)})
    return rows


def bridge_rate(base, seeds=C.FEAS_SEEDS):
    """Über `seeds` Instanzen (Einstellungen von `base`): Anteil der Instanzen mit mindestens einer Brücke und mittlere Zahl der Brücken."""
    counts = [len(analyse(replace(base, seed=s, b=0.0)).bridges) for s in seeds]
    return {"n_runs": len(counts), "share": sum(c > 0 for c in counts) / len(counts), "mean": sum(counts) / len(counts), "max": max(counts)}


def reliability_curve(base, ps=(0.2, 0.3, 0.5), seeds=C.FEAS_SEEDS, trials=1000):
    """Median über `seeds` Instanzen (Einstellungen von `base`) der Verbundenheitswahrscheinlichkeit je Ausfallwahrscheinlichkeit (Monte-Carlo), dazu der kleinste Wert und der feste Baum (1-p)^(n-1)."""
    an = [analyse(replace(base, seed=s, b=0.0)) for s in seeds]
    out = {}
    for p in ps:
        vals = [survive(a, p, trials)[0] for a in an]
        out[p] = {"median": float(np.median(vals)), "min": min(vals), "tree": (1.0 - p) ** (an[0].n - 1)}
    return out
