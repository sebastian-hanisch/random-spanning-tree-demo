"""Zählen und Würfeln statt Optimieren: Spannbäume zählen (Kirchhoff, Matrix-Baum-Satz), gleichverteilt (oder nach Kantengewichten) würfeln (Wilson, Aldous-Broder) und die Zuverlässigkeit eines Netzes.

**Matrix-Baum-Satz** (Kirchhoff 1847): die Zahl der Spannbäume eines zusammenhängenden Graphen ist die Determinante des Laplace-Minors (Laplace-Matrix ohne eine Zeile und Spalte); mit Kantengewichten w_e ist es die
Summe über alle Bäume T des Produkts der Gewichte (`count_trees` exakt ganzzahlig per Bareiss, `log_partition` gewichtet). Die Wahrscheinlichkeit, dass ein zufälliger Baum die Kante e enthält, ist
w_e * R_eff(e) (effektiver Widerstand, `inclusion_probabilities`); ihre Summe ist n - 1 (Satz von Foster), Brücken haben Wahrscheinlichkeit 1.

**Temperatur:** mit w_e = exp(-beta * c_e) wird das Baumgewicht zu exp(-beta * Kosten): eine Gibbs-Verteilung über alle Spannbäume. beta = 0 ist gleichverteilt, beta -> unendlich der billigste Baum (MST).
Die erwartete Baumlänge ist die Summe c_e * p_e (`expected_cost`), die Wahrscheinlichkeit des MST ist exakt bekannt (`p_mst`).

**Würfeln:** `wilson` (schleifenfreie Zufallswege, Wilson 1996), `aldous_broder` (Zufallsweg bis alle besucht sind, erste Besuche zählen) liefern Bäume mit Wahrscheinlichkeit proportional zum Gewichtsprodukt;
`random_kruskal` (Kruskal mit zufälliger Kantenordnung) ist dagegen NICHT gleichverteilt (`random_kruskal_distribution` berechnet die Verteilung exakt).

**Zuverlässigkeit:** jede Kante fällt mit Wahrscheinlichkeit p aus; das Netz bleibt verbunden genau dann, wenn ein Spannbaum überlebt (`survival_exact` über die Zahl verbundener Teilgraphen, `survival_mc`)."""

import math
import random
from bisect import bisect_right
from itertools import combinations, permutations

import numpy as np

from rst_unionfind import UnionFind

Z999 = 3.0902323061678132                     # Quantil der Standardnormalverteilung zu 0.999


def make_rng(*parts):
    """Plattformstabiler Zufallsstrom (Mersenne-Twister, Seed aus dem Text der Teile)."""
    return random.Random(":".join(str(p) for p in parts))


def bareiss_det(mat):
    """Exakte Determinante einer ganzzahligen Matrix (Bareiss, keine Rundung)."""
    m = [row[:] for row in mat]
    n = len(m)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for i in range(n - 1):
        if m[i][i] == 0:
            swap = next((r for r in range(i + 1, n) if m[r][i] != 0), None)
            if swap is None:
                return 0
            m[i], m[swap] = m[swap], m[i]
            sign = -sign
        for r in range(i + 1, n):
            for c in range(i + 1, n):
                m[r][c] = (m[r][c] * m[i][i] - m[r][i] * m[i][c]) // prev
        prev = m[i][i]
    return sign * m[n - 1][n - 1]


# --- Zählen (Kirchhoff) ---------------------------------------------------------------------------------------------------------------------------


def count_trees(n, edges):
    """Zahl der Spannbäume (exakt): Determinante des Laplace-Minors. `edges` = Folge (u, v, ...) ohne Mehrfachkanten; n = 1 -> 1."""
    if n <= 1:
        return 1
    lap = [[0] * n for _ in range(n)]
    for u, v, *_ in edges:
        lap[u][u] += 1
        lap[v][v] += 1
        lap[u][v] -= 1
        lap[v][u] -= 1
    return bareiss_det([row[:-1] for row in lap[:-1]])


def laplacian(n, edges, weights):
    lap = np.zeros((n, n))
    for (u, v, *_), w in zip(edges, weights):
        lap[u, u] += w
        lap[v, v] += w
        lap[u, v] -= w
        lap[v, u] -= w
    return lap


def log_partition(n, edges, weights):
    """ln der Summe über alle Spannbäume von Produkt(Kantengewichte) (Determinante des gewichteten Minors); -inf, wenn es keinen Spannbaum gibt."""
    if n <= 1:
        return 0.0
    sign, logdet = np.linalg.slogdet(laplacian(n, edges, weights)[:-1, :-1])
    return float(logdet) if sign > 0 else -math.inf


def inclusion_probabilities(n, edges, weights=None):
    """Wahrscheinlichkeit je Kante, in einem (nach Gewichtsprodukt) zufälligen Spannbaum zu liegen: w_e * R_eff(e), R_eff über die Pseudoinverse der Laplace-Matrix."""
    weights = [1.0] * len(edges) if weights is None else weights
    pinv = np.linalg.pinv(laplacian(n, edges, weights))
    out = []
    for (u, v, *_), w in zip(edges, weights):
        out.append(float(w * (pinv[u, u] + pinv[v, v] - 2.0 * pinv[u, v])))
    return out


def edge_weights(costs, beta):
    """Kantengewichte exp(-beta * (c - c_min)): die Verschiebung ändert die Verteilung nicht, vermeidet aber Überlauf."""
    cmin = min(costs)
    return [math.exp(-beta * (c - cmin)) for c in costs]


def expected_cost(n, edges, costs, beta):
    """Erwartete Baumlänge bei Temperatur beta: Summe c_e * p_e(beta); beta = 0 gleichverteilt."""
    p = inclusion_probabilities(n, edges, edge_weights(costs, beta))
    return float(sum(c * q for c, q in zip(costs, p)))


def log_prob_tree(n, edges, costs, beta, tree):
    """ln der Wahrscheinlichkeit eines Baums (Kantennummern) bei Temperatur beta."""
    cmin = min(costs)
    w = edge_weights(costs, beta)
    return -beta * sum(costs[e] - cmin for e in tree) - log_partition(n, edges, w)


def mst_edges(n, edges, costs):
    """Kantennummern des billigsten Baums (Kruskal, Ordnung (Kosten, Nummer))."""
    uf = UnionFind(n, "full")
    return [e for e in sorted(range(len(edges)), key=lambda i: (costs[i], i)) if uf.union(edges[e][0], edges[e][1])]


def p_mst(n, edges, costs, beta):
    """Wahrscheinlichkeit, dass der bei Temperatur beta gewürfelte Baum der billigste (MST) ist."""
    return math.exp(log_prob_tree(n, edges, costs, beta, mst_edges(n, edges, costs)))


def enumerate_trees(n, edges):
    """Alle Spannbäume als sortierte Tupel von Kantennummern (Brute-Force für Tests; nur kleine Graphen)."""
    out = []
    for sub in combinations(range(len(edges)), n - 1):
        uf = UnionFind(n, "full")
        if all(uf.union(edges[e][0], edges[e][1]) for e in sub):
            out.append(sub)
    return out


def tree_weight(weights, tree):
    return math.prod(weights[e] for e in tree)


def tree_stats(n, edges, tree):
    """(Blätter, größter Grad) eines Baums."""
    deg = [0] * n
    for e in tree:
        deg[edges[e][0]] += 1
        deg[edges[e][1]] += 1
    return sum(1 for d in deg if d == 1), max(deg)


# --- Würfeln --------------------------------------------------------------------------------------------------------------------------------------


class Walker:
    """Nachbarlisten mit kumulierten Gewichten für gewichtete Zufallsschritte."""

    def __init__(self, n, edges, weights=None):
        weights = [1.0] * len(edges) if weights is None else weights
        self.n = n
        self.nbrs = [[] for _ in range(n)]
        self.eids = [[] for _ in range(n)]
        self.cum = [[] for _ in range(n)]
        for e, ((u, v, *_), w) in enumerate(zip(edges, weights)):
            for a, b in ((u, v), (v, u)):
                self.nbrs[a].append(b)
                self.eids[a].append(e)
                self.cum[a].append(w)
        for a in range(n):
            acc, out = 0.0, []
            for w in self.cum[a]:
                acc += w
                out.append(acc)
            self.cum[a] = out

    def step(self, u, rng):
        c = self.cum[u]
        i = bisect_right(c, rng.random() * c[-1])
        if i >= len(c):
            i = len(c) - 1
        return self.nbrs[u][i], self.eids[u][i]


class BudgetExceeded(Exception):
    """Ein Zufallsweg-Verfahren hat sein Schrittbudget überschritten (bei niedriger Temperatur laufen die Wege sehr lang)."""


def wilson(walker, rng, root=0, record=False, max_steps=None):
    """Wilson: der Baum startet am Wurzelknoten; für jeden Knoten (in Nummernfolge), der noch nicht im Baum ist, läuft ein Zufallsweg bis zum Baum, Schleifen werden gelöscht (zuletzt gewählter Ausgang zählt), der
    schleifenfreie Weg kommt in den Baum. Gibt (Baum als sortiertes Tupel von Kantennummern, Schritte, Wege); `Wege` (nur mit `record`): je Start {"start", "walk" (alle besuchten Knoten), "path" (schleifenfreier
    Weg als Knotenfolge bis zum Baum), "erased" (Zahl gelöschter Schleifenschritte)}."""
    n = walker.n
    in_tree = [False] * n
    in_tree[root] = True
    nxt = [None] * n
    steps = 0
    walks = []
    for s in range(n):
        if in_tree[s]:
            continue
        u = s
        walk = [u] if record else None
        while not in_tree[u]:
            v, e = walker.step(u, rng)
            nxt[u] = (v, e)
            u = v
            steps += 1
            if max_steps is not None and steps > max_steps:
                raise BudgetExceeded(steps)
            if record:
                walk.append(u)
        path = [s]
        u = s
        while not in_tree[u]:
            in_tree[u] = True
            u = nxt[u][0]
            path.append(u)
        if record:
            walks.append({"start": s, "walk": walk, "path": path, "erased": (len(walk) - 1) - (len(path) - 1)})
    tree = tuple(sorted(nxt[v][1] for v in range(n) if v != root and nxt[v] is not None))
    return tree, steps, walks


def aldous_broder(walker, rng, root=0, max_steps=None):
    """Aldous-Broder: Zufallsweg ab `root`, bis jeder Knoten besucht wurde; die Kante des ersten Besuchs jedes Knotens gehört zum Baum. Gibt (Baum, Schritte)."""
    n = walker.n
    seen = [False] * n
    seen[root] = True
    left = n - 1
    u = root
    tree = []
    steps = 0
    while left:
        v, e = walker.step(u, rng)
        steps += 1
        if max_steps is not None and steps > max_steps:
            raise BudgetExceeded(steps)
        if not seen[v]:
            seen[v] = True
            tree.append(e)
            left -= 1
        u = v
    return tuple(sorted(tree)), steps


def random_kruskal(n, edges, rng, weights=None):
    """Kruskal mit zufälliger Kantenordnung: gibt (Baum, betrachtete Kanten). NICHT gleichverteilt. Ohne Gewichte (oder bei lauter gleichen) wird die Ordnung gemischt; sonst bekommt jede Kante eine
    exponentialverteilte Uhr mit Rate w_e und die Kanten werden nach ihrer Uhr geordnet (hohes Gewicht = früh): die Temperatur wirkt, aber die Verteilung ist trotzdem nicht die Gibbs-Verteilung."""
    order = list(range(len(edges)))
    if weights is None or min(weights) == max(weights):
        rng.shuffle(order)
    else:
        clock = [rng.expovariate(w) for w in weights]
        order.sort(key=clock.__getitem__)
    uf = UnionFind(n, "full")
    tree = []
    examined = 0
    for e in order:
        examined += 1
        if uf.union(edges[e][0], edges[e][1]):
            tree.append(e)
            if len(tree) == n - 1:
                break
    return tuple(sorted(tree)), examined


def random_kruskal_distribution(n, edges):
    """Exakte Verteilung von `random_kruskal` (alle Kantenordnungen; nur m <= 8): {Baum: Wahrscheinlichkeit}."""
    m = len(edges)
    counts = {}
    total = 0
    for order in permutations(range(m)):
        uf = UnionFind(n, "full")
        tree = tuple(sorted(e for e in order if uf.union(edges[e][0], edges[e][1])))
        counts[tree] = counts.get(tree, 0) + 1
        total += 1
    return {t: c / total for t, c in counts.items()}


def sample_trees(kind, n, edges, weights, count, seed, root=0, budget=None):
    """`count` Bäume mit dem Verfahren `kind` ("wilson", "aldous", "kruskal"); gibt (Liste der Bäume, Gesamtschritte). Mit `budget` (Zufallsweg-Schritte insgesamt) endet die Stichprobe früher, wenn das Budget
    aufgebraucht ist: die Liste ist dann kürzer als `count` (ein angefangener Baum zählt nicht)."""
    rng = make_rng(seed, kind)
    out, steps = [], 0
    walker = Walker(n, edges, weights) if kind != "kruskal" else None
    for _ in range(count):
        left = None if budget is None else budget - steps
        if left is not None and left <= 0:
            break
        try:
            if kind == "wilson":
                t, s, _w = wilson(walker, rng, root, max_steps=left)
            elif kind == "aldous":
                t, s = aldous_broder(walker, rng, root, max_steps=left)
            elif kind == "kruskal":
                t, s = random_kruskal(n, edges, rng, weights)
            else:
                raise ValueError(f"unbekanntes Verfahren {kind}")
        except BudgetExceeded:
            steps = budget
            break
        out.append(t)
        steps += s
    return out, steps


def is_spanning_tree(n, edges, tree):
    if len(tree) != n - 1 or len(set(tree)) != len(tree):
        return False
    uf = UnionFind(n, "full")
    return all(uf.union(edges[e][0], edges[e][1]) for e in tree)


# --- Statistik ------------------------------------------------------------------------------------------------------------------------------------


def chi2_crit(df, z=Z999):
    """Kritischer Wert der Chi-Quadrat-Verteilung (Näherung nach Wilson-Hilferty; Standard: 99.9-%-Quantil)."""
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + z * math.sqrt(a)) ** 3


def chi2_stat(counts, probs):
    """Chi-Quadrat-Statistik der Häufigkeiten gegen die Wahrscheinlichkeiten (Zellen mit Wahrscheinlichkeit 0 müssen Häufigkeit 0 haben)."""
    total = sum(counts.values())
    stat = 0.0
    for t, p in probs.items():
        exp = total * p
        stat += (counts.get(t, 0) - exp) ** 2 / exp
    assert all(t in probs for t in counts), "Baum außerhalb der Verteilung"
    return stat


def tv_distance(counts, probs):
    total = sum(counts.values())
    keys = set(counts) | set(probs)
    return 0.5 * sum(abs(counts.get(t, 0) / total - probs.get(t, 0.0)) for t in keys)


def exact_tree_probs(n, edges, weights):
    """Exakte Verteilung über alle Bäume (Aufzählung): Produkt der Gewichte, normiert."""
    trees = enumerate_trees(n, edges)
    ws = [tree_weight(weights, t) for t in trees]
    z = sum(ws)
    return {t: w / z for t, w in zip(trees, ws)}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(xs, ys):
    """Rangkorrelation (ohne scipy)."""
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


# --- Zuverlässigkeit ------------------------------------------------------------------------------------------------------------------------------


def connected_subgraph_counts(n, edges):
    """Zahl der zusammenhängenden aufspannenden Teilgraphen nach Kantenzahl k (Liste c[k], exakt; alle 2^m Teilmengen, m <= 18)."""
    m = len(edges)
    counts = [0] * (m + 1)
    for mask in range(1 << m):
        uf = UnionFind(n, "full")
        k = 0
        for e in range(m):
            if mask >> e & 1:
                k += 1
                uf.union(edges[e][0], edges[e][1])
        if uf.components == 1:
            counts[k] += 1
    return counts


def survival_exact(counts, m, p):
    """Wahrscheinlichkeit, dass das Netz bei Kantenausfall p verbunden bleibt: Summe c_k (1-p)^k p^(m-k)."""
    return sum(c * (1.0 - p) ** k * p ** (m - k) for k, c in enumerate(counts))


def failure_pattern(n, edges, p, rng):
    """Ein Ausfallmuster: (ausgefallene Kantennummern, Komponente je Knoten, verbunden?)."""
    failed = [e for e in range(len(edges)) if rng.random() < p]
    fs = set(failed)
    uf = UnionFind(n, "full")
    for e in range(len(edges)):
        if e not in fs:
            uf.union(edges[e][0], edges[e][1])
    comp = [uf.find(v) for v in range(n)]
    return failed, comp, uf.components == 1


def survival_mc(n, edges, p, trials, seed):
    """Monte-Carlo-Schätzer der Verbundenheitswahrscheinlichkeit: (Schätzwert, Standardfehler). Schnelle Schleife: jede Kante überlebt mit Wahrscheinlichkeit 1 - p, Union-Find zählt die Komponenten."""
    rng = make_rng(seed, "surv", p)
    rnd = rng.random
    ok = 0
    for _ in range(trials):
        parent = list(range(n))
        comps = n
        for u, v, *_ in edges:
            if rnd() >= p:
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                if u != v:
                    parent[u] = v
                    comps -= 1
                    if comps == 1:
                        break
        ok += comps == 1
    q = ok / trials
    return q, math.sqrt(q * (1.0 - q) / trials)


def expected_surviving_trees(n, edges, p):
    """Erwartete Zahl überlebender Spannbäume: tau * (1-p)^(n-1) (Linearität des Erwartungswerts; kein Wahrscheinlichkeitsmaß, > 1 möglich)."""
    return count_trees(n, edges) * (1.0 - p) ** (n - 1)
