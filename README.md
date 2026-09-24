# Zufällige Spannbäume – zählen und würfeln statt optimieren – Streamlit-Demo

Elftes und **letztes** Stück der **Spannbaum-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Bisher wurde optimiert: der *billigste* Baum. Ein Netz hat aber unvorstellbar viele Spannbäume, und ihre Gesamtheit hat Struktur. Die Demo stellt vier Fragen, alle gemessen: **(1) Zählen** – wie viele Spannbäume gibt es (Matrix-Baum-Satz, Kirchhoff 1847), und mit welcher Wahrscheinlichkeit liegt eine Kante in einem zufälligen Baum? **(2) Würfeln** – wie zieht man einen Baum gleichverteilt (Wilson, Aldous-Broder), und warum funktioniert "Kruskal mit Zufallsordnung" *nicht*? **(3) Vom Zufall zum MST** – wie teuer ist ein Zufallsbaum, und wie weit muss man die "Temperatur" absenken, bis der billigste Baum herauskommt? **(4) Zuverlässigkeit** – fällt jede Kante mit Wahrscheinlichkeit p aus, bleibt das Netz genau dann verbunden, wenn *ein* Spannbaum überlebt.

**Einordnung in die Reihe:** geplant waren elf Stücke, alle sind gebaut – die Reihe ist damit vollständig:

```
Kruskal (Wurzel)                                                                           [gebaut: kruskal-demo]
 ├─ Prim (Kontrast: wächst von einem Punkt)                                                [gebaut: prim-demo]
 ├─ Borůvka (Kontrast: alle Komponenten parallel)                                          [gebaut: boruvka-demo]
 ├─ Euklidischer MST (keine n²-Kantenliste, Delaunay)                                      [gebaut: euclidean-mst-demo]
 ├─ Gerichteter Spannbaum (Chu-Liu/Edmonds)                                                [gebaut: arborescence-demo]
 ├─ Bottleneck-/Grad-/Hop-beschränkter Spannbaum                                           [gebaut: constrained-mst-demo]
 │    └─ Kapazitierter MST                                                                 [gebaut: cmst-demo]
 ├─ Steiner-Baum                                                                           [gebaut: steiner-tree-demo]
 │    └─ Prize-Collecting Steiner-Baum                                                     [gebaut: pcst-demo]
 ├─ MST-Sensitivität & dynamischer MST                                                     [gebaut: mst-sensitivity-demo]
 └─ Zufällige Spannbäume & Kirchhoff                                                       [DIESES STÜCK]
```

Ergebnis in Kürze: **Ein gleichverteilter Zufallsbaum kennt keine Kosten: MST-Kanten liegen im Mittel in 24.9 % der Bäume, alle übrigen in 25.9 % – und ein Zufallsbaum ist 1.88-mal so lang wie der MST (20 Filialen, k = 6; im vollständigen Graphen 3.37-mal). Erst die Temperatur drückt das Verhältnis (1.10 bei b = 4, 1.03 bei b = 8), aber der MST selbst kommt auch bei b = 16 nur mit 24.8 % (Median) heraus, und das Würfeln wird kalt teuer: Wilson braucht bei b = 0 47 Zufallsschritte je Baum (Aldous-Broder 100), bei b = 8 gelingen ihm 100 Bäume nur auf 2 von 5 Instanzen, bei b = 16 auf keiner. "Kruskal mit Zufallsordnung" ist nicht gleichverteilt (Chi-Quadrat 189.9 gegen 45.4 kritisch; exakte Abweichung 4.1 %). Ein fester Baum überlebt bei p = 0.2 nur in 1.15 % der Fälle, das Netz (k = 6) in 99.9 %, bei k = 3 nur in 77.6 %; mehr Spannbäume heißen meist, aber nicht streng, mehr Zuverlässigkeit (Rangkorrelation 0.71 bis 0.84).**

| Frage | Ergebnis (20 Filialen + Depot, k = 6 nächste Nachbarn, Geländezuschlag 0.3, sofern nicht anders angegeben; **Median** über 5 feste Instanzen, Seeds 100000–100004, bzw. **50 Instanzen**, Seeds 200000–200049; alle Würfe mit `random.Random` und festen Zeichenketten-Seeds, plattformstabil) |
|---|---|
| **Stimmt die Zählung?** | ✅ Matrix-Baum-Satz gegen Aufzählung aller Spannbäume (120 Zufallsgraphen), Cayley n^(n−2) für n ≤ 9, Baum → 1, Kreis C_n → n, K_{a,b}, gewichtet (Summe der Gewichtsprodukte, 40 Graphen); Einschlusswahrscheinlichkeit w · R_eff gegen Aufzählung (60 Graphen), Foster: Summe = n − 1, Brücken → 1; auch bei 60 Filialen im vollständigen Graphen exakt (ganzzahlige Bareiss-Determinante) |
| **Wie viele Spannbäume?** | log10 der Zahl bei n = 8/12/20/30/40: **6.2/9.6/15.7/23.9/31.2**; bei k = 3/4/6/10/vollständig: 8.0/11.0/15.7/20.4/25.1 (vollständig: n^(n−2), Cayley). Brücken (Kanten in jedem Baum): k = 3 hat bei 40 % der Instanzen mindestens eine, k = 4 bei 8 %, ab k = 6 keine |
| **Sind MST-Kanten in Zufallsbäumen häufiger?** | ❌ **Vorab-Hypothese widerlegt.** Ohne Temperatur liegen MST-Kanten im Mittel in **24.9 %** der Bäume, alle anderen in **25.9 %**: die Gleichverteilung kennt keine Kosten |
| **Wie teuer ist ein Zufallsbaum?** | Erwartete Baumlänge / MST bei n = 8/12/20/30/40: **1.98/2.05/1.88/1.82/1.79** (kaum vom n abhängig); bei k = 3/6/vollständig: **1.30/1.88/3.37** – je dichter das Netz, desto mehr teure Kanten zur Auswahl |
| **Temperatur** | Verhältnis bei b = 2/4/8: **1.28/1.10/1.03**; Wahrscheinlichkeit des MST (Median): b = 4 **0.03 %**, b = 8 **2.7 %**, b = 16 **24.8 %** – auch bei b = 16 ist der MST bei 20 Filialen nicht der wahrscheinlichste Baum "fast sicher". Erwartungswert = −d ln Z/dβ exakt (Test gegen Differenzenquotient und Aufzählung) |
| **Würfeln: Wilson gegen Aldous-Broder** | Zufallsschritte je Baum bei n = 8/12/20/30/40: Wilson **13.8/22.8/46.9/93/138**, Aldous-Broder 22/45/100/190/286 (Verhältnis 0.59/0.49/0.47/0.50/0.49) – Wilson braucht etwa halb so viele |
| **Würfeln bei niedriger Temperatur** | ⚠️ b = 2/4: Wilson 87/882 Schritte je Baum, Aldous-Broder 439/5 734; **b = 8**: 100 Bäume gelingen Wilson in einer Million Schritten nur auf **2 von 5** Instanzen (im besten Fall 9 058 Schritte je Baum), Aldous-Broder auf keiner; **b = 16**: auf **keiner**. Die Demo hat ein Schrittbudget und weist es aus |
| **Gleichverteilung** | Lehrbuch-Graph (21 Spannbäume, je 20 000 Würfe), Chi-Quadrat (kritischer Wert 45.4 bei 99.9 %): **Wilson 15.1, Aldous-Broder 17.5, Kruskal mit Zufallsordnung 189.9**; K4 (16 Bäume): 18.1/13.5/60.8 (kritisch 37.8). Exakte Abweichung von Kruskal mit Zufallsordnung: **4.1 %** (alle Kantenordnungen aufgezählt); auf der Karte weicht die größte Kantenhäufigkeit um 0.045 ab (Wilson 0.024 = Stichprobenrauschen bei 2000 Bäumen) |
| **Kruskal mit Temperatur** | Ordnet man die Kanten nach exponentiellen Uhren mit Rate exp(−β · Länge), folgt Kruskal der Temperatur – aber falsch: im Lehrbuch-Graphen bei b = 2 mittlere Länge **16.41** gegen den exakten Erwartungswert **16.81** (Wilson 16.80) |
| **Zuverlässigkeit** | Ein fester Baum überlebt mit (1 − p)^(n−1): **1.15 %** bei p = 0.2, **0.08 %** bei p = 0.3. Das Netz (Median über 50 Instanzen, 1000 Monte-Carlo-Läufe) bleibt bei p = 0.2/0.3 verbunden: k = 6 **99.9/99.3 %**, k = 4 **97.6/87.9 %**, k = 3 **77.6/49.6 %**; bei p = 0.5: 81.5 % (k = 6), etwa 4 % (k = 3) |
| **Hängt Zuverlässigkeit an der Baumzahl?** | Meist, nicht streng: Spearman-Rangkorrelation (log10 Baumzahl gegen Verbundenheit, 50 Instanzen) **0.71** (k = 4, p = 0.3), **0.72** (k = 3, p = 0.2), **0.84** (k = 6, p = 0.5). Die Baumzahl sagt nicht, wo die dünnen Stellen liegen |
| **Erwartete überlebende Bäume (Markov)** | Weit über 1, also keine Schranke: 6.6 x 10^14 bei k = 6, p = 0.1; noch 8.7 x 10^4 bei k = 3, p = 0.3, obwohl das Netz dort nur in 44 % der Fälle verbunden bleibt |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Zählen** (Karte mit Kanten nach Einschlusswahrscheinlichkeit gefärbt, Brücken blau gestrichelt, dünne Linie = MST; τ als Zehnerpotenz und exakt; die sechs wahrscheinlichsten Kanten; Kurve der Baumzahl über n gegen n^(n−2); im Lehrbuchbeispiel die Laplace-Matrix von Hand) → **Würfeln** (Slider über die Zufallswege von Wilson: Zufallsweg orange gepunktet, schleifenfreier Weg orange dick, Baum wächst; Tabelle Schritte je Baum und Abweichung für alle drei Verfahren) → **Verteilung** (Histogramm der Baumlängen mit MST und Erwartungswert, Streudiagramm Kantenhäufigkeit gegen exakte Wahrscheinlichkeit, Gleichverteilungstest als Balken) → **Zuverlässigkeit** (Slider über 20 feste Ausfallmuster: ausgefallene Kanten rot gestrichelt, Komponenten farbig; Kurve Netz gegen fester Baum über p).
2. **Kennzahlen:** Spannbäume (Zehnerpotenz), Zufallsbaum / MST, P(MST), Netz verbunden.
3. **🔬 Auf Abruf:** Korrelations-Experiment (50 Instanzen), Sweeps über n / k / Gelände / Temperatur / Ausfall für Baumzahl, Kosten, Schritte, Zuverlässigkeit und Abweichung der Verfahren (5 feste Instanzen, Median und 10./90. Perzentil).

Presets (9): Standardfall, Lehrbuchbeispiel, dünnes Netz (Brücken), vollständiger Graph (Cayley), Wilson in Aktion, kalt (der MST kommt heraus), Kruskal mit Zufallsordnung, hohe Ausfallquote, dünn und zerbrechlich.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Standardfall** | 20 Filialen, k = 6, Seed 35 | 75 Kandidatenkanten, 1.9 x 10^15 Spannbäume; jede Kante in 20 bis 37 % der Bäume, keine Brücke; Zufallsbaum 1.87-mal so lang wie der MST (666.46 gegen 357.32); MST mit Wahrscheinlichkeit 5.3 x 10^-16 |
| **Lehrbuchbeispiel** | 5 Knoten, 7 Kanten | 21 Spannbäume (Determinante des Laplace-Minors); jede Kante in 48 bis 62 % der Bäume; Zufallsbaum 20.00 gegen 15 (1.33-fach), MST mit Wahrscheinlichkeit 1/21 = 4.8 % |
| **Dünnes Netz** | k = 3, Seed 19 | 37 Kanten, 12 720 000 Spannbäume, 2 Brücken (Wahrscheinlichkeit 1); Zufallsbaum 1.26-mal so lang wie der MST |
| **Vollständiger Graph** | 9 Filialen, vollständig | 10^8 Spannbäume (Cayley), jede Kante in genau 20 % (2/n); Zufallsbaum 2.28-mal so lang (473.06 gegen 207.70) |
| **Wilson in Aktion** | 12 Filialen, Weg 1 | Weg 1 von 7: 31 Zufallsschritte, 27 als Schleifen gelöscht, 4 Kanten bleiben; ganzer Baum 39 Schritte (Länge 506.15 gegen MST 248.04); Mittel über 100 Bäume: Wilson 25.4, Aldous-Broder 46.1 Schritte |
| **Kalt** | 6 Filialen, b = 8 | MST mit Wahrscheinlichkeit 74.8 % (exakt), in 500 gewürfelten Bäumen 72.2 %; mittlere Länge 193.99 gegen MST 190.97 (Erwartungswert 193.94); 321 Zufallsschritte je Baum |
| **Kruskal mit Zufallsordnung** | Lehrbuch, 2000 Bäume | Chi-Quadrat Wilson 15.1, Aldous-Broder 17.5, Kruskal 189.9 (kritisch 45.4); exakte Abweichung 4.1 % |
| **Hohe Ausfallquote** | k = 4, Seed 1, p = 0.3 | 51 Kanten, 1.6 x 10^11 Bäume; Muster 1: 16 Kanten fallen aus, 2 Teile; Netz bleibt in 84.8 % verbunden, ein fester Baum in 0.08 %; erwartete überlebende Bäume 1.3 x 10^8 |
| **Dünn und zerbrechlich** | k = 3, Seed 3, p = 0.2 | 37 Kanten, 2.0 x 10^7 Bäume, 1 Brücke; Muster 1 zerfällt (7 Kanten, 2 Teile); Netz bleibt in 62.0 % verbunden (fester Baum 1.15 %), obwohl 2.3 x 10^5 Bäume erwartet werden: viele Bäume, aber eine Brücke genügt |

## Modell und Verfahren

- **Instanz** (`rst_scenario.py`): die Karte der kruskal-demo (Depot + Filialen, Kosten = Länge x Geländefaktor, k nächste Nachbarn oder vollständig) und ein handgebautes Lehrbuchbeispiel (5 Knoten, 7 Kanten, 21 Spannbäume).
- **Zählen** (`rst_algorithm.py`): Laplace-Matrix, Determinante mit **Bareiss** in ganzen Zahlen (exakt); gewichtet über `slogdet`. Einschlusswahrscheinlichkeit einer Kante = Gewicht mal effektiver Widerstand (Pseudoinverse); Summe = n − 1 (Foster).
- **Temperatur:** Kantengewicht exp(−β · Länge) macht das Gewicht eines Baums zu exp(−β · Baumlänge), eine Gibbs-Verteilung über alle Bäume; b = β mal mittlere MST-Kantenlänge (b = 0 gleichverteilt, b = 16 ist die kälteste Stufe). Erwartete Länge = Σ_e Länge_e · P(e ∈ T).
- **Würfeln:** **Wilson** (schleifenfreie Zufallswege, gewichtete Übergänge, jede Wurzel liefert dieselbe Verteilung), **Aldous-Broder** (Zufallsweg bis alle Knoten besucht sind), **Kruskal mit Zufallsordnung** (gemischte oder nach exponentiellen Uhren geordnete Kanten). Schrittbudget je Stichprobe, weil Zufallswege bei niedriger Temperatur explodieren.
- **Zuverlässigkeit:** Verbundenheitswahrscheinlichkeit exakt über die Zahl zusammenhängender Kantenteilmengen (kleine Graphen), sonst Monte-Carlo mit Union-Find; erwartete überlebende Bäume = τ · (1 − p)^(n−1); Chi-Quadrat-Quantil per Wilson-Hilferty-Näherung, Spearman ohne scipy.
- Nur numpy (App), keine Zusatzbibliothek; die Tests brauchen nur pytest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "MST-Kanten sind in Zufallsbäumen wahrscheinlicher" – widerlegt.** Ohne Temperatur 24.9 % gegen 25.9 %.
- **Vorab-Hypothese "bei b = 16 ist der MST fast sicher" – widerlegt** für 20 Filialen: Median 24.8 %; das Verhältnis Länge/MST ist dort allerdings nur noch 1.004.
- **Wilson ist nur bei hoher Temperatur billig.** Das Verhältnis zu Aldous-Broder (etwa 1:2) bleibt, aber beide explodieren, wenn das Gewicht die billigen Kanten stark bevorzugt: bei b = 16 gelingt in einer Million Schritten auf keiner der fünf Instanzen ein Baum. Vermutete Ursache (nicht isoliert): der Zufallsweg bleibt auf billigen Kanten hängen und nimmt die teuren, die zum Baum führen, selten.
- **Kruskal mit Zufallsordnung ist verzerrt**, auch mit Temperatur (16.41 gegen 16.81), zeigt aber, dass "plausibel" nicht "gleichverteilt" ist. Bei 100 Bäumen ist die Abweichung der Kantenhäufigkeit reines Rauschen (etwa 0.1), darum verwendet der Sweep 2000 Bäume.
- **Zuverlässigkeit ist keine Determinante.** Die Zahl der Bäume ist exakt und billig; die Wahrscheinlichkeit, verbunden zu bleiben, ist #P-schwer (Provan und Ball 1983), hier Monte-Carlo mit 1000 Läufen (±2–3 %), exakt nur im Lehrbuch. Das Näherungsschema von Karger (2001) ist nicht gebaut.
- **Baumzahl ≠ Zuverlässigkeit.** Rangkorrelation 0.71–0.84, keine Gleichheit: Brücken und dünne Schnitte zählen mehr als die Gesamtzahl.
- **Markov-Schranke wertlos.** Die erwartete Zahl überlebender Bäume liegt in allen betrachteten Netzen weit über 1.
- **Nur die Grundform von Wilson.** Schneller Verfahren (Schild 2018, fast lineare Laufzeit) sind nicht gebaut; das Kantengewicht exp(−β · Länge) ist eine Modellwahl, keine Physik des Netzes; Zufallsschritte sind kein Sekundenmaß.
- **Synthetische Instanzen.** 5 bzw. 50 feste Instanzen je Zahl; Ausfälle unabhängig und gleich wahrscheinlich, keine Knotenausfälle, keine Kapazitäten.

## Verifikation

- `tests/test_algorithm.py` (24 Tests): Matrix-Baum-Satz gegen Aufzählung (120 Graphen), Cayley, Kreise, K_{a,b}, gewichtete Zustandssumme, Einschluss gegen Aufzählung und Foster, Dreieck von Hand, Sampler gegen die exakte (gewichtete) Verteilung mit Chi-Quadrat (K4, Lehrbuch, Graph mit Brücke), jede Wurzel, Kruskal mit Zufallsordnung gegen seine exakte Verteilung (und nicht gleichverteilt), Wilson-Wiedergabe reproduziert den Baum, Determinismus, Temperatur (Erwartungswert, Ableitung, Grenzwerte, P(MST)), Zuverlässigkeit exakt gegen geschlossene Formeln und Monte-Carlo, erwartete überlebende Bäume gegen Aufzählung, Sonderfälle.
- `tests/test_scenario.py`, `test_evaluation.py` (Analyse, Schrittbudget, Sampler-Vergleich, Gleichverteilungstest, Zuverlässigkeit, Korrelation, Sweeps), `test_presets.py` (Bänder + jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt und jede Position der Schritt-Regler, Randwerte, Budgetwarnungen, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer).
- Für die Prüfung genügt **pytest**.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Kirchhoff, G. (1847). *Ueber die Auflösung der Gleichungen, auf welche man bei der Untersuchung der linearen Vertheilung galvanischer Ströme geführt wird.* Annalen der Physik und Chemie 72, 497–508.
- Wilson, D. B. (1996). *Generating random spanning trees more quickly than the cover time.* STOC 1996, 296–303.
- Broder, A. (1989). *Generating random spanning trees.* FOCS 1989, 442–447.
- Aldous, D. J. (1990). *The random walk construction of uniform spanning trees and uniform labelled trees.* SIAM Journal on Discrete Mathematics 3(4), 450–465.
- Provan, J. S., & Ball, M. O. (1983). *The complexity of counting cuts and of computing the probability that a graph is connected.* SIAM Journal on Computing 12(4), 777–788.
- Karger, D. R. (2001). *A randomized fully polynomial time approximation scheme for the all-terminal network reliability problem.* SIAM Review 43(3), 499–522 (nur genannt, nicht gebaut).
- Kruskal (1956) und Union-Find (Tarjan 1975) wie in der kruskal-demo.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
