# RailMind — AI Train Route Optimizer & Delay Predictor

## Project Report

---

## 1. Problem Statement

Indian Railways is the fourth-largest railway network in the world, carrying over 23 million passengers daily across more than 7,000 stations. Despite this massive scale, passengers routinely face two frustrating pain points:

1. **Unpredictable Delays** — Train delays are endemic to the Indian rail system. According to publicly available statistics, the average punctuality of Indian long-distance trains has historically hovered around 65–75%. Passengers have no reliable way to anticipate *how late* a specific train will be on a given day, making trip planning a guessing game.

2. **Poor Connection Planning** — For journeys that require changing trains at an intermediate station, passengers must manually estimate whether a delayed first train will still allow them to catch their connecting service. There is no automated tool that combines delay forecasts with timetable data to answer the question: *"Will I make my connection?"*

**RailMind** addresses both problems by combining a **graph-based route search engine** with a **machine-learning delay prediction model** to give passengers intelligent, delay-aware route recommendations.

---

## 2. Why This Problem Matters

This is not an abstract academic exercise. The problem was chosen because it is observable in daily life:

- **Personal experience**: Anyone who has traveled by Indian Railways on multi-leg routes knows the anxiety of watching a 20-minute delay balloon and wondering whether the connection at the next junction will hold.
- **No existing tool does this**: Current platforms like NTES (National Train Enquiry System) and IRCTC show live running status *after the fact*, but do not predict delays *before* travel. No publicly available tool combines predictive delay analytics with route planning.
- **Real-world impact**: A missed connection can mean an unplanned overnight stay at an unfamiliar station, additional ticket costs, and significant inconvenience. Even a probabilistic warning — "this connection is risky" — can save passengers from making a poor booking decision.

The project applies core data science and AI concepts — **graph algorithms**, **supervised learning**, and **feature engineering** — to a problem space where they deliver genuine, practical value.

---

## 3. Approach & Solution Architecture

RailMind is a command-line application built in Python. It is intentionally designed as a CLI tool (not a web app) to keep the focus on the algorithmic and ML layers rather than front-end concerns.

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface (Typer + Rich)          │
│   Commands: search | stations | schedule | predict-delay│
│              train-model | generate-data | stats        │
└─────────────────┬───────────────────────────────────────┘
                  │
     ┌────────────┼────────────────────┐
     ▼            ▼                    ▼
┌──────────┐ ┌──────────────┐  ┌──────────────────┐
│ Railway  │ │   Route      │  │  Delay           │
│ Graph    │ │   Planner    │  │  Predictor       │
│ Builder  │ │  (A*/Yen)    │  │  (Random Forest) │
└────┬─────┘ └──────┬───────┘  └────────┬─────────┘
     │              │                   │
     │         ┌────┴──────┐     ┌──────┴──────┐
     │         │Connection │     │ Recommender │
     │         │ Checker   │     │  (Scoring)  │
     │         └───────────┘     └─────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│          Data Layer (CSV)            │
│  stations.csv | train_schedule.csv   │
│         historical_delays.csv        │
└──────────────────────────────────────┘
```

### 3.2 Module Breakdown

| Module | File | Responsibility |
|---|---|---|
| **Graph Builder** | `modules/graph_builder.py` | Constructs a NetworkX `MultiDiGraph` from station and schedule CSVs. Nodes are stations; edges are direct train segments between consecutive stops. Implements a Haversine-based admissible heuristic for A*. |
| **Route Planner** | `modules/route_planner.py` | Finds up to *k* routes between any two stations. Checks for (1) direct trains, (2) one-interchange routes, and (3) two-interchange routes. Validates connection timing with minimum 20-minute buffer windows. |
| **Delay Predictor** | `modules/delay_predictor.py` | Trains and serves a scikit-learn `RandomForestRegressor` (120 trees, max depth 12). Features include encoded train ID, station code, zone, scheduled hour, day-of-week, month, and historical average delay. Outputs `(predicted_delay, confidence)` where confidence is derived from inter-tree variance. |
| **Connection Checker** | `modules/connection_checker.py` | For each interchange in a route, predicts the first train's delay, computes a "buffer window" to the next departure, and labels the connection as **SAFE** (>30 min), **RISKY** (15–30 min), or **NOT POSSIBLE** (<15 min). |
| **Recommender** | `modules/recommender.py` | Scores and ranks routes using a weighted formula: 40% travel time + 40% predicted reliability + 20% simplicity (fewer interchanges). Labels the top route as "Best Route," the fastest as "Fastest," and the most reliable as "Safest." |
| **Display** | `utils/display.py` | Rich-powered terminal rendering: colored route cards, connection feasibility indicators, delay severity bars, model metrics panels. |

---

## 4. Data Pipeline

### 4.1 Dataset Overview

| File | Records | Description |
|---|---|---|
| `stations.csv` | 78 stations | Major Indian railway junctions with geo-coordinates, zone, and state information. |
| `train_schedule.csv` | 235 stops across 64 trains | Timetable data for Rajdhani, Shatabdi, Superfast, and Express services with absolute-minute timestamps for arrival/departure. |
| `historical_delays.csv` | 15,280 records | Simulated historical delay data covering multiple months, days-of-week, and hours for each train–station pair. Includes the actual delay in minutes and the historical average. |

### 4.2 Data Generation Strategy

Real-time delay data from Indian Railways is not available through any public API. NTES provides live running status on their website, but it is protected by anti-bot measures and rate limiting. We implemented a web scraper (`utils/ntes_scraper.py`) for NTES but found that large-scale data collection was impractical for a course project timeline.

Instead, the project uses a **synthetic data generator** (`utils/data_generator.py`) that produces realistic delay distributions:

- Delays follow a skewed distribution (most trains are slightly late, some are very late).
- Zone-level and station-level variance is injected to simulate regional reliability differences.
- Seasonal patterns (monsoon months see higher delays) and time-of-day effects (early-morning trains tend to be more punctual) are incorporated.

This approach is explicitly documented as synthetic, and the system architecture is designed to pivot to real data when it becomes available.

---

## 5. Key Technical Decisions

### 5.1 Why a MultiDiGraph?

Indian Railways frequently has multiple trains running the same route segment (e.g., Mumbai–Surat is served by dozens of trains). A standard graph would collapse these into a single edge. A **MultiDiGraph** preserves each train as a separate edge, allowing the route planner to enumerate all direct train options and compare their schedules.

### 5.2 Why Random Forest over Deep Learning?

The delay prediction problem is a tabular regression task with ~15,000 records and 7 features. In this regime:

- **Random Forests** are well-known to outperform neural networks on small-to-medium tabular data (see the Grinsztajn et al. 2022 benchmark study).
- They provide interpretable feature importances, which helped validate that the model was learning sensible patterns (e.g., historical average delay was the strongest predictor, followed by time-of-day).
- Training is near-instant (~2 seconds), which matters for a CLI tool that can retrain on demand.

A gradient-boosted model (XGBoost) was considered but provided marginal improvement for significantly more hyperparameter tuning complexity.

### 5.3 Confidence from Inter-Tree Variance

Rather than outputting only a point prediction, the delay predictor computes predictions from all 120 individual trees and uses the standard deviation as a proxy for uncertainty. High agreement across trees → high confidence → the model "knows" this scenario well. This is not a formal Bayesian interval but is a pragmatic, well-established technique for Random Forest uncertainty quantification.

### 5.4 Connection Buffer Thresholds

The 30-minute (safe), 15-minute (risky), and ≤15-minute (not possible) thresholds were chosen based on typical Indian Railways platform transfer times at major junctions. These are configurable constants in `connection_checker.py`.

### 5.5 Weighted Ranking Formula

The recommender uses `0.40 * time + 0.40 * reliability + 0.20 * simplicity`. The equal weighting of time and reliability reflects the insight that *a fast route you're likely to miss is worse than a slightly slower reliable one*. Simplicity (fewer interchanges) gets a lower weight because connections are already risk-assessed.

---

## 6. Challenges Faced

### 6.1 Web Scraping Limitations

The original plan was to scrape real delay data from NTES. We built a fully functional scraper with rotating user agents, request throttling, and checkpoint-based resumption. However:

- NTES's anti-bot protections became increasingly aggressive during testing.
- The site's HTML structure changed mid-development, requiring scraper updates.
- Scraping 3,000+ station pairs at respectful intervals would take ~2 hours per run, making rapid iteration impractical.

**Resolution**: We pivoted to synthetic data generation while preserving the scraper as an optional module (`scrape-all` CLI command) for users who wish to attempt real data collection.

### 6.2 Graph Scalability

With 78 stations and 64 trains, the two-interchange route search involves an O(n³) loop over all possible mid-station pairs (78 × 78 × 78 ≈ 474,552 combinations). To keep search responsive:

- We cap results at 20 two-interchange routes and 30 one-interchange routes.
- The `_trains_between()` method short-circuits if no trains serve a station pair.
- Practical testing showed search times under 3 seconds for any station pair.

Scaling to the full Indian Railways network (7,000+ stations) would require a fundamentally different approach — likely a contraction hierarchy or a pre-computed connection scan algorithm.

### 6.3 Feature Engineering for Unseen Trains/Stations

The Random Forest uses label-encoded train IDs and station codes. At prediction time, a train or station not present in the training set would throw an error. We handle this gracefully by:

- Mapping unseen labels to index `0` (a fallback).
- Falling back to a global average delay of 12 minutes when station-specific averages are unavailable.

This ensures the predict-delay command never crashes, even for ad-hoc queries.

### 6.4 Windows Terminal Encoding

Rich's Unicode-heavy output (emoji, box-drawing characters) caused encoding errors on some Windows terminal configurations. We resolved this by force-reconfiguring `sys.stdout` to UTF-8 with error replacement at the top of `display.py`.

---

## 7. Results & Evaluation

### 7.1 Model Performance

The Random Forest delay prediction model achieves the following metrics on a held-out 20% test set:

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ~5.5 minutes |
| Root Mean Squared Error (RMSE) | ~7.2 minutes |
| R² Score | ~0.87 |

These numbers should be interpreted in the context of a synthetic dataset. On real-world data, we would expect higher MAE (delays are noisier and influenced by factors not captured in our features, such as weather and signaling failures). However, the model architecture and pipeline are designed to handle a simple data swap.

### 7.2 Route Search Quality

The route planner successfully finds direct, one-hop, and two-hop routes between all tested station pairs. Sample search results:

- **Mumbai Central → New Delhi**: Finds direct Rajdhani and Superfast options, plus one-interchange alternatives via Vadodara, Kota, and Bhopal.
- **Chennai → Kolkata**: Finds multi-leg options via Vijayawada and Nagpur.
- All routes include connection feasibility assessments with predicted delays.

### 7.3 Connection Checker Accuracy

In synthetic testing, the connection checker correctly identified:

- **SAFE** connections where buffer windows were ample (>30 minutes).
- **RISKY** connections where predicted delays consumed most of the buffer.
- **NOT POSSIBLE** connections where even optimistic predictions left insufficient transfer time.

---

## 8. What I Learned

### 8.1 Graph Theory in Practice

Building this project solidified my understanding of how graph representations map to real-world networks. The choice between a simple graph, digraph, and multi-digraph had direct consequences for the expressiveness of route queries. The Haversine-based heuristic for A* search was a satisfying application of geographic math to inform pathfinding.

### 8.2 ML on Tabular Data

Working with Random Forests reinforced that deep learning is not always the answer. For structured, tabular datasets of moderate size, ensemble tree methods remain highly competitive and offer practical advantages: fast training, no GPU requirement, built-in feature importance, and natural uncertainty estimation through inter-tree variance.

### 8.3 System Design Thinking

The project forced me to think about how multiple components interact: the graph feeds routes to the planner, which feeds legs to the connection checker, which queries the delay predictor, which feeds results to the recommender, which finally renders through the display layer. Keeping these interfaces clean and well-typed (using dataclasses) was essential for maintainability.

### 8.4 Data Realism vs. Pragmatism

The scraping saga taught me an important lesson: when real data acquisition is impractical within project constraints, it's better to build a robust pipeline with synthetic data and document the limitation honestly than to deliver a fragile system tied to a flaky data source. The architecture is data-source-agnostic by design.

### 8.5 CLI UX Matters

Using Rich for terminal output transformed the tool from a wall of print statements into something visually clear and even enjoyable to use. Color-coding connection risk levels, rendering route cards in bordered panels, and showing spinner animations during computation — these details don't affect the algorithm but dramatically improve the user experience.

---

## 9. Future Work

If this project were to be extended beyond the course:

1. **Real Data Integration** — Partner with a data provider or build a more resilient scraper to replace synthetic data with actual delay records.
2. **Live Running Status** — Integrate real-time train position data to update delay predictions dynamically during travel.
3. **Web/Mobile Interface** — Wrap the backend in a Flask/FastAPI service and build a responsive front-end.
4. **Advanced ML** — Experiment with gradient boosting (LightGBM) and incorporate weather, festival, and signaling data as additional features.
5. **Full Network Scale** — Implement contraction hierarchies or RAPTOR to handle the full 7,000-station Indian Railways network.

---

## 10. Conclusion

RailMind demonstrates that meaningful AI applications don't require exotic architectures or massive compute. By combining a well-chosen graph representation (NetworkX MultiDiGraph), a practical ML model (Random Forest with confidence estimation), and thoughtful system design (modular pipeline with clean interfaces), it delivers a tool that addresses a genuine, everyday frustration for Indian railway passengers.

The project is not a polished production system — it operates on synthetic data and covers a subset of stations. But the architecture, the algorithmic choices, and the engineering are designed for extensibility. The most important outcome is not the tool itself but the practice of identifying a real problem, decomposing it into tractable sub-problems, and applying course concepts to build a clear, well-documented solution.

---

*Submitted as part of the AI/ML course final project.*
*Author: Rishi*
*Date: March 2026*
