# Project Statement — RailMind
### AI Train Route Optimizer & Delay Predictor

**Author:** Rishi
**Date:** March 2026
**Course:** AI / Machine Learning

---

## 1. The Problem I Chose

I chose to build an intelligent route planning and delay prediction system for Indian Railways.

Indian Railways is the fourth-largest railway network in the world, carrying over 23 million passengers daily across more than 7,000 stations. Despite this massive scale, passengers routinely face two frustrating problems that no existing tool adequately solves:

**Problem 1 — Unpredictable Delays**

Train delays are a persistent reality in the Indian rail system. Long-distance train punctuality has historically hovered around 65–75%. When a passenger books a train, they have no reliable way to anticipate how late it will be on a given day, at a given hour, or in a given season. Platforms like IRCTC and NTES show live running status only after the train has already departed — there is no tool that predicts delays *before* you travel.

**Problem 2 — Unsafe Connection Planning**

Multi-leg journeys — where you board one train, get off at an intermediate city, and board a second train — are common across India. The danger is simple: if your first train runs late, you may miss your connecting train. Today, passengers estimate this risk manually and informally. There is no system that takes a planned itinerary, predicts how late each train is likely to be, and tells you whether your connection is safe, risky, or impossible.

**RailMind** was built to address both of these problems in a single, integrated tool.

---

## 2. Why This Problem Matters

This is not an abstract academic exercise. I chose this problem because it is directly observable in everyday life.

Anyone who has traveled by Indian Railways on a multi-leg journey knows the anxiety of watching a 20-minute delay grow larger and wondering whether the connection at the next junction will still hold. A missed connection can mean an unplanned overnight stay at an unfamiliar station, additional ticket costs, and significant inconvenience — especially for passengers traveling long distances or in unfamiliar regions.

More importantly, no publicly available tool addresses this. Current platforms (NTES, IRCTC, RailYatri) show historical or live delay information after the fact. None of them combine predictive delay analytics with route planning to give a passenger a forward-looking answer: *"Based on what we know about this train's behavior, will your connection hold?"*

This is a problem where data science and AI can deliver genuine, practical value — not just in a research paper, but in a tool someone could actually use before buying a ticket.

---

## 3. My Approach to Solving It

I decomposed the problem into five distinct sub-problems and built a modular pipeline to address each one.

### 3.1 Modelling the Railway Network as a Graph

The railway network is fundamentally a graph — stations are nodes, and train segments between consecutive stops are directed edges. I chose a **NetworkX MultiDiGraph** for this representation because multiple trains can serve the same station pair (e.g., Mumbai–Surat is served by dozens of trains). A standard graph would collapse these into a single edge and lose all scheduling information. The MultiDiGraph preserves each train as a separate, uniquely identifiable edge, which allows the route planner to enumerate every direct train option and compare their timetables.

### 3.2 Finding Routes Across Multiple Legs

Route search operates in three tiers:

1. **Direct trains** — scans the schedule for trains that serve both the origin and destination in the correct order.
2. **One-interchange routes** — iterates over all possible intermediate stations, checks whether Train A serves origin→mid and Train B serves mid→destination, and validates that the connecting departure is at least 20 minutes after the scheduled arrival.
3. **Two-interchange routes** — extends the search to three-leg journeys when fewer than the requested number of routes are found.

Results are capped (20 one-interchange and 30 two-interchange routes) to keep search times under 3 seconds even in worst-case scenarios.

### 3.3 Predicting Delays with Machine Learning

For each leg of a journey, I needed to estimate how late the train was likely to be at the relevant station. I trained a **Random Forest Regressor** on historical delay records. The model takes the following features as input:

- Label-encoded train ID
- Label-encoded station code
- Label-encoded railway zone
- Scheduled arrival hour (0–23)
- Day of week (0 = Monday … 6 = Sunday)
- Month (1–12)
- Historical average delay for that train–station pair

The model outputs a **predicted delay in minutes** along with a **confidence score**, derived from the standard deviation of predictions across all 120 individual decision trees. High agreement among trees signals that the model has seen similar scenarios before and is making a confident prediction. Low agreement signals uncertainty — useful information for the passenger.

### 3.4 Checking Connection Feasibility

For each interchange in a multi-leg route, the system:

1. Predicts the incoming train's delay at the interchange station.
2. Computes the buffer: `next_departure_time − (scheduled_arrival + predicted_delay)`.
3. Labels the connection:
   - **SAFE** — buffer exceeds 30 minutes
   - **RISKY** — buffer is 15–30 minutes
   - **NOT POSSIBLE** — buffer is 15 minutes or less

These thresholds reflect realistic platform transfer times at major Indian junctions and are configurable constants in the codebase.

### 3.5 Ranking Routes

Finally, routes are scored using a weighted formula:

```
score = 0.40 × time_score + 0.40 × reliability_score + 0.20 × simplicity_score
```

The equal weighting of travel time and reliability reflects a deliberate insight: a fast route you are likely to miss is worse than a slightly slower but dependable one. Simplicity (fewer interchanges) receives a lower weight because connections are already risk-assessed independently.

The top-ranked route is labeled **Best Route**, the fastest is labeled **Fastest**, and the most reliable is labeled **Safest**.

---

## 4. Key Technical Decisions

### Why Random Forest over Deep Learning?

The delay prediction problem is tabular regression — structured rows with categorical and numerical features — on approximately 15,000 records. For this regime, Random Forests are the right tool:

- Research (Grinsztajn et al., 2022) has shown that tree-based models consistently outperform neural networks on small-to-medium tabular datasets.
- Feature importances are interpretable: I could verify that the model was learning sensible patterns (historical average delay was the strongest predictor, followed by scheduled hour of arrival).
- Training completes in approximately 2 seconds, which matters for a CLI tool that supports on-demand retraining.
- No GPU is required, making the project runnable on any laptop.

I considered XGBoost as an alternative, but found that it provided only marginal predictive improvement for significantly more hyperparameter complexity. Random Forest was the pragmatic choice.

### Why a CLI tool, not a web app?

I deliberately chose to build a command-line interface rather than a web application. The goal was to keep the focus on the algorithmic and ML layers — the graph search, the model training, the connection logic, the ranking — rather than front-end concerns. A clean CLI with rich terminal formatting (via the Rich library) allowed me to build something visually usable without the overhead of a full web stack. The architecture is explicitly designed for a web API wrapper to be added later.

### Confidence from inter-tree variance

Rather than providing only a point prediction, the delay predictor computes predictions from all 120 individual trees and uses their standard deviation as an uncertainty proxy. This is not a formal Bayesian confidence interval, but it is a well-established, practical technique for Random Forest uncertainty quantification — and it gives passengers actionable information beyond a single number.

### Handling unseen labels at prediction time

Label-encoded train IDs and station codes create a known risk: at inference time, an unseen train or station would cause the encoder to throw an error. I handled this by mapping unseen labels to index `0` as a fallback and using a global average delay of 12 minutes when no station-specific historical average is available. This ensures the system never crashes on ad-hoc queries.

---

## 5. Challenges I Faced

### Challenge 1 — Real Data Was Not Available

The original plan was to train the model on real delay data scraped from NTES (the National Train Enquiry System). I built a fully functional web scraper with rotating user agents, request throttling (to be respectful of the server), and checkpoint-based resumption so that long scraping runs could be interrupted and continued.

However, several problems made this approach impractical within the project timeline:

- NTES has anti-bot protections that became increasingly aggressive during testing.
- The site's HTML structure changed mid-development, requiring scraper updates.
- Scraping 3,000+ station pairs at respectful intervals would take approximately 2 hours per complete run, making rapid iteration impossible.

I pivoted to a **synthetic data generator** that produces realistic delay distributions — incorporating skewed delays, zone-level variance, seasonal patterns, and time-of-day effects. The synthetic approach is explicitly documented everywhere in the codebase and README, and the scraper is preserved as an optional CLI command (`scrape-all`) for users who wish to attempt real data collection. The system architecture treats the CSV files as a replaceable data source; swapping in real data requires only matching the column schema.

This was a valuable lesson: when real data acquisition is impractical within project constraints, it is better to build a robust, honest pipeline with synthetic data than to deliver a fragile system tied to an unreliable scraping target.

### Challenge 2 — Graph Scalability

Two-interchange route search involves an O(n³) loop over all possible mid-station combinations. With 78 stations, this is approximately 474,552 possible combinations per query. To keep search response times acceptable:

- I cap results early (20 one-interchange, 30 two-interchange routes).
- The `_trains_between()` lookup short-circuits immediately if no trains serve a given station pair.
- I pre-index station adjacency at graph build time to avoid repeated full-schedule scans.

In testing, all search queries resolved in under 3 seconds. However, I am aware that scaling to the full Indian Railways network (7,000+ stations) would require a fundamentally different approach — likely a RAPTOR (Round-based Public Transit Optimized Router) algorithm or contraction hierarchies — rather than brute-force multi-hop search.

### Challenge 3 — Windows Terminal Encoding

Rich's Unicode-heavy output (emoji, box-drawing characters, Unicode progress bars) caused encoding errors on some Windows terminal configurations where the default code page is not UTF-8. I resolved this by force-reconfiguring `sys.stdout` to UTF-8 with error replacement at startup in `display.py`. This was a small issue but a good reminder that terminal environment assumptions differ significantly across operating systems.

### Challenge 4 — Designing the Ranking Formula

Deciding how to weight travel time, reliability, and simplicity in the route ranking formula required more thought than I expected. The naive approach — ranking purely by travel time — would recommend fast routes that are likely to miss connections. Ranking purely by reliability would recommend the safest route regardless of how much longer it takes.

The 40/40/20 split (time / reliability / simplicity) came from thinking about what a reasonable passenger would prioritize: time and reliability are roughly equally important, but simplicity is secondary because connection risk is already captured in the reliability score. Simplicity (fewer interchanges) gets a modest weight as a tiebreaker.

---

## 6. What I Learned

### Graph Theory Has Direct Real-World Mappings

Building this project gave me a much more concrete understanding of how graph data structures map to real-world networks. The choice between a Graph, DiGraph, and MultiDiGraph had direct consequences for what questions I could ask of the data. The Haversine-based heuristic for A* was a satisfying application of geographic distance math to guide more efficient pathfinding. These are not abstract textbook concepts — they are design decisions with real tradeoffs.

### ML Model Selection Is Context-Dependent

The most important ML insight from this project was that deep learning is not always the answer. For structured, tabular datasets of moderate size, ensemble tree methods remain highly competitive and offer practical advantages: fast training, no GPU requirement, built-in feature importance, and natural uncertainty estimation through inter-tree variance. Knowing *when* to use which class of model is as important as knowing how to use any specific model.

### System Design Requires Clean Interfaces

The project has five distinct computational layers (graph → planner → predictor → checker → recommender) that feed into each other. Keeping the interfaces between these layers clean — using typed dataclasses and clear function signatures — was essential for being able to debug, modify, and extend individual components without breaking others. This is a software engineering principle I understood theoretically before this project; I understand it practically now.

### Data Realism vs. Pragmatism

The scraping saga taught me an important and transferable lesson: when real data acquisition is impractical within project constraints, it is better to build a robust pipeline with synthetic data and document the limitation honestly than to deliver a fragile system tied to a flaky data source. Good system architecture is data-source-agnostic by design.

### Terminal UX Is Underrated

Using Rich for terminal output transformed the tool from a wall of print statements into something visually clear and enjoyable to use. Color-coding connection risk levels (green for SAFE, yellow for RISKY, red for NOT POSSIBLE), rendering route cards in bordered panels, showing spinner animations during computation, and formatting delay severity bars — these details do not affect the algorithm, but they dramatically change whether someone will actually use the tool. UX matters even in command-line software.

---

## 7. Honest Limitations

I want to be transparent about what this project is and is not:

- The dataset is **synthetic**, not derived from real Indian Railways delay records. The model's R² of ~0.87 reflects performance on data it was trained to predict; real-world performance would be lower.
- The network covers **78 stations and 64 trains** — a small subset of the actual Indian Railways network. The architecture scales, but the current data does not.
- The connection buffer thresholds (SAFE / RISKY / NOT POSSIBLE) are **heuristic**, based on general knowledge of platform transfer times, not empirically validated.
- The scraper (`scrape-all`) exists but is **not reliable** enough for production use given NTES's anti-bot protections.

These limitations are documented in the README and are not hidden from evaluators or users.

---

## 8. Summary

RailMind applies graph algorithms and supervised machine learning to a real, observable problem: helping Indian railway passengers plan safer, smarter journeys. The project demonstrates that meaningful AI applications do not require exotic architectures or massive compute — a well-chosen graph representation, a practical ML model with honest uncertainty estimation, and thoughtful system design can deliver a tool with genuine practical value.

The most important outcomes of this project are not the accuracy numbers. They are the practice of decomposing a complex real-world problem into tractable sub-problems, making and justifying technical decisions under constraints, handling the gap between ideal data and available data honestly, and building a system with clean enough architecture to be extended in the future.

---

*Submitted as part of the AI/ML course final project.*
*Author: Rishi | March 2026*
