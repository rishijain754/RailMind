# 🚆 RailMind — AI Train Route Optimizer & Delay Predictor

**RailMind** is a command-line AI system for Indian railway route planning with machine-learning-powered delay prediction. It finds optimal train routes between stations, predicts delays using a Random Forest model, and assesses whether connecting trains at interchange stations are feasible given predicted delays.

---

## Features

- **Intelligent Route Search** — Finds direct, one-interchange, and two-interchange train routes between any two stations.
- **ML Delay Prediction** — Predicts per-train, per-station delays using a Random Forest Regressor trained on historical data, with confidence scores derived from inter-tree variance.
- **Connection Feasibility** — Automatically checks whether you'll make your connecting train at each interchange, labeling connections as SAFE, RISKY, or NOT POSSIBLE.
- **Smart Ranking** — Routes are scored and ranked using a weighted formula (40% travel time + 40% reliability + 20% simplicity) and labeled as Best Route, Fastest, or Safest.
- **Rich Terminal UI** — Color-coded route cards, delay severity bars, spinner animations, and formatted tables via the [Rich](https://github.com/Textualize/rich) library.

---

## Project Structure

```
train_optimizer/
├── main.py                    # CLI entry point (Typer app with all commands)
├── requirements.txt           # Python dependencies
├── PROJECT_REPORT.md          # Detailed project report
│
├── modules/                   # Core logic
│   ├── graph_builder.py       # NetworkX MultiDiGraph construction + Haversine heuristic
│   ├── route_planner.py       # Multi-strategy route finder (direct + interchange)
│   ├── delay_predictor.py     # Random Forest training, persistence, and prediction
│   ├── connection_checker.py  # Interchange feasibility analysis
│   └── recommender.py         # Route scoring and labeling
│
├── utils/                     # Helpers
│   ├── display.py             # Rich-powered terminal rendering
│   └── data_generator.py      # Synthetic dataset generator
│
├── data/                      # Dataset files (generated)
│   ├── stations.csv           # 78 Indian railway stations with coordinates
│   ├── train_schedule.csv     # 64 trains, 235 stops, with timetable data
│   └── historical_delays.csv  # 15,280 historical delay records
│
└── models/                    # Trained ML model (auto-generated)
    ├── delay_model.pkl         # Serialized Random Forest model
    └── delay_meta.pkl          # Label encoders and metadata
```

---

## Prerequisites

- **Python 3.9+** (tested on 3.10, 3.11, 3.12)
- **pip** (Python package manager)
- A terminal with UTF-8 support (Windows Terminal, PowerShell, macOS Terminal, or any modern Linux terminal)
- typer[all]>=0.9.0
-rich>=13.0.0
-networkx>=3.0
-scikit-learn>=1.3.0
-pandas>=2.0.0
-numpy>=1.24.0
-joblib>=1.3.0
-requests>=2.31.0
-beautifulsoup4>=4.12.2
-fake-useragent>=1.2.1
-tqdm>=4.66.0

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/train-optimizer.git
cd train-optimizer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The required packages are:

| Package | Purpose |
|---|---|
| `typer[all]` | CLI framework with auto-help |
| `rich` | Terminal formatting and colors |
| `networkx` | Graph construction and traversal |
| `scikit-learn` | Random Forest model |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `joblib` | Model serialization |

---

## Quick Start

### Step 1: Generate the dataset

```bash
python main.py generate-data
```

This creates `stations.csv`, `train_schedule.csv`, and `historical_delays.csv` in the `data/` directory.

> **Note:** If the `data/` directory already contains these files, you can skip this step.

### Step 2: Train the ML model

```bash
python main.py train-model
```

This trains a Random Forest Regressor on the historical delay data and saves it to `models/`. Training takes approximately 2–5 seconds.

### Step 3: Search for routes

```bash
python main.py search --from "Mumbai Central" --to "New Delhi"
```

This will:
1. Load the railway graph
2. Find up to 5 routes (direct and with interchanges)
3. Predict delays for each leg
4. Check connection feasibility at interchange stations
5. Rank and display results with labels (Best Route, Fastest, Safest)

---

## Important: Data Limitations

> **⚠️ The dataset included in this project is a curated subset, not the complete Indian Railways network.**

- The project covers **78 major stations** out of 7,000+ stations in the real network, and **64 trains** out of 13,000+ active services.
- Historical delay data (**15,280 records**) is **synthetically generated** — not scraped from live sources. Real-world delay data from Indian Railways is not available through any public API, and the NTES website is protected against automated scraping.
- The `data/` directory may be **empty** when you first clone the repository. You **must run `python main.py generate-data`** before using any other command to populate the dataset files.
- The trained ML model (`models/delay_model.pkl`) is also **not included** in the repository due to file size (~20 MB). Run `python main.py train-model` to generate it locally.

Despite these limitations, the system architecture is designed to be **data-source-agnostic** — swapping in real data requires only replacing the CSV files with the same column schema. The algorithms, ML pipeline, and CLI interface will work identically.

---

## All Commands

### `generate-data` — Generate the dataset

```bash
python main.py generate-data
```

Creates synthetic station, schedule, and delay data in the `data/` directory.

---

### `train-model` — Train the delay prediction model

```bash
python main.py train-model          # Train only if no cached model exists
python main.py train-model --force  # Force retrain from scratch
```

Displays model metrics (MAE, RMSE, R² score) after training.

---

### `search` — Find routes between two stations

```bash
python main.py search --from "Mumbai Central" --to "New Delhi"
python main.py search --from BCT --to NDLS                        # Use station codes
python main.py search --from "Chennai" --to "Kolkata" --date 2026-04-15
python main.py search --from "Pune" --to "Ahmedabad" --max 3      # Limit results
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--from`, `-f` | Source station (name or code) | *required* |
| `--to`, `-t` | Destination station (name or code) | *required* |
| `--date`, `-d` | Travel date (YYYY-MM-DD) | Today |
| `--max`, `-n` | Maximum routes to display | 5 |

---

### `stations` — List all stations

```bash
python main.py stations                    # List all 78 stations
python main.py stations --zone WR          # Filter by railway zone
python main.py stations --query "Mumbai"   # Search by name or city
```

---

### `schedule` — View a train's timetable

```bash
python main.py schedule --train-id 12951   # Mumbai Rajdhani Express
```

---

### `predict-delay` — Predict delay for a specific train at a station

```bash
python main.py predict-delay --train-id 12951 --station NDLS --hour 8 --dow 1 --month 4
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--train-id`, `-t` | Train number | *required* |
| `--station`, `-s` | Station code | *required* |
| `--hour`, `-H` | Scheduled arrival hour (0–23) | 8 |
| `--dow`, `-w` | Day of week (0=Mon … 6=Sun) | 0 |
| `--month`, `-m` | Month (1–12) | 4 |

---

### `stats` — Show system statistics

```bash
python main.py stats
```

Displays dataset statistics (stations, trains, records), model status, and a list of all available trains.

---

## Example Output

### Route Search

```
╭──────────────────────────────────────────────────────────────╮
│   ** RAILMIND  -- AI Train Route Optimizer & Delay Predictor │
╰──────────────────────────────────────────────────────────────╯

──────── Search Results  Mumbai Central > New Delhi (2026-03-30) ────────

  Found 4 route(s). Predicting delays…

╭── [BEST] Best Route  Total: 17h 55m  Delay: 0min  Interchanges: 0 ────╮
│ Train                  ID     Type      From          Dep   To      Arr│
│ Mumbai Rajdhani Exp    12951  Rajdhani  BCT           17:00 NDLS    10:55│
│                                                                        │
╰────────────────────────────────────────────────────────────────────────╯
```

### Delay Prediction

```
╭────────────────── Delay Prediction ──────────────────╮
│  Mumbai Rajdhani Express (12951) @ NDLS              │
│                                                       │
│  Predicted Delay  : 8.3 minutes                       │
│  Confidence       : 82%                               │
│  Severity bar     : ||||                              │
╰──────────────────────────────────────────────────────╯
```

---

## How It Works

### 1. Graph Construction

The railway network is modeled as a **NetworkX MultiDiGraph**:
- **Nodes** = 78 railway stations (with lat/lon coordinates)
- **Edges** = train segments between consecutive stops (one edge per train per segment)
- Multiple trains on the same route segment exist as parallel edges

### 2. Route Search

The route planner uses a multi-strategy approach:
1. **Direct trains** — scans the schedule for trains serving both stations in order
2. **One-interchange routes** — iterates over all possible mid-stations and validates connection timing (minimum 20-minute buffer)
3. **Two-interchange routes** — extends to three-leg journeys if fewer than *k* routes found

### 3. Delay Prediction

A **Random Forest Regressor** (120 trees, max depth 12) is trained on historical delay data with features:
- Encoded train ID, station code, and railway zone
- Scheduled arrival hour, day of week, and month
- Historical average delay for that station

**Confidence** is computed from the standard deviation of predictions across all 120 trees — low variance means high agreement, thus high confidence.

### 4. Connection Feasibility

For each interchange, the system:
1. Predicts the arriving train's delay
2. Computes `buffer = next_train_departure - (scheduled_arrival + predicted_delay)`
3. Labels: **SAFE** (>30 min), **RISKY** (15–30 min), **NOT POSSIBLE** (≤15 min)

### 5. Route Ranking

Routes are scored: `0.4 × time_score + 0.4 × reliability_score + 0.2 × simplicity_score` and labeled as Best Route, Fastest, or Safest.

---

## Tech Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.9+ | Core language |
| NetworkX | ≥ 3.0 | Graph data structure and algorithms |
| scikit-learn | ≥ 1.3 | Random Forest model |
| Pandas | ≥ 2.0 | Data manipulation |
| NumPy | ≥ 1.24 | Numerical operations |
| Typer | ≥ 0.9 | CLI framework |
| Rich | ≥ 13.0 | Terminal UI rendering |
| joblib | ≥ 1.3 | Model serialization |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `FileNotFoundError: Data files not found` | Run `python main.py generate-data` first |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Unicode/emoji rendering issues | Use Windows Terminal or PowerShell 7+. Set `chcp 65001` for UTF-8. |
| Model not found | Run `python main.py train-model` to train the Random Forest |
| Station not found | Use `python main.py stations` to see valid station names/codes |

---

## License

This project was created as a course project. Feel free to use and modify it for educational purposes.

---

## Author

**Rishi**
March 2026
