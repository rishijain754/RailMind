"""
main.py — RailMind CLI entry point
Usage:
    python main.py --help
    python main.py generate-data
    python main.py train-model
    python main.py search --from-station "Mumbai Central" --to-station "New Delhi"
    python main.py stations
    python main.py schedule --train-id 12951
    python main.py predict-delay --train-id 12951 --station NDLS --hour 8 --dow 1 --month 4
    python main.py stats
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

# ── Path setup so sub-packages resolve correctly ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_generator import generate_all
from utils.display import (
    console,
    print_banner,
    print_stations,
    print_schedule,
    print_search_header,
    print_ranked_routes,
    print_delay_prediction,
    print_model_metrics,
    print_stats,
)
from modules.graph_builder   import RailwayGraph
from modules.route_planner   import RoutePlanner
from modules.delay_predictor import DelayPredictor
from modules.connection_checker import ConnectionChecker
from modules.recommender     import Recommender

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── APP ──────────────────────────────────────────────────────────────────────
app = typer.Typer(
    name="railmind",
    help="🚆 RailMind — AI Train Route Optimizer & Delay Predictor",
    add_completion=False,
    rich_markup_mode="rich",
)


# ─── SHARED HELPERS ───────────────────────────────────────────────────────────

def _build_graph() -> RailwayGraph:
    rg = RailwayGraph(DATA_DIR)
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Loading railway graph…"),
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        rg.build()
    return rg


def _build_predictor(force: bool = False) -> tuple[DelayPredictor, bool]:
    predictor = DelayPredictor(MODEL_DIR, DATA_DIR)
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Loading ML model…"),
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        loaded = predictor.load_or_train(force=force)
    return predictor, loaded


def _resolve_station(rg: RailwayGraph, query: str) -> Optional[str]:
    """Match station code or partial name (case-insensitive)."""
    q = query.strip().upper()
    # Exact code match
    if q in rg.stations:
        return q
    # Partial name match
    for code, info in rg.stations.items():
        if q in info["station_name"].upper() or q in info["city"].upper():
            return code
    return None


# ─── COMMANDS ─────────────────────────────────────────────────────────────────

@app.command("generate-data")
def cmd_generate_data():
    """Generate the simulated railway dataset (stations, schedules, delays)."""
    print_banner()
    console.print("[cyan]Generating dataset…[/]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=False,
    ) as prog:
        task = prog.add_task("Building stations & schedules…", total=3)
        os.makedirs(DATA_DIR, exist_ok=True)
        stats = generate_all(DATA_DIR)
        prog.advance(task, 3)

    console.print(
        f"\n[bold green]✅ Dataset generated:[/]  "
        f"[cyan]{stats['stations']}[/] stations, "
        f"[cyan]{stats['trains']}[/] trains, "
        f"[cyan]{stats['stops']}[/] stops, "
        f"[cyan]{stats['records']:,}[/] historical records\n"
    )
    console.print(f"  Files saved to [dim]{DATA_DIR}[/]\n")


@app.command("train-model")
def cmd_train_model(
    force: bool = typer.Option(False, "--force", "-f", help="Force retrain even if model exists"),
):
    """Train (or retrain) the Random Forest delay prediction model."""
    print_banner()
    predictor = DelayPredictor(MODEL_DIR, DATA_DIR)

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        transient=False,
    ) as prog:
        task = prog.add_task("Training Random Forest…", total=None)
        metrics = predictor.retrain() if force else None
        if metrics is None:
            loaded = predictor.load_or_train(force=True)
            metrics = predictor.metrics
        prog.update(task, description="[green]Done!")

    print_model_metrics(metrics, loaded_from_cache=False)


@app.command("search")
def cmd_search(
    from_station: str = typer.Option(..., "--from", "-f",  help="Source station code or name"),
    to_station:   str = typer.Option(..., "--to",   "-t",  help="Destination station code or name"),
    date:         str = typer.Option(
        datetime.today().strftime("%Y-%m-%d"),
        "--date", "-d",
        help="Travel date YYYY-MM-DD",
    ),
    max_routes:   int = typer.Option(5, "--max", "-n", help="Max routes to show"),
):
    """🔍 Search for train routes between two stations with delay predictions."""
    print_banner()

    # Build graph
    rg = _build_graph()

    # Resolve stations
    src_code = _resolve_station(rg, from_station)
    dst_code = _resolve_station(rg, to_station)

    if not src_code:
        console.print(f"[red]Station not found:[/] {from_station}")
        console.print("[dim]Use `python main.py stations` to see available stations.[/]")
        raise typer.Exit(1)
    if not dst_code:
        console.print(f"[red]Station not found:[/] {to_station}")
        console.print("[dim]Use `python main.py stations` to see available stations.[/]")
        raise typer.Exit(1)

    src_info = rg.station_info(src_code)
    dst_info = rg.station_info(dst_code)

    print_search_header(
        src_info["station_name"],
        dst_info["station_name"],
        date,
    )

    # Parse date for ML features
    try:
        travel_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        travel_date = datetime.today()
    dow   = travel_date.weekday()
    month = travel_date.month

    # Find routes
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Searching routes…"),
        transient=True,
    ) as prog:
        prog.add_task("", total=None)
        planner = RoutePlanner(rg)
        routes  = planner.find_routes(src_code, dst_code, max_routes=max_routes)

    if not routes:
        console.print(
            "[bold red]No routes found.[/]\n"
            "The stations may not be directly or indirectly connected in this dataset.\n"
            "Try different stations."
        )
        raise typer.Exit(0)

    console.print(f"  Found [bold cyan]{len(routes)}[/] route(s). Predicting delays…\n")

    # Load ML model
    predictor, loaded = _build_predictor()
    checker    = ConnectionChecker(predictor, rg)
    recommender = Recommender()

    connections_per_route = []
    for route in routes:
        conns = checker.check_route(route, day_of_week=dow, month=month)
        connections_per_route.append(conns)

    ranked = recommender.rank(routes, connections_per_route)
    print_ranked_routes(ranked)

    # Summary footer
    console.print(
        f"  [dim]Model confidence avg:[/] "
        f"[cyan]{int(sum(r.avg_confidence for r in ranked) / len(ranked) * 100)}%[/]    "
        f"[dim]Date:[/] [white]{date}[/] "
        f"([dim]Mon=0 … Sun=6, today is day {dow}[/])\n"
    )


@app.command("stations")
def cmd_stations(
    zone: Optional[str] = typer.Option(None, "--zone", "-z", help="Filter by railway zone"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search by name or city"),
):
    """📋 List all available stations."""
    print_banner()
    rg = _build_graph()
    stations = rg.station_list()

    if zone:
        stations = [s for s in stations if s["zone"].upper() == zone.upper()]
    if query:
        q = query.upper()
        stations = [
            s for s in stations
            if q in s["station_name"].upper() or q in s["city"].upper() or q in s["station_code"].upper()
        ]

    print_stations(stations)
    console.print(f"\n  [dim]Total: {len(stations)} stations[/]\n")


@app.command("schedule")
def cmd_schedule(
    train_id: int = typer.Option(..., "--train-id", "-t", help="Train number (e.g. 12951)"),
):
    """🗓️  Show the complete timetable for a specific train."""
    print_banner()
    rg = _build_graph()
    sched = rg.train_schedule(train_id)

    if sched is None or sched.empty:
        console.print(f"[red]Train {train_id} not found in schedule.[/]")
        console.print(
            f"  Available trains: [cyan]{', '.join(str(t) for t in rg.all_train_ids())}[/]"
        )
        raise typer.Exit(1)

    train_name = sched.iloc[0]["train_name"]
    rows = []
    for _, row in sched.iterrows():
        code = row["station_code"]
        info = rg.station_info(code)
        name = info["station_name"] if info else code
        rows.append({
            "station_name":   name,
            "station_code":   code,
            "arrival_time":   row["arrival_time"],
            "departure_time": row["departure_time"],
        })

    print_schedule(train_id, train_name, rows)


@app.command("predict-delay")
def cmd_predict_delay(
    train_id: int = typer.Option(..., "--train-id", "-t",  help="Train ID (e.g. 12951)"),
    station:  str = typer.Option(..., "--station",  "-s",  help="Station code (e.g. NDLS)"),
    hour:     int = typer.Option(8,   "--hour",     "-H",  help="Scheduled arrival hour 0-23"),
    dow:      int = typer.Option(0,   "--dow",      "-w",  help="Day of week 0=Mon … 6=Sun"),
    month:    int = typer.Option(4,   "--month",    "-m",  help="Month 1-12"),
):
    """🔮 Predict delay for a specific train at a station."""
    print_banner()
    rg = _build_graph()
    info = rg.station_info(station.upper())
    zone = info["zone"] if info else "NR"
    train_name = "Unknown Train"
    sched = rg.train_schedule(train_id)
    if sched is not None and not sched.empty:
        train_name = sched.iloc[0]["train_name"]

    predictor, _ = _build_predictor()
    delay, confidence = predictor.predict(
        train_id=train_id,
        station_code=station.upper(),
        scheduled_hour=hour,
        day_of_week=dow,
        month=month,
        zone=zone,
    )
    print_delay_prediction(train_id, train_name, station.upper(), delay, confidence)


@app.command("stats")
def cmd_stats():
    """📈 Show system statistics."""
    print_banner()

    # Data stats
    import pandas as pd
    data_stats: dict = {}
    sch_path = os.path.join(DATA_DIR, "train_schedule.csv")
    stn_path = os.path.join(DATA_DIR, "stations.csv")
    del_path = os.path.join(DATA_DIR, "historical_delays.csv")

    if os.path.exists(stn_path):
        df = pd.read_csv(stn_path)
        data_stats["Total Stations"] = len(df)
        data_stats["Railway Zones"]  = df["zone"].nunique()
    if os.path.exists(sch_path):
        df = pd.read_csv(sch_path)
        data_stats["Total Trains"]   = df["train_id"].nunique()
        data_stats["Total Stops"]    = len(df)
        data_stats["Train Types"]    = df["train_type"].nunique()
    if os.path.exists(del_path):
        df = pd.read_csv(del_path)
        data_stats["Historical Records"]   = f"{len(df):,}"
        data_stats["Avg Delay (dataset)"]  = f"{df['actual_delay_min'].mean():.1f} min"
        data_stats["Max Delay (dataset)"]  = f"{df['actual_delay_min'].max():.0f} min"

    model_path = os.path.join(MODEL_DIR, "delay_model.pkl")
    data_stats["ML Model Cached"] = "Yes" if os.path.exists(model_path) else "No"

    print_stats(data_stats)

    # Print available train IDs
    if os.path.exists(sch_path):
        import pandas as pd
        df = pd.read_csv(sch_path)
        trains = df.groupby("train_id")["train_name"].first().reset_index()
        console.print("\n  [bold cyan]Available Trains:[/]")
        for _, row in trains.iterrows():
            console.print(f"  [cyan]{int(row['train_id'])}[/]  {row['train_name']}")
        console.print()


@app.command("scrape-all")
def cmd_scrape_all():
    """🌐 Scrape NTES for schedules of all trains between stations in stations.csv."""
    print_banner()
    console.print("[bold yellow]WARNING:[/] This job will scrape over 3,000 station combinations.")
    console.print("It has built-in delays to avoid IP bans and will take ~2 hours.")
    console.print("Progress is saved to a checkpoint file so you can safely Ctrl+C.")
    console.print()
    
    confirm = typer.confirm("Do you want to proceed?")
    if not confirm:
        console.print("[red]Aborted.[/]")
        raise typer.Exit(0)
        
    try:
        from utils.ntes_scraper import TrainDataScraper
        scraper = TrainDataScraper(data_dir=DATA_DIR)
        scraper.run_full_scrape()
        console.print("\n[bold green]✅ Scraping run completed.[/]")
    except Exception as e:
        console.print(f"\n[bold red]Scraper failed:[/] {str(e)}")
        raise typer.Exit(1)

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
