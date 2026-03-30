"""
display.py -- Rich-based terminal rendering for the Train Optimizer CLI.
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import List

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from modules.route_planner import Route, Leg
from modules.connection_checker import ConnectionStatus
from modules.recommender import RankedRoute

console = Console(highlight=False)

# ─── PALETTE ──────────────────────────────────────────────────────────────────
C_BRAND   = "bold cyan"
C_SAFE    = "bold green"
C_RISKY   = "bold yellow"
C_DANGER  = "bold red"
C_DIM     = "dim white"
C_GOLD    = "bold yellow"
C_HEADER  = "bold white on dark_blue"


def _status_style(status: str) -> str:
    return {
        "SAFE":         C_SAFE,
        "RISKY":        C_RISKY,
        "NOT POSSIBLE": C_DANGER,
    }.get(status, C_DIM)


def _type_color(ttype: str) -> str:
    return {
        "Rajdhani": "bright_red",
        "Shatabdi": "bright_cyan",
        "Duronto":  "bright_magenta",
        "Superfast":"bright_yellow",
        "Express":  "bright_white",
        "Mail":     "bright_blue",
    }.get(ttype, "white")


# ─── BANNER ───────────────────────────────────────────────────────────────────

def print_banner():
    banner = Text()
    banner.append("  ** ", style="bold yellow")
    banner.append("RAIL", style="bold bright_cyan")
    banner.append("MIND", style="bold bright_white")
    banner.append("  -- AI Train Route Optimizer & Delay Predictor **", style="dim white")
    console.print(Panel(Align.center(banner), border_style="cyan", padding=(0, 2)))
    console.print()


# ─── STATION TABLE ────────────────────────────────────────────────────────────

def print_stations(station_list: list):
    t = Table(
        title="[bold cyan]Indian Railway Stations[/]",
        box=box.ROUNDED,
        show_lines=False,
        border_style="cyan",
        header_style=C_HEADER,
    )
    t.add_column("Code",    style="bold cyan",   no_wrap=True, width=7)
    t.add_column("Name",    style="white",        width=28)
    t.add_column("City",    style="dim white",    width=18)
    t.add_column("Zone",    style="bright_yellow",width=7)
    t.add_column("State",   style="dim white",    width=20)

    for s in sorted(station_list, key=lambda x: x["station_code"]):
        t.add_row(
            s["station_code"],
            s["station_name"],
            s["city"],
            s["zone"],
            s["state"],
        )
    console.print(t)


# ─── SCHEDULE TABLE ───────────────────────────────────────────────────────────

def print_schedule(train_id: int, train_name: str, rows: list):
    t = Table(
        title=f"[bold cyan]{train_name}  ({train_id})[/]",
        box=box.SIMPLE_HEAVY,
        border_style="bright_blue",
        header_style=C_HEADER,
    )
    t.add_column("#",        style="dim",         width=3)
    t.add_column("Station",  style="bold white",  width=28)
    t.add_column("Code",     style="cyan",        width=7)
    t.add_column("Arrival",  style="green",       width=8)
    t.add_column("Departure",style="yellow",      width=10)

    for i, r in enumerate(rows):
        arr = r.get("arrival_time",   "--")
        dep = r.get("departure_time", "--")
        t.add_row(
            str(i + 1),
            str(r.get("station_name", r.get("station_code", "?"))),
            str(r.get("station_code", "")),
            str(arr) if arr else "--",
            str(dep) if dep else "--",
        )
    console.print(t)


# ─── ROUTE CARDS ──────────────────────────────────────────────────────────────

def print_search_header(src_name: str, dst_name: str, date: str):
    console.print(Rule(style="cyan"))
    console.print(
        f"  [bold cyan]Search Results[/]  "
        f"[white]{src_name}[/]  [dim]>[/]  [white]{dst_name}[/]  "
        f"[dim]({date})[/]"
    )
    console.print(Rule(style="cyan"))
    console.print()


def _leg_table(legs: List[Leg]) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim white", padding=(0, 1))
    t.add_column("Train",    style="white",  width=30)
    t.add_column("ID",       style="cyan",   width=7)
    t.add_column("Type",     width=10)
    t.add_column("From",     style="white",  width=22)
    t.add_column("Dep",      style="yellow", width=7)
    t.add_column("To",       style="white",  width=22)
    t.add_column("Arr",      style="green",  width=7)
    t.add_column("Duration", style="dim",    width=10)

    for leg in legs:
        h, m = divmod(leg.travel_min, 60)
        dur  = f"{h}h {m:02d}m"
        col  = _type_color(leg.train_type)
        t.add_row(
            leg.train_name,
            str(leg.train_id),
            Text(leg.train_type, style=col),
            f"{leg.from_name} ({leg.from_station})",
            leg.dep_time,
            f"{leg.to_name} ({leg.to_station})",
            leg.arr_time,
            dur,
        )
    return t


def _connection_lines(conns: List[ConnectionStatus]) -> str:
    if not conns:
        return ""
    lines = []
    for c in conns:
        color = _status_style(c.status)
        tag   = {"SAFE": "[SAFE]", "RISKY": "[RISKY]", "NOT POSSIBLE": "[!!!]"}.get(c.status, "")
        lines.append(
            f"  {tag} [{color}]{c.status}[/]  "
            f"[dim]@[/] [white]{c.interchange_station}[/]  "
            f"[dim]Sched arr:[/] {c.scheduled_arrival}  "
            f"[dim]+delay[/] [yellow]{c.predicted_delay:.0f}min[/]  "
            f"[dim]pred arr:[/] {c.predicted_arrival}  "
            f"[dim]Next dep:[/] {c.next_departure}  "
            f"[dim]Buffer:[/] [{color}]{c.window_min:.0f}min[/]  "
            f"[dim]Conf:[/] {int(c.confidence * 100)}%"
        )
    return "\n".join(lines)


def print_ranked_route(rr: RankedRoute, show_conn: bool = True):
    route = rr.route
    label_str = "  ".join(rr.labels)
    h, m      = divmod(route.total_min, 60)
    total_str = f"{h}h {m:02d}m"

    title = (
        f"{label_str}   "
        f"[dim]Total:[/] [bold white]{total_str}[/]   "
        f"[dim]Delay:[/] [yellow]{rr.total_delay_min:.0f}min[/]   "
        f"[dim]Interchanges:[/] [white]{route.num_interchanges}[/]   "
        f"[dim]Score:[/] [cyan]{rr.score}[/]"
    )

    leg_tbl = _leg_table(route.legs)
    conn_txt = _connection_lines(rr.connections) if show_conn and rr.connections else ""

    border = {
        "SAFE":         "green",
        "RISKY":        "yellow",
        "NOT POSSIBLE": "red",
    }.get(rr.overall_status, "cyan")

    if conn_txt:
        from rich.console import Group
        content = Group(
            leg_tbl,
            Text.from_markup("\n[bold dim]Connection Feasibility:[/]"),
            Text.from_markup(conn_txt),
        )
    else:
        content = leg_tbl

    console.print(Panel(content, title=title, border_style=border, padding=(0, 1)))
    console.print()


def print_ranked_routes(ranked: List[RankedRoute]):
    if not ranked:
        console.print(Panel(
            "[bold red]No routes found.[/]\n"
            "Try different stations or use: python main.py stations",
            border_style="red",
        ))
        return
    for rr in ranked:
        print_ranked_route(rr)


# ─── DELAY PREDICTION ─────────────────────────────────────────────────────────

def print_delay_prediction(
    train_id: int,
    train_name: str,
    station: str,
    delay: float,
    confidence: float,
):
    color = C_SAFE if delay < 10 else (C_RISKY if delay < 30 else C_DANGER)
    bar   = "|" * int(min(delay / 2, 30))

    content = (
        f"  [bold white]{train_name}[/] [dim]({train_id})[/] @ [cyan]{station}[/]\n\n"
        f"  Predicted Delay  : [{color}]{delay:.1f} minutes[/]\n"
        f"  Confidence       : [white]{int(confidence * 100)}%[/]\n"
        f"  Severity bar     : [{color}]{bar}[/]\n"
    )
    console.print(Panel(content, title="[bold cyan]Delay Prediction[/]", border_style="cyan"))


# ─── ML METRICS ──────────────────────────────────────────────────────────────

def print_model_metrics(metrics: dict, loaded_from_cache: bool):
    source = "[dim]loaded from cache[/]" if loaded_from_cache else "[yellow]freshly trained[/]"
    mae_v  = metrics.get("mae")
    rmse_v = metrics.get("rmse")
    r2_v   = metrics.get("r2")

    content = (
        f"  Model    : [bold cyan]Random Forest Regressor (120 trees)[/] {source}\n"
        f"  MAE      : [green]{f'{mae_v:.2f} min' if mae_v else 'N/A'}[/]\n"
        f"  RMSE     : [yellow]{f'{rmse_v:.2f} min' if rmse_v else 'N/A'}[/]\n"
        f"  R2 Score : [cyan]{f'{r2_v:.4f}' if r2_v else 'N/A'}[/]\n"
    )
    console.print(Panel(content, title="[bold cyan]Model Metrics[/]", border_style="cyan"))


# ─── SYSTEM STATS ────────────────────────────────────────────────────────────

def print_stats(stats: dict):
    t = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    t.add_column("Metric", style="bold cyan",  width=28)
    t.add_column("Value",  style="bold white", width=12)
    for key, val in stats.items():
        t.add_row(key, str(val))
    console.print(Panel(t, title="[bold cyan]System Stats[/]", border_style="cyan"))
