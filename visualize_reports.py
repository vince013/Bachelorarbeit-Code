#!/usr/bin/env python3
"""Erzeugt die Abbildungen aus den geprüften Analyse- und Benchmarkberichten."""

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bachelorarbeit_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "analysis_report.json"
DEFAULT_BENCHMARK_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "benchmark_summary.csv"
DEFAULT_BENCHMARK_ENVIRONMENT_PATH = PROJECT_ROOT / "results" / "raw" / "benchmark_environment.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
EXPECTED_REPORT_SCHEMA_VERSION = "1.1"
REPORTED_GROUPINGS = ("sex", "race", "sex_race")

TASK_PRESENTATION = {
    "acs_income": {
        "name": "ACSIncome",
        "color": "#31688e",
    },
    "acs_employment": {
        "name": "ACSEmployment",
        "color": "#26828e",
    },
    "acs_public_coverage": {
        "name": "ACSPublicCoverage",
        "color": "#d17c2f",
    },
}

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


def parse_args():
    """Liest die Kommandozeile für die Erzeugung der Abbildungen ein."""

    parser = argparse.ArgumentParser(description="Erzeugt Abbildungen aus dem geprüften JSON Bericht.")
    return parser.parse_args()


def load_report(path):
    """Lädt den Analysebericht und prüft Schema, Validierungsstatus und Herkunftsangaben."""

    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}\nRun fairness_auswertung.py report first.")
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError("The analysis report must contain one JSON object.")
    if report.get("report_schema_version") != EXPECTED_REPORT_SCHEMA_VERSION:
        raise ValueError(f"Expected report schema {EXPECTED_REPORT_SCHEMA_VERSION!r}.")
    if report.get("status") != "SQL and Pandas results agree":
        raise ValueError("Figures may only be created from a successfully validated report.")
    if not isinstance(report.get("provenance"), dict):
        raise ValueError("The analysis report lacks verified provenance.")
    required_provenance = {"data_source", "preparation", "prepared_data_file", "software", "database", "verification"}
    missing = sorted(required_provenance.difference(report["provenance"]))
    if missing:
        raise ValueError(f"The analysis report lacks provenance sections: {missing}")
    if set(report.get("tasks", {})) != set(TASK_PRESENTATION):
        raise ValueError("The visualization expects all three confirmed analysis tasks.")
    return report


def format_count(value):
    """Formatiert eine ganze Zahl mit Punkten als Tausendertrennzeichen."""

    return f"{int(value):,}".replace(",", ".")


def format_decimal(value, digits=3):
    """Formatiert eine Zahl mit festgelegten Nachkommastellen und deutschem Dezimalkomma."""

    return f"{value:.{digits}f}".replace(".", ",")


def annotation_color(cmap, value):
    """Wählt schwarze oder weiße Beschriftung passend zur Helligkeit der Zellenfarbe."""

    red, green, blue, _ = cmap(value)
    return "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.55 else "white"


def plot_descriptive_figures(report, report_path, output_dir):
    """Erzeugt die Abbildungen für Abschnitt 4.2 aus geprüften Summen."""

    source = report["provenance"]["data_source"]
    preparation = report["provenance"]["preparation"]
    if (source["states"], source["survey_year"], source["horizon"], source["survey"]) != (["CA"], "2018", "1-Year", "person"):
        raise ValueError("The thesis figures are specified for California, ACS PUMS 2018, 1-Year, person.")
    if preparation["analysis_weighting"] != "unweighted":
        raise ValueError("The thesis figures require an unweighted analysis.")
    selections = {}
    for task_id, task in report["tasks"].items():
        for grouping, expected_codes in [("sex", {1, 2}), ("race", set(range(1, 10)))]:
            rows = [row for row in task["group_statistics"] if row["grouping_name"] == grouping]
            if len(rows) != len(expected_codes) or {row["group_code_1"] for row in rows} != expected_codes:
                raise ValueError(f"Unexpected observed or missing groups: {task_id}/{grouping}")
            if sum(row["group_size"] for row in rows) != task["rows"]:
                raise ValueError(f"Group sizes do not sum to task size: {task_id}/{grouping}")
            for row in rows:
                n = row["group_size"]
                if n <= 0 or row["positive_labels"] + row["negative_labels"] != n:
                    raise ValueError("Invalid group counts in report.")
                if not math.isclose(row["representation_rate"], n / task["rows"], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError("Group representation is inconsistent with counts.")
                if not math.isclose(row["positive_label_rate"], row["positive_labels"] / n, abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError("Positive label rate is inconsistent with counts.")
            selections[task_id, grouping] = {int(row["group_code_1"]): row for row in rows}

    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("ca_race_representation.pdf", "race", "representation_rate", 5.7),
        ("ca_sex_label_rates.pdf", "sex", "positive_label_rate", 3.6),
    ]
    label_lines = {
        "acs_income": "Positiv: Einkommen\nüber 50.000 US-Dollar",
        "acs_employment": "Positiv: zivil erwerbstätig\nund aktuell arbeitend",
        "acs_public_coverage": "Positiv: öffentliche\nKrankenversicherung",
    }
    race_labels = ["White alone (1)", "Black or African\nAmerican alone (2)", "American Indian alone (3)", "Alaska Native alone (4)", "Weitere AI/AN-\nZuordnungen (5)", "Asian alone (6)", "Native Hawaiian and Other\nPacific Islander alone (7)", "Some Other Race alone (8)", "Two or More Races (9)"]
    manifest = {
        "source_report": str(report_path.resolve().relative_to(PROJECT_ROOT)) if report_path.resolve().is_relative_to(PROJECT_ROOT) else str(report_path.resolve()),
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "prepared_csv_sha256": report["provenance"]["prepared_data_file"]["sha256"],
        "scope": source, "analysis_weighting": "unweighted",
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "pandas": pd.__version__},
        "figures": [],
    }
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.axisbelow": True}
    with plt.rc_context(style):
        for filename, grouping, metric, height in definitions:
            figure, axes = plt.subplots(1, 3, figsize=(6.3, height))
            if grouping == "race":
                figure.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.84, wspace=0.18)
            else:
                figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.64, wspace=0.22)
            records = []
            for task_index, (axis, task_id) in enumerate(zip(axes, TASK_PRESENTATION, strict=True)):
                presentation = TASK_PRESENTATION[task_id]
                color = presentation["color"]
                rows = selections[task_id, grouping]
                task_size = report["tasks"][task_id]["rows"]
                axis.set_title(presentation["name"], pad=37 if grouping != "race" else 23, fontsize=9)
                context = f"N = {format_count(task_size)}"
                if grouping != "race":
                    context = label_lines[task_id] + "\n" + context
                axis.text(0.5, 1.015, context, transform=axis.transAxes, ha="center", va="bottom", fontsize=8, linespacing=1.35)
                axis.spines["left"].set_visible(False)
                axis.tick_params(axis="both", length=0)
                if grouping == "sex":
                    for index, code in enumerate([1, 2]):
                        row = rows[code]
                        value = row[metric] * 100
                        axis.bar(index, value, color=color, width=0.55, edgecolor="white", linewidth=0.4)
                        axis.text(index, value + 3, format_decimal(value, 2) + " %", ha="center", va="bottom", fontsize=8)
                    axis.set_xticks([0, 1], [f"{label}\nn = {format_count(rows[code]['group_size'])}" for code, label in [(1, "Männlich"), (2, "Weiblich")]], fontsize=8, linespacing=1.5)
                    axis.set_ylim(0, 100)
                    axis.set_yticks([0, 25, 50, 75, 100])
                    axis.grid(axis="y", color="#dedede", linewidth=0.5)
                    if task_index == 0:
                        axis.set_ylabel("Positive Labelrate (%)")
                    else:
                        axis.tick_params(labelleft=False)
                else:
                    codes = list(range(1, 10))
                    axis.set_xlim(0, 100)
                    axis.set_xticks([0, 50, 100])
                    axis.set_ylim(len(codes) - 0.35, -0.75)
                    axis.set_yticks(range(len(codes)), race_labels)
                    axis.grid(axis="x", color="#dedede", linewidth=0.5)
                    if task_index:
                        axis.tick_params(labelleft=False)
                    for index, code in enumerate(codes):
                        row = rows[code]
                        value = row[metric] * 100
                        axis.barh(index - 0.12, value, height=0.22, color=color)
                        axis.plot(value, index - 0.12, "o", markersize=3, color=color, clip_on=False)
                        axis.text(0, index + 0.21, f"n = {format_count(row['group_size'])}", ha="left", va="center", fontsize=8.2, color="#444444")
                    if task_index == 1:
                        axis.set_xlabel("Gruppenanteil (%)", labelpad=8)
                for code, row in rows.items():
                    records.append({"task_id": task_id, "grouping": grouping, "group_code": code, "group_size": row["group_size"], "positive_labels": row["positive_labels"], "negative_labels": row["negative_labels"], "metric": metric, "value": row[metric]})

            footer = "Kalifornien | ACS PUMS 2018, 1-Year | ungewichtet | eigene Darstellung"
            figure.text(0.02, 0.02, footer, fontsize=7, color="#555555", ha="left")
            output_path = output_dir / filename
            figure.savefig(output_path, format="pdf", metadata={"Creator": "visualize_reports.py", "CreationDate": None, "ModDate": None})
            plt.close(figure)
            manifest["figures"].append({"file": filename, "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "grouping": grouping, "metric": metric, "axis_percent_limits": [0, 100], "plotted_groups": records})
    (output_dir / "ca_descriptive_figures.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


EXPECTED_CODES = {
    "sex_race": {task_id: {(sex, race) for sex in (1, 2) for race in range(1, 10)} for task_id in TASK_PRESENTATION},
}
RACE_LABELS = [
    "White alone (1)", "Black or African\nAmerican alone (2)", "American Indian\nalone (3)",
    "Alaska Native\nalone (4)", "Weitere AI/AN-\nZuordnungen (5)", "Asian alone (6)",
    "Native Hawaiian/\nOther Pacific (7)", "Some Other Race\nalone (8)", "Two or More\nRaces (9)",
]
TASK_SUBTITLES = {
    "acs_income": "Positiv: Einkommen\n> 50.000 US-Dollar",
    "acs_employment": "Positiv: zivil\nerwerbstätig und\naktuell arbeitend",
    "acs_public_coverage": "Positiv: öffentliche\nKrankenversicherung",
}


def check_results(report):
    """Prüft Gruppenstatistiken, Paarmetriken und Mehrgruppenwerte des Analyseberichts."""

    source = report["provenance"]["data_source"]
    preparation = report["provenance"]["preparation"]
    if (source["states"], source["survey_year"], source["horizon"], source["survey"]) != (["CA"], "2018", "1-Year", "person"):
        raise ValueError("Section 4.3 figures require California ACS PUMS 2018, 1-Year, person.")
    if preparation["analysis_weighting"] != "unweighted":
        raise ValueError("Section 4.3 figures require an unweighted analysis.")

    selected = {}
    metrics = []
    for task_id, task in report["tasks"].items():
        by_grouping = {}
        for row in task["group_statistics"]:
            grouping = row["grouping_name"]
            key = (row["group_code_1"],) if row["group_code_2"] is None else (row["group_code_1"], int(row["group_code_2"]))
            if row["group_size"] <= 0 or row["positive_labels"] + row["negative_labels"] != row["group_size"]:
                raise ValueError(f"Invalid label counts in {task_id}/{grouping}/{key}.")
            if not math.isclose(row["positive_label_rate"], row["positive_labels"] / row["group_size"], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Invalid positive label rate in {task_id}/{grouping}/{key}.")
            by_grouping.setdefault(grouping, {})[key] = row
        for grouping, groups in by_grouping.items():
            if sum(row["group_size"] for row in groups.values()) != task["rows"]:
                raise ValueError(f"Incomplete partition in {task_id}/{grouping}.")
            result = task["multigroup_metrics"][grouping]
            rates = {key: row["positive_label_rate"] for key, row in groups.items()}
            lo, hi = min(rates.values()), max(rates.values())
            minimum = sorted([list(key) for key, rate in rates.items() if math.isclose(rate, lo, abs_tol=1e-12, rel_tol=1e-12)])
            maximum = sorted([list(key) for key, rate in rates.items() if math.isclose(rate, hi, abs_tol=1e-12, rel_tol=1e-12)])
            if result["included_group_count"] != len(groups) or result["minimum_groups"] != minimum or result["maximum_groups"] != maximum:
                raise ValueError(f"Wrong multigroup extremes in {task_id}/{grouping}.")
            if not math.isclose(result["demographic_parity_difference"], hi - lo, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Wrong DPD in {task_id}/{grouping}.")
            if not math.isclose(result["demographic_parity_ratio"], lo / hi, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Wrong DPR in {task_id}/{grouping}.")
            pairwise = task["pairwise_metrics"][grouping]
            if len(pairwise) != len(groups) * (len(groups) - 1) // 2:
                raise ValueError(f"Wrong pair count in {task_id}/{grouping}.")
            for pair in pairwise:
                ref = (pair["reference_code_1"],) if pair["reference_code_2"] is None else (pair["reference_code_1"], int(pair["reference_code_2"]))
                cmp = (pair["comparison_code_1"],) if pair["comparison_code_2"] is None else (pair["comparison_code_1"], int(pair["comparison_code_2"]))
                a, b = groups[ref], groups[cmp]
                if not ref < cmp or pair["reference_group_size"] != a["group_size"] or pair["comparison_group_size"] != b["group_size"]:
                    raise ValueError(f"Wrong pair direction or group size in {task_id}/{grouping}.")
                if not math.isclose(pair["statistical_parity_difference"], b["positive_label_rate"] - a["positive_label_rate"], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError(f"Wrong SPD in {task_id}/{grouping}.")
                expected_dir = b["positive_label_rate"] / a["positive_label_rate"] if a["positive_label_rate"] else None
                actual_dir = pair["disparate_impact_ratio"]
                if (expected_dir is None) != (actual_dir is None) or (expected_dir is not None and not math.isclose(actual_dir, expected_dir, abs_tol=1e-12, rel_tol=1e-12)):
                    raise ValueError(f"Wrong DIR in {task_id}/{grouping}.")
            if grouping in REPORTED_GROUPINGS:
                metrics.append({"task_id": task_id, "grouping": grouping, **result})
        for grouping in EXPECTED_CODES:
            if set(by_grouping[grouping]) != EXPECTED_CODES[grouping][task_id]:
                raise ValueError(f"Unexpected intersection groups in {task_id}/{grouping}.")
            selected[task_id, grouping] = by_grouping[grouping]
    return selected, metrics


def draw_intersection(report, selected, grouping, output_path):
    """Zeichnet eine intersektionale Heatmap mit positiven Labelraten und Gruppengrößen."""

    codes = list(range(1, 10))
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 5.7))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e9edf1")
    figure.subplots_adjust(left=0.245, right=0.95, top=0.69, bottom=0.18, wspace=0.10)
    plotted = []
    for index, (axis, task_id) in enumerate(zip(axes, TASK_PRESENTATION, strict=True)):
        rows = selected[task_id, grouping]
        grid = np.full((len(codes), 2), np.nan)
        for y, code in enumerate(codes):
            for x, sex in enumerate((1, 2)):
                row = rows.get((sex, code))
                if row is None:
                    plotted.append({"task_id": task_id, "sex_code": sex, "group_code": code, "status": "excluded_by_task_filter"})
                    continue
                grid[y, x] = row["positive_label_rate"]
                plotted.append({"task_id": task_id, "sex_code": sex, "group_code": code, "group_size": row["group_size"], "positive_labels": row["positive_labels"], "negative_labels": row["negative_labels"], "positive_label_rate": row["positive_label_rate"]})
        mesh = axis.pcolormesh(np.arange(3) - 0.5, np.arange(len(codes) + 1) - 0.5, np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=0.7)
        axis.set(xlim=(-0.5, 1.5), ylim=(len(codes) - 0.5, -0.5))
        axis.set_xticks((0, 1), ("Männlich", "Weiblich"), fontsize=8.2)
        axis.set_yticks(range(len(codes)), RACE_LABELS if index == 0 else [])
        axis.tick_params(axis="y", labelsize=8.3, length=0)
        axis.tick_params(axis="x", length=0, pad=5)
        center = (axis.get_position().x0 + axis.get_position().x1) / 2
        figure.text(center, 0.93, TASK_PRESENTATION[task_id]["name"], ha="center", va="top", fontsize=9.5, fontweight="bold")
        figure.text(center, 0.88, TASK_SUBTITLES[task_id] + "\nN = " + format_count(report["tasks"][task_id]["rows"]), ha="center", va="top", fontsize=8, linespacing=1.3)
        for y, code in enumerate(codes):
            for x, sex in enumerate((1, 2)):
                row = rows.get((sex, code))
                if row is None:
                    axis.text(x, y, "Filter", ha="center", va="center", fontsize=8, color="#666666")
                    continue
                rate = row["positive_label_rate"]
                marker = "*" if row["group_size"] < 30 else ""
                axis.text(x, y, f"{format_decimal(rate * 100, 1)} %\nn={format_count(row['group_size'])}{marker}", ha="center", va="center", fontsize=7.8, color=annotation_color(cmap, rate), linespacing=1.35)
    cbar = figure.colorbar(mesh, ax=list(axes), location="right", fraction=0.038, pad=0.018)
    cbar.solids.set_rasterized(False)
    cbar.set_ticks((0, 0.25, 0.5, 0.75, 1))
    cbar.set_ticklabels(("0", "25", "50", "75", "100 %"))
    cbar.ax.tick_params(labelsize=8, length=2)
    note = "Zellen: positive Labelrate und Gruppengröße. * n < 30 (Hinweis, kein Ausschluss)."
    figure.text(0.02, 0.075, note, fontsize=8, ha="left", va="bottom")
    figure.text(0.02, 0.02, "Kalifornien | ACS PUMS 2018, 1-Year | ungewichtet | eigene Darstellung", fontsize=7.5, color="#555555", ha="left")
    figure.savefig(output_path, format="pdf", metadata={"Creator": "visualize_reports.py", "CreationDate": None, "ModDate": None})
    plt.close(figure)
    return plotted


def plot_metric_figures(report, report_path, output_dir):
    """Erzeugt die Intersektionsabbildung und dokumentiert die geprüften Metriken und Quellen."""

    selected, metrics = check_results(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_report": str(report_path.resolve().relative_to(PROJECT_ROOT)) if report_path.resolve().is_relative_to(PROJECT_ROOT) else str(report_path.resolve()),
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "prepared_csv_sha256": report["provenance"]["prepared_data_file"]["sha256"],
        "scope": report["provenance"]["data_source"],
        "analysis_weighting": "unweighted",
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "validated_multigroup_metrics": metrics, "figures": [],
    }
    style = {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}
    with plt.rc_context(style):
        for filename, grouping in (("ca_sex_race_label_rates.pdf", "sex_race"),):
            output_path = output_dir / filename
            plotted = draw_intersection(report, selected, grouping, output_path)
            manifest["figures"].append({"file": filename, "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "grouping": grouping, "color_scale": [0, 1], "plotted_groups": plotted})
    (output_dir / "ca_metric_figures.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


BENCHMARK_SCOPE_LABELS = {"compute_only": "Compute-only", "end_to_end": "End-to-End"}
BENCHMARK_IMPLEMENTATIONS = {"sql": ("SQL", "#31688e"), "pandas": ("Pandas", "#d17c2f")}
BENCHMARK_GROUPINGS = list(REPORTED_GROUPINGS)
BENCHMARK_GROUP_LABELS = {
    "sex": "Geschlecht",
    "race": "„Race“",
    "sex_race": "Geschlecht\n× „Race“",
}


def file_sha256(path):
    """Berechnet die SHA-256-Prüfsumme einer Datei."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_benchmark_summary(summary_path, environment_path):
    """Lädt die Hauptmessung und prüft Prüfsumme, Versuchsdesign und gültige Zeitwerte."""

    if not summary_path.is_file() or not environment_path.is_file():
        raise FileNotFoundError("Benchmark-Zusammenfassung fehlt. Zuerst fairness_auswertung.py benchmark --main ausführen.")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("status") != "completed" or environment.get("mode") != "main":
        raise ValueError("Benchmark-Abbildungen erfordern eine abgeschlossene Hauptmessung.")
    if environment.get("results", {}).get("summary_sha256") != file_sha256(summary_path):
        raise ValueError("Prüfsumme der Benchmark-Zusammenfassung stimmt nicht mit dem Umgebungsprotokoll überein.")
    summary = pd.read_csv(summary_path)
    required_columns = {
        "configuration_id", "task_id", "implementation", "measurement_scope", "dataset_size",
        "grouping", "observed_group_count", "metric_bundle", "measured_runs", "first_quartile_ms",
        "median_ms", "third_quartile_ms", "interquartile_range_ms",
    }
    if required_columns.difference(summary.columns):
        raise ValueError("Benchmark-Zusammenfassung enthält nicht alle benötigten Spalten.")
    if len(summary) != 136 or summary["configuration_id"].nunique() != 136 or not (summary["measured_runs"] == 20).all():
        raise ValueError("Benchmark-Zusammenfassung entspricht nicht dem festgelegten Hauptmessungsdesign.")
    if set(summary["task_id"]) != set(TASK_PRESENTATION):
        raise ValueError("Benchmark-Zusammenfassung enthält nicht die drei bestätigten Aufgaben.")
    if not np.isfinite(summary[["first_quartile_ms", "median_ms", "third_quartile_ms"]].to_numpy()).all():
        raise ValueError("Benchmark-Zusammenfassung enthält ungültige Zeitwerte.")
    return summary, environment


def format_size_tick(value, _position=None):
    """Formatiert die Zeilenzahl einer Achsenmarkierung in Tausend oder Millionen."""

    if value >= 1_000_000:
        return format_decimal(value / 1_000_000, 0 if value % 1_000_000 == 0 else 1) + " Mio."
    return format_count(value // 1_000) + " Tsd."


def style_benchmark_axis(axis):
    """Vereinheitlicht Gitternetz und sichtbare Rahmenlinien einer Benchmarkachse."""

    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def benchmark_legend(figure, handles):
    """Ergänzt die gemeinsame Legende für SQL und Pandas oberhalb der Teilabbildungen."""

    figure.legend(handles, [BENCHMARK_IMPLEMENTATIONS[key][0] for key in BENCHMARK_IMPLEMENTATIONS], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=9)


def plot_benchmark_size(summary, output_path):
    """Zeichnet Laufzeitmediane und Interquartilsbereiche bei wachsender Zeilenzahl."""

    selected = summary[(summary["grouping"] == "sex_race") & (summary["metric_bundle"] == "Stufe 3")]
    figure, axes = plt.subplots(2, 3, figsize=(6.6, 6.0), sharex=True, sharey="row")
    handles = []
    for column, task_id in enumerate(TASK_PRESENTATION):
        for row, scope in enumerate(BENCHMARK_SCOPE_LABELS):
            axis = axes[row, column]
            panel = selected[(selected["task_id"] == task_id) & (selected["measurement_scope"] == scope)]
            for implementation, (_label, color) in BENCHMARK_IMPLEMENTATIONS.items():
                values = panel[panel["implementation"] == implementation].sort_values("dataset_size")
                handle, = axis.plot(values["dataset_size"], values["median_ms"], marker="o", markersize=3.2, linewidth=1.5, color=color)
                axis.fill_between(values["dataset_size"], values["first_quartile_ms"], values["third_quartile_ms"], color=color, alpha=0.15, linewidth=0)
                if row == 0 and column == 0:
                    handles.append(handle)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(9_000, 3_600_000)
            axis.set_xticks([10_000, 100_000, 1_000_000])
            axis.xaxis.set_major_formatter(mticker.FuncFormatter(format_size_tick))
            axis.xaxis.set_minor_formatter(mticker.NullFormatter())
            axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
            axis.yaxis.set_minor_formatter(mticker.NullFormatter())
            style_benchmark_axis(axis)
            if row == 0:
                maximum = int(panel["dataset_size"].max())
                axis.set_title(f"{TASK_PRESENTATION[task_id]['name']}\nNmax = {format_count(maximum)}", fontsize=9.5, pad=8)
            if column == 0:
                axis.set_ylabel(BENCHMARK_SCOPE_LABELS[scope] + "\nMedian (ms)")
            if row == 1:
                axis.set_xlabel("Zeilenzahl")
    benchmark_legend(figure, handles)
    figure.text(0.02, 0.012, "Punkte: Median; Bänder: Interquartilsabstand; Gruppierung Geschlecht × „Race“; Stufe 3.\nBeide Achsen logarithmisch.", fontsize=7.8, color="#555555", linespacing=1.3)
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.87, hspace=0.32, wspace=0.16)
    figure.savefig(output_path, format="pdf", metadata={"Creator": "visualize_reports.py", "CreationDate": None, "ModDate": None})
    plt.close(figure)


def plot_benchmark_categories(summary, output_path, dimension):
    """Vergleicht Laufzeiten für Gruppierungen oder Metrikumfänge bei einer Million Zeilen."""

    if dimension == "grouping":
        selected = summary[(summary["dataset_size"] == 1_000_000) & (summary["metric_bundle"] == "Stufe 3")]
        categories = BENCHMARK_GROUPINGS
        labels = BENCHMARK_GROUP_LABELS
        footer = "Punkte: Median; Bänder: Interquartilsabstand; Stufe 3.\nN = Zeilenzahl je Aufgabe und Gruppierung; k = Gruppenzahl."
    else:
        selected = summary[(summary["dataset_size"] == 1_000_000) & (summary["grouping"] == "sex_race")]
        categories = ["Stufe 1", "Stufe 2", "Stufe 3"]
        labels = {key: key for key in categories}
        footer = "Punkte: Median; Bänder: Interquartilsabstand; N = 1.000.000; Gruppierung Geschlecht × „Race“.\nKumulative Berechnungsstufen 1 bis 3."
    figure, axes = plt.subplots(2, 3, figsize=(6.6, 6.0 if dimension == "grouping" else 5.6), sharex=dimension == "grouping", sharey="row")
    handles = []
    positions = np.arange(len(categories))
    for column, task_id in enumerate(TASK_PRESENTATION):
        for row, scope in enumerate(BENCHMARK_SCOPE_LABELS):
            axis = axes[row, column]
            panel = selected[(selected["task_id"] == task_id) & (selected["measurement_scope"] == scope)]
            for implementation, (_label, color) in BENCHMARK_IMPLEMENTATIONS.items():
                values = panel[panel["implementation"] == implementation].set_index(dimension).loc[categories]
                handle, = axis.plot(positions, values["median_ms"], marker="o", markersize=3.5, linewidth=1.5, color=color)
                axis.fill_between(positions, values["first_quartile_ms"], values["third_quartile_ms"], color=color, alpha=0.15, linewidth=0)
                if row == 0 and column == 0:
                    handles.append(handle)
            if dimension == "grouping":
                counts = panel.drop_duplicates("grouping").set_index("grouping")["observed_group_count"]
                tick_labels = [f"{labels[key]}\n(k = {int(counts[key])})" for key in categories]
                axis.set_xlim(-0.4, len(categories) - 0.6)
            else:
                tick_labels = [labels[key] for key in categories]
            axis.set_xticks(positions, tick_labels, fontsize=8 if dimension == "grouping" else 8.5)
            axis.set_ylim(bottom=0)
            style_benchmark_axis(axis)
            if row == 0:
                if dimension == "grouping":
                    axis.set_title(TASK_PRESENTATION[task_id]["name"], fontsize=9.5, pad=24)
                    axis.text(0.5, 1.02, "N = 1.000.000", transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
                else:
                    axis.set_title(TASK_PRESENTATION[task_id]["name"], fontsize=9.5, pad=8)
            if column == 0:
                axis.set_ylabel(BENCHMARK_SCOPE_LABELS[scope] + "\nMedian (ms)")
            if row == 1:
                axis.set_xlabel("Gruppierung" if dimension == "grouping" else "Metrikumfang", labelpad=10 if dimension == "grouping" else None)
    benchmark_legend(figure, handles)
    figure.text(0.02, 0.012, footer, fontsize=7.8, color="#555555", linespacing=1.3)
    if dimension == "grouping":
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.83, hspace=0.30, wspace=0.28)
    else:
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.87, hspace=0.31, wspace=0.20)
    figure.savefig(output_path, format="pdf", metadata={"Creator": "visualize_reports.py", "CreationDate": None, "ModDate": None})
    plt.close(figure)


def plot_benchmark_figures(summary, environment, summary_path, environment_path, output_dir):
    """Erzeugt die drei Benchmarkabbildungen samt Prüfsummen und Herkunftsnachweis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("benchmark_zeilenzahl.pdf", "Z"),
        ("benchmark_gruppenkomplexitaet.pdf", "G"),
        ("benchmark_metrikumfang.pdf", "M"),
    ]
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "pdf.fonttype": 42}
    with plt.rc_context(style):
        plot_benchmark_size(summary, output_dir / definitions[0][0])
        plot_benchmark_categories(summary, output_dir / definitions[1][0], "grouping")
        plot_benchmark_categories(summary, output_dir / definitions[2][0], "metric_bundle")
    manifest = {
        "run_id": environment["run_id"],
        "source_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "source_summary_sha256": file_sha256(summary_path),
        "source_environment": str(environment_path.relative_to(PROJECT_ROOT)),
        "source_environment_sha256": file_sha256(environment_path),
        "measured_runs_per_configuration": 20,
        "figures": [{"file": filename, "experiment": experiment, "sha256": file_sha256(output_dir / filename)} for filename, experiment in definitions],
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output_dir / "benchmark_figures.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    """Lädt die geprüften Berichte und erzeugt alle sechs Analyse- und Benchmarkabbildungen."""

    parse_args()
    report = load_report(DEFAULT_REPORT_PATH)
    benchmark_summary, benchmark_environment = load_benchmark_summary(DEFAULT_BENCHMARK_SUMMARY_PATH, DEFAULT_BENCHMARK_ENVIRONMENT_PATH)
    plot_descriptive_figures(report, DEFAULT_REPORT_PATH, DEFAULT_OUTPUT_DIR)
    plot_metric_figures(report, DEFAULT_REPORT_PATH, DEFAULT_OUTPUT_DIR)
    plot_benchmark_figures(benchmark_summary, benchmark_environment, DEFAULT_BENCHMARK_SUMMARY_PATH, DEFAULT_BENCHMARK_ENVIRONMENT_PATH, DEFAULT_OUTPUT_DIR)
    print(f"Gespeichert: sechs PDFs und Herkunftsdateien unter {DEFAULT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
