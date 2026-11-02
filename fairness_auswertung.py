from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import tempfile
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns
import pandas as pd
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo
try:
    from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, state_list
    from folktables.acs import adult_filter, public_coverage_filter
except ImportError as exc:
    raise SystemExit("Folktables ist nicht installiert. Bitte zuerst die Projektabhängigkeiten installieren.") from exc

SURVEY_YEAR = "2018"
HORIZON = "1-Year"
SURVEY = "person"
DEFAULT_STATE = "CA"
DEFAULT_SEED = 42
FLOAT_TOLERANCE = 1e-12
BENCHMARK_STATES = tuple(state_list)
BENCHMARK_FIXED_SIZES = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000)
BENCHMARK_GROUPINGS = ("sex", "race", "sex_race")
BENCHMARK_METRIC_BUNDLES = ("Stufe 1", "Stufe 2", "Stufe 3")
BENCHMARK_METRIC_BUNDLE_SLUGS = {
    bundle: f"stufe_{index}" for index, bundle in enumerate(BENCHMARK_METRIC_BUNDLES, start=1)
}
BENCHMARK_WARMUPS = 3
BENCHMARK_MEASURED_BLOCKS = 20
BENCHMARK_PILOT_WARMUPS = 1
BENCHMARK_PILOT_BLOCKS = 2

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "folktables"
DEFAULT_PREPARED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "acs_tasks_ca_2018_1year.csv"
DEFAULT_REPORT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "acs_tasks_ca_2018_1year.metadata.json"
DEFAULT_SCALING_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "acs_tasks_all_regions_2018_1year.csv"
DEFAULT_SCALING_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "acs_tasks_all_regions_2018_1year.metadata.json"
DEFAULT_SCALING_VALIDATION_PATH = PROJECT_ROOT / "results" / "raw" / "full_regions_validation.json"
DEFAULT_BENCHMARK_RAW_PATH = PROJECT_ROOT / "results" / "raw" / "benchmark_runs.csv"
DEFAULT_BENCHMARK_ENVIRONMENT_PATH = PROJECT_ROOT / "results" / "raw" / "benchmark_environment.json"
DEFAULT_BENCHMARK_PLAN_ROOT = PROJECT_ROOT / "results" / "raw" / "query_plans"
DEFAULT_BENCHMARK_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "benchmark_summary.csv"
DEFAULT_RESULT_PATH = PROJECT_ROOT / "results" / "python_ergebnisse.json"
DEFAULT_ANALYSIS_REPORT_PATH = PROJECT_ROOT / "results" / "analysis_report.json"
REPORT_SCHEMA_VERSION = "1.1"
def identity_filter(frame):
    """Behält die ACSEmployment Population unverändert."""

    return frame


TASK_DEFINITIONS = {
    "acs_income": {
        "dataset_name": "Folktables ACSIncome",
        "problem": ACSIncome,
        "preprocess": adult_filter,
        "population_definition": "AGEP > 16, PINCP > 100, WKHP > 0 and PWGTP >= 1",
        "positive_label_definition": "PINCP > 50000 after ACSIncome filters",
    },
    "acs_employment": {
        "dataset_name": "Folktables ACSEmployment",
        "problem": ACSEmployment,
        "preprocess": identity_filter,
        "population_definition": "All selected ACS person records without an additional Folktables task filter",
        "positive_label_definition": "ESR == 1 in the ACSEmployment population",
    },
    "acs_public_coverage": {
        "dataset_name": "Folktables ACSPublicCoverage",
        "problem": ACSPublicCoverage,
        "preprocess": public_coverage_filter,
        "population_definition": "AGEP < 65 and PINCP <= 30000",
        "positive_label_definition": "PUBCOV == 1 after AGEP < 65 and PINCP <= 30000",
    },
}
TASK_IDS = tuple(TASK_DEFINITIONS)

GROUPINGS: dict[str, list[str]] = {
    "sex": ["sex"],
    "race": ["race"],
    "sex_race": ["sex", "race"],
}

PAIRWISE_COLUMNS = (
    "grouping_name reference_code_1 reference_code_2 comparison_code_1 comparison_code_2 "
    "reference_group_size comparison_group_size reference_positive_label_rate "
    "comparison_positive_label_rate statistical_parity_difference disparate_impact_ratio"
).split()
MULTIGROUP_COLUMNS = (
    "grouping_name included_group_count minimum_positive_label_rate maximum_positive_label_rate "
    "minimum_groups maximum_groups demographic_parity_difference demographic_parity_ratio"
).split()
BENCHMARK_GROUP_COLUMNS = (
    "grouping_name group_code_1 group_code_2 group_size positive_labels negative_labels "
    "representation_rate positive_label_rate"
).split()
BENCHMARK_RAW_FIELDS = (
    "run_id experiment configuration_id task_id implementation measurement_scope dataset_size grouping "
    "observed_group_count metric_bundle warmup_or_measured round execution_order duration_ms "
    "client_wall_duration_ms timestamp_utc seed result_group_rows result_pair_rows "
    "result_multigroup_rows sql_planning_time_ms"
).split()

class PreparationError(RuntimeError):
    """Fehler bei nicht eindeutig aufbereitbaren Quelldaten."""

class MetricInputError(ValueError):
    """Fehler bei ungültigen Metrikeingaben."""

class ReportError(RuntimeError):
    """Fehler bei nicht passenden Berichtsmetadaten."""

class BenchmarkError(RuntimeError):
    """Fehler bei einem nicht reproduzierbaren Benchmarklauf."""

@dataclass(frozen=True)
class MultigroupMetrics:
    """Mehrgruppenmetriken aus kleinster und größter Gruppenrate."""

    grouping_name: str
    included_group_count: int
    minimum_positive_label_rate: float | None
    maximum_positive_label_rate: float | None
    minimum_groups: tuple[tuple[int, ...], ...]
    maximum_groups: tuple[tuple[int, ...], ...]
    demographic_parity_difference: float | None
    demographic_parity_ratio: float | None

    def to_dict(self):
        """Wandelt Mehrgruppenwerte und Gruppencodes in eine JSON-kompatible Struktur um."""

        result = asdict(self)
        result["minimum_groups"] = [list(group) for group in self.minimum_groups]
        result["maximum_groups"] = [list(group) for group in self.maximum_groups]
        return result

# Datenaufbereitung

def package_version(package_name):
    """Liefert die installierte Paketversion oder unknown bei einem unbekannten Paket."""

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"

def file_hash(path):
    """Berechnet die SHA-256-Prüfsumme einer Datei blockweise."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()

def write_atomic(output_path: Path, data: pd.DataFrame | dict | list | str):
    """Schreibt CSV, JSON oder Text atomar in das Zielverzeichnis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f"{output_path.stem}.",
            suffix=".tmp", dir=output_path.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            if isinstance(data, pd.DataFrame):
                data.to_csv(handle, index=False)
            elif data and isinstance(data, list) and all(isinstance(path, Path) for path in data):
                for index, path in enumerate(data):
                    with path.open(encoding="utf-8", newline="") as source:
                        if index:
                            next(source, None)
                        shutil.copyfileobj(source, handle)
            elif isinstance(data, (dict, list)):
                json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            elif isinstance(data, str):
                handle.write(data)
            else:
                raise TypeError(f"Unsupported atomic output type: {type(data).__name__}")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def numeric_column(series, *, column_name):
    """Prüft ACS Werte ohne fehlerhafte Werte in fehlende Werte umzuwandeln."""

    try:
        return pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{column_name} contains a non-numeric value.") from exc

def normalise_age(series, *, error_type=MetricInputError):
    """Prüft ganzzahlige Alterswerte von 0 bis 99 und erhält fehlende Werte."""

    numeric = pd.to_numeric(series, errors="coerce")
    present = numeric.notna()
    infinite = numeric.isin([math.inf, -math.inf])
    finite = present & ~infinite
    errors = {
        "non-numeric": int((series.notna() & numeric.isna()).sum()),
        "non-finite": int(infinite.sum()),
        "non-integer": int((finite & numeric.mod(1).ne(0)).sum()),
        "outside 0..99": int((finite & ~numeric.between(0, 99)).sum()),
    }
    if any(errors.values()):
        details = ", ".join(f"{kind}={count}" for kind, count in errors.items() if count)
        raise error_type(f"AGEP/age must be a finite integer in 0..99: {details}.")
    return numeric.astype("Int16")

def rank_task_frame(frame, *, seed):
    """Mischt genau eine Aufgabe deterministisch und vergibt sample_rank."""

    if "sample_rank" in frame.columns:
        raise PreparationError("sample_rank must not exist before the deterministic shuffle.")
    if "task_id" not in frame.columns or frame.empty or frame["task_id"].nunique(dropna=False) != 1:
        raise PreparationError("Ranking requires a non-empty frame containing exactly one task_id.")

    ranked = frame.sample(frac=1, random_state=seed, ignore_index=True)
    ranked.insert(1, "sample_rank", pd.Series(range(1, len(ranked) + 1), dtype="int64"))
    return ranked


def prepare_tasks(raw_data, *, task_ids=TASK_IDS, seed, assign_sample_rank=True):
    """Bereitet ausgewählte Aufgaben aus einer ACS Quelle gemeinsam auf."""

    selected = list(task_ids)
    if not selected:
        raise PreparationError("At least one task must be selected.")
    if len(set(selected)) != len(selected):
        raise PreparationError("Each task may be selected only once.")

    frames = []
    for task_id in selected:
        if task_id not in TASK_DEFINITIONS:
            raise PreparationError(f"Unknown task: {task_id!r}.")
        definition = TASK_DEFINITIONS[task_id]
        required_columns = {
            *definition["problem"].features,
            definition["problem"].target,
            "PWGTP",
            "SEX",
            "RAC1P",
        }
        missing_columns = sorted(required_columns.difference(raw_data.columns))
        if missing_columns:
            raise PreparationError(
                f"The downloaded ACS person data for {task_id} are missing "
                "required columns: " + ", ".join(missing_columns)
            )

        validated_raw = raw_data.copy()
        validated_raw["AGEP"] = normalise_age(raw_data["AGEP"], error_type=PreparationError)
        filtered = definition["preprocess"](validated_raw).copy()
        _, folktables_labels, _ = definition["problem"].df_to_pandas(validated_raw)

        if len(filtered) != len(folktables_labels):
            raise PreparationError(f"The independently retained {task_id} rows do not align with the Folktables task output.")

        expected_labels = folktables_labels.iloc[:, 0].reset_index(drop=True).astype("boolean")
        derived_labels = definition["problem"].target_transform(filtered[definition["problem"].target]).reset_index(drop=True).astype("boolean")
        if not expected_labels.equals(derived_labels):
            raise PreparationError(f"The exported labels differ from Folktables {task_id} labels.")

        prepared = pd.DataFrame(
            {
                "sex": numeric_column(filtered["SEX"], column_name="SEX").reset_index(drop=True).astype("Int16"),
                "race": numeric_column(filtered["RAC1P"], column_name="RAC1P").reset_index(drop=True).astype("Int16"),
                "label": expected_labels,
                "person_weight": numeric_column(filtered["PWGTP"], column_name="PWGTP").reset_index(drop=True).astype("Float64"),
                "age": filtered["AGEP"].reset_index(drop=True).astype("Int16"),
            }
        )

        if prepared["label"].isna().any():
            raise PreparationError(f"The {task_id} label contains missing values.")
        if not prepared["sex"].dropna().between(1, 2).all():
            raise PreparationError("SEX contains a code outside 1..2.")
        if not prepared["race"].dropna().between(1, 9).all():
            raise PreparationError("RAC1P contains a code outside 1..9.")
        if prepared["person_weight"].notna().any() and (prepared["person_weight"].dropna() < 1).any():
            raise PreparationError("PWGTP contains a value below 1.")

        prepared.insert(0, "task_id", task_id)
        if assign_sample_rank:
            prepared = rank_task_frame(prepared, seed=seed)
        frames.append(prepared)

    combined = pd.concat(frames, ignore_index=True)
    if assign_sample_rank and combined.duplicated(["task_id", "sample_rank"]).any():
        raise PreparationError("task_id and sample_rank must be unique together.")
    return combined


def build_task_metadata(prepared):
    """Fasst einen bereits gerankten Aufgabenbestand zusammen."""

    if "sample_rank" not in prepared.columns:
        raise PreparationError("Task metadata require sample_rank.")
    selected_task_ids = list(dict.fromkeys(prepared["task_id"].tolist()))
    unknown_tasks = sorted(set(selected_task_ids).difference(TASK_DEFINITIONS))
    if unknown_tasks:
        raise PreparationError(f"Metadata received unknown tasks: {unknown_tasks}")

    task_metadata = {}
    for task_id in selected_task_ids:
        current = prepared.loc[prepared["task_id"].eq(task_id)]
        definition = TASK_DEFINITIONS[task_id]
        task_metadata[task_id] = {
            "dataset": definition["dataset_name"],
            "population_definition": definition["population_definition"],
            "positive_label_definition": definition["positive_label_definition"],
            "prepared_rows": len(current),
            "sample_rank_range": [int(current["sample_rank"].min()), int(current["sample_rank"].max())],
            "negative_labels": int((~current["label"].astype(bool)).sum()),
            "positive_labels": int(current["label"].astype(bool).sum()),
            "missing_sex": int(current["sex"].isna().sum()),
            "missing_race": int(current["race"].isna().sum()),
            "missing_person_weight": int(current["person_weight"].isna().sum()),
            "missing_age": int(current["age"].isna().sum()),
        }

    return task_metadata


def build_metadata(*, state, seed, raw_rows, raw_missing_age, prepared, output_path):
    """Erstellt Metadaten für die erzeugte CSV."""

    selected_task_ids = list(dict.fromkeys(prepared["task_id"].tolist()))
    task_metadata = build_task_metadata(prepared)
    return {
        "dataset": "Folktables ACS analysis tasks",
        "source": "American Community Survey PUMS",
        "source_url": "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/",
        "survey_year": SURVEY_YEAR,
        "horizon": HORIZON,
        "survey": SURVEY,
        "states": [state],
        "seed": seed,
        "analysis_weighting": "unweighted",
        "person_weight_retained": True,
        "task_ids": selected_task_ids,
        "tasks": task_metadata,
        "columns": list(prepared.columns),
        "raw_rows": raw_rows,
        "raw_missing_age": raw_missing_age,
        "age_source_variable": "AGEP",
        "prepared_rows": len(prepared),
        "folktables_version": package_version("folktables"),
        "pandas_version": package_version("pandas"),
        "python_version": platform.python_version(),
        "csv_path": str(output_path),
    }

def prepare_data():
    """Lädt die kalifornischen ACS-Daten und speichert die drei Aufgaben samt Metadaten."""

    DEFAULT_RAW_ROOT.mkdir(parents=True, exist_ok=True)

    source = ACSDataSource(survey_year=SURVEY_YEAR, horizon=HORIZON, survey=SURVEY, root_dir=str(DEFAULT_RAW_ROOT))
    raw_data = source.get_data(states=[DEFAULT_STATE], download=True)
    prepared = prepare_tasks(raw_data, task_ids=TASK_IDS, seed=DEFAULT_SEED)
    write_atomic(DEFAULT_PREPARED_CSV_PATH, prepared)

    metadata = build_metadata(
        state=DEFAULT_STATE,
        seed=DEFAULT_SEED,
        raw_rows=len(raw_data),
        raw_missing_age=int(raw_data["AGEP"].isna().sum()),
        prepared=prepared,
        output_path=DEFAULT_PREPARED_CSV_PATH,
    )
    metadata["csv_sha256"] = file_hash(DEFAULT_PREPARED_CSV_PATH)
    write_atomic(DEFAULT_REPORT_METADATA_PATH, metadata)

    print(f"Metadaten gespeichert: {DEFAULT_REPORT_METADATA_PATH}")
    return 0


def benchmark_size_plan(task_sizes, *, fixed_sizes=BENCHMARK_FIXED_SIZES):
    """Leitet gültige verschachtelte Stufen und die gemeinsame Stufe ab."""

    if set(task_sizes) != set(TASK_IDS):
        raise PreparationError("Benchmark size planning requires exactly the three confirmed tasks.")
    if any(not isinstance(size, int) or size <= 0 for size in task_sizes.values()):
        raise PreparationError("Benchmark task sizes must be positive integers.")
    if tuple(sorted(set(fixed_sizes))) != tuple(fixed_sizes) or any(size <= 0 for size in fixed_sizes):
        raise PreparationError("Fixed benchmark sizes must be unique positive integers in ascending order.")

    stages = {}
    for task_id in TASK_IDS:
        full_size = task_sizes[task_id]
        current = [size for size in fixed_sizes if size <= full_size]
        if full_size not in current:
            current.append(full_size)
        stages[task_id] = current

    common_candidates = [size for size in fixed_sizes if all(task_sizes[task_id] >= size for task_id in TASK_IDS)]
    if not common_candidates:
        raise PreparationError("No fixed benchmark size is available in all three tasks.")

    return {
        "fixed_candidate_sizes": list(fixed_sizes),
        "full_sizes": dict(task_sizes),
        "stages": stages,
        "n_common": max(common_candidates),
        "subset_rule": "task_id fixed and sample_rank <= dataset_size",
    }


def raw_file_for_state(raw_data):
    """Bestimmt die von Folktables geladene ACS-Personendatei."""

    if "ST" not in raw_data.columns:
        raise PreparationError("The downloaded ACS person data do not contain ST.")
    state_codes = numeric_column(raw_data["ST"], column_name="ST").dropna().astype("int64").unique()
    if len(state_codes) != 1:
        raise PreparationError(f"Expected exactly one state code, got {state_codes.tolist()}.")

    raw_file = DEFAULT_RAW_ROOT / SURVEY_YEAR / HORIZON / f"psam_p{int(state_codes[0]):02d}.csv"
    if not raw_file.is_file():
        raise PreparationError(f"Downloaded ACS source file is missing: {raw_file}")
    return raw_file


def regional_task_counts(prepared):
    """Erzeugt kompakte Kontrollsummen vor der globalen Mischung."""

    result = {}
    for task_id in TASK_IDS:
        current = prepared.loc[prepared["task_id"].eq(task_id)]
        result[task_id] = {
            "rows": len(current),
            "positive_labels": int(current["label"].astype(bool).sum()),
            "negative_labels": int((~current["label"].astype(bool)).sum()),
            "missing_sex": int(current["sex"].isna().sum()),
            "missing_race": int(current["race"].isna().sum()),
            "missing_age": int(current["age"].isna().sum()),
        }
    return result


def read_staged_task(path):
    """Liest eine regionsweise erzeugte Aufgabe mit stabilen Datentypen."""

    return pd.read_csv(
        path,
        dtype={
            "task_id": "string",
            "sex": "Int16",
            "race": "Int16",
            "label": "boolean",
            "person_weight": "Float64",
            "age": "Int16",
        },
    )


def prepare_scaling_data():
    """Bereitet den vollständigen Regionsumfang für Skalierbarkeit vor."""

    DEFAULT_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_SCALING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    source = ACSDataSource(survey_year=SURVEY_YEAR, horizon=HORIZON, survey=SURVEY, root_dir=str(DEFAULT_RAW_ROOT))
    region_metadata = []

    with tempfile.TemporaryDirectory(prefix="scaling-preparation-", dir=DEFAULT_SCALING_CSV_PATH.parent) as staging_directory:
        staging_root = Path(staging_directory)
        staging_paths = {task_id: staging_root / f"{task_id}.csv" for task_id in TASK_IDS}

        for index, state in enumerate(BENCHMARK_STATES, start=1):
            print(f"[{index}/{len(BENCHMARK_STATES)}] Region {state} laden", flush=True)
            raw_data = source.get_data(states=[state], download=True)
            raw_file = raw_file_for_state(raw_data)
            prepared = prepare_tasks(raw_data, task_ids=TASK_IDS, seed=DEFAULT_SEED, assign_sample_rank=False)
            task_counts = regional_task_counts(prepared)

            for task_id, path in staging_paths.items():
                current = prepared.loc[prepared["task_id"].eq(task_id)]
                current.to_csv(path, mode="a", header=not path.exists(), index=False)

            region_metadata.append({
                "state": state,
                "raw_rows": len(raw_data),
                "raw_missing_age": int(raw_data["AGEP"].isna().sum()),
                "raw_file": str(raw_file.relative_to(PROJECT_ROOT)),
                "raw_file_bytes": raw_file.stat().st_size,
                "raw_file_sha256": file_hash(raw_file),
                "tasks": task_counts,
            })
            print(
                f"[{index}/{len(BENCHMARK_STATES)}] Region {state} geprüft: "
                + ", ".join(f"{task_id}={task_counts[task_id]['rows']}" for task_id in TASK_IDS),
                flush=True,
            )

        task_metadata = {}
        task_sizes = {}
        for task_id in TASK_IDS:
            ranked = rank_task_frame(read_staged_task(staging_paths[task_id]), seed=DEFAULT_SEED)
            task_sizes[task_id] = len(ranked)
            task_metadata[task_id] = build_task_metadata(ranked)[task_id]
            ranked.to_csv(staging_paths[task_id], index=False)
            print(f"Globale Mischung abgeschlossen: {task_id}={len(ranked)}", flush=True)

        write_atomic(DEFAULT_SCALING_CSV_PATH, list(staging_paths.values()))

    size_plan = benchmark_size_plan(task_sizes)
    metadata = {
        "metadata_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Folktables ACS analysis tasks",
        "purpose": "scalability experiments only",
        "content_analysis_scope": ["CA"],
        "source": "American Community Survey PUMS",
        "source_url": "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/",
        "survey_year": SURVEY_YEAR,
        "horizon": HORIZON,
        "survey": SURVEY,
        "states": list(BENCHMARK_STATES),
        "state_order_source": "folktables.state_list",
        "region_count": len(BENCHMARK_STATES),
        "regions": region_metadata,
        "seed": DEFAULT_SEED,
        "analysis_weighting": "unweighted",
        "person_weight_retained": True,
        "task_ids": list(TASK_IDS),
        "tasks": task_metadata,
        "columns": ["task_id", "sample_rank", "sex", "race", "label", "person_weight", "age"],
        "raw_rows": sum(region["raw_rows"] for region in region_metadata),
        "raw_missing_age": sum(region["raw_missing_age"] for region in region_metadata),
        "prepared_rows": sum(task_sizes.values()),
        "age_source_variable": "AGEP",
        "reporting_groupings": list(BENCHMARK_GROUPINGS),
        "benchmark_sizes": size_plan,
        "folktables_version": package_version("folktables"),
        "pandas_version": package_version("pandas"),
        "python_version": platform.python_version(),
        "csv_path": str(DEFAULT_SCALING_CSV_PATH),
        "csv_sha256": file_hash(DEFAULT_SCALING_CSV_PATH),
    }
    write_atomic(DEFAULT_SCALING_METADATA_PATH, metadata)
    print(f"Skalierungsdaten gespeichert: {DEFAULT_SCALING_CSV_PATH}")
    print(f"Metadaten gespeichert: {DEFAULT_SCALING_METADATA_PATH}")
    return 0

# Pandas Referenz

def normalise_labels(series):
    """Prüft binäre Labels ohne fehlende Werte und wandelt sie in boolesche Werte um."""

    if series.isna().any():
        raise MetricInputError("label must not contain missing values.")

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
        raise MetricInputError("label must contain only boolean or 0/1 values.")
    return numeric.astype(bool)

def group_statistics(frame, group_columns, *, label_column="label"):
    """Berechnet ungewichtete Gruppen und Labelstatistiken."""

    if not group_columns:
        raise MetricInputError("At least one group column is required.")

    required = {*group_columns, label_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MetricInputError(f"Missing required columns: {missing}")
    if frame.empty:
        raise MetricInputError("Cannot calculate metrics for an empty frame.")

    work = frame[list(required)].copy()
    work["_positive"] = normalise_labels(work[label_column]).astype("int8")

    statistics = (
        work.groupby(list(group_columns), dropna=False, sort=False)
        .agg(group_size=("_positive", "size"), positive_labels=("_positive", "sum"))
        .reset_index()
    )
    statistics["group_size"] = statistics["group_size"].astype("int64")
    statistics["positive_labels"] = statistics["positive_labels"].astype("int64")
    statistics["negative_labels"] = statistics["group_size"] - statistics["positive_labels"]
    statistics["representation_rate"] = (statistics["group_size"] / len(work)).astype("float64")
    statistics["positive_label_rate"] = (statistics["positive_labels"] / statistics["group_size"]).astype("float64")

    return statistics.sort_values(list(group_columns), na_position="last", ignore_index=True)


def group_tuple(group_value, width):
    """Normalisiert einen Gruppenschlüssel und prüft seine Länge und Vollständigkeit."""

    if width == 1 and not isinstance(group_value, tuple):
        result = (group_value,)
    elif isinstance(group_value, tuple) and len(group_value) == width:
        result = group_value
    else:
        raise MetricInputError(f"Expected a group value with {width} code(s), got {group_value!r}.")

    if any(pd.isna(value) for value in result):
        raise MetricInputError("Requested pairwise group codes must not be missing.")
    return result

def pairwise_from_statistics(statistics, *, grouping_name, group_columns, pairs=None):
    """Leitet Paarmetriken ohne erneute Gruppenaggregation ab."""

    if len(group_columns) not in (1, 2):
        raise MetricInputError("The consolidated SQL/Pandas analysis supports one or two group columns.")
    group_rows = {
        tuple(int(row[column]) for column in group_columns): row
        for _, row in statistics.dropna(subset=list(group_columns)).iterrows()
    }
    if pairs is None:
        selected_pairs = combinations(group_rows, 2)
    else:
        selected_pairs = (
            (group_tuple(reference, len(group_columns)), group_tuple(comparison, len(group_columns)))
            for reference, comparison in pairs
        )

    rows = []
    for reference_codes, comparison_codes in selected_pairs:
        if reference_codes == comparison_codes:
            raise MetricInputError("Reference and comparison group must be different.")

        reference = group_rows.get(reference_codes)
        comparison = group_rows.get(comparison_codes)
        reference_rate = None if reference is None else float(reference["positive_label_rate"])
        comparison_rate = None if comparison is None else float(comparison["positive_label_rate"])
        spd = None if reference_rate is None or comparison_rate is None else comparison_rate - reference_rate
        dir_ratio = None if spd is None or reference_rate == 0 else comparison_rate / reference_rate

        rows.append(
            {
                "grouping_name": grouping_name,
                "reference_code_1": reference_codes[0],
                "reference_code_2": reference_codes[1] if len(reference_codes) == 2 else None,
                "comparison_code_1": comparison_codes[0],
                "comparison_code_2": comparison_codes[1] if len(comparison_codes) == 2 else None,
                "reference_group_size": 0 if reference is None else int(reference["group_size"]),
                "comparison_group_size": 0 if comparison is None else int(comparison["group_size"]),
                "reference_positive_label_rate": reference_rate,
                "comparison_positive_label_rate": comparison_rate,
                "statistical_parity_difference": spd,
                "disparate_impact_ratio": dir_ratio,
            }
        )

    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def pairwise_metrics(frame, *, grouping_name, group_columns, pairs=None, label_column="label"):
    """Berechnet ausgewählte oder alle beobachteten vollständigen Paare."""

    return pairwise_from_statistics(
        group_statistics(frame, group_columns, label_column=label_column),
        grouping_name=grouping_name, group_columns=group_columns, pairs=pairs,
    )


def multigroup_from_statistics(statistics, *, grouping_name, group_columns):
    """Leitet DPD und DPR ohne erneute Gruppenaggregation ab."""

    if grouping_name not in GROUPINGS:
        raise MetricInputError(f"Unknown grouping: {grouping_name!r}.")
    if list(group_columns) != GROUPINGS[grouping_name]:
        raise MetricInputError(f"Grouping {grouping_name!r} requires columns {GROUPINGS[grouping_name]!r}.")
    if len(group_columns) not in (1, 2):
        raise MetricInputError("The consolidated SQL/Pandas analysis supports one or two group columns.")

    statistics = statistics.dropna(subset=list(group_columns))

    exact_rates = []
    for _, row in statistics.iterrows():
        group = tuple(int(row[column]) for column in group_columns)
        rate = Fraction(int(row["positive_labels"]), int(row["group_size"]))
        exact_rates.append((group, rate))

    exact_rates.sort(key=lambda item: item[0])
    included_group_count = len(exact_rates)
    if included_group_count == 0:
        return MultigroupMetrics(
            grouping_name=grouping_name,
            included_group_count=0,
            minimum_positive_label_rate=None,
            maximum_positive_label_rate=None,
            minimum_groups=(),
            maximum_groups=(),
            demographic_parity_difference=None,
            demographic_parity_ratio=None,
        )

    minimum_rate = min(rate for _, rate in exact_rates)
    maximum_rate = max(rate for _, rate in exact_rates)
    minimum_groups = tuple(group for group, rate in exact_rates if rate == minimum_rate)
    maximum_groups = tuple(group for group, rate in exact_rates if rate == maximum_rate)

    if included_group_count < 2:
        difference = None
        ratio = None
    else:
        difference = float(maximum_rate - minimum_rate)
        ratio = None if maximum_rate == 0 else float(minimum_rate / maximum_rate)

    return MultigroupMetrics(
        grouping_name=grouping_name,
        included_group_count=included_group_count,
        minimum_positive_label_rate=float(minimum_rate),
        maximum_positive_label_rate=float(maximum_rate),
        minimum_groups=minimum_groups,
        maximum_groups=maximum_groups,
        demographic_parity_difference=difference,
        demographic_parity_ratio=ratio,
    )


def multigroup_metrics(frame, *, grouping_name, group_columns, label_column="label"):
    """Berechnet DPD und DPR über beobachtete vollständige Gruppen."""

    if grouping_name not in GROUPINGS:
        raise MetricInputError(f"Unknown grouping: {grouping_name!r}.")
    if list(group_columns) != GROUPINGS[grouping_name]:
        raise MetricInputError(f"Grouping {grouping_name!r} requires columns {GROUPINGS[grouping_name]!r}.")
    if len(group_columns) not in (1, 2):
        raise MetricInputError("The consolidated SQL/Pandas analysis supports one or two group columns.")

    statistics = group_statistics(frame, group_columns, label_column=label_column)
    return multigroup_from_statistics(
        statistics,
        grouping_name=grouping_name,
        group_columns=group_columns,
    )


def benchmark_pandas_bundle(frame, *, grouping_name, metric_bundle):
    """Materialisiert den angeforderten Benchmarkumfang aus einer Aggregation."""

    if grouping_name not in BENCHMARK_GROUPINGS:
        raise BenchmarkError(f"Benchmark grouping is not enabled: {grouping_name!r}.")
    if metric_bundle not in BENCHMARK_METRIC_BUNDLES:
        raise BenchmarkError(f"Unknown benchmark metric bundle: {metric_bundle!r}.")

    group_columns = GROUPINGS[grouping_name]
    statistics = group_statistics(frame, group_columns)
    group_rows = statistics.rename(
        columns={column: f"group_code_{index}" for index, column in enumerate(group_columns, 1)}
    )
    if len(group_columns) == 1:
        group_rows.insert(1, "group_code_2", pd.Series(pd.NA, index=group_rows.index, dtype="Int16"))
    group_rows.insert(0, "grouping_name", grouping_name)
    group_rows = normalise_codes(group_rows, ["group_code_1", "group_code_2"])

    result = {
        "group_statistics": json_records(group_rows[BENCHMARK_GROUP_COLUMNS]),
        "pairwise_metrics": [],
        "multigroup_metrics": [],
    }
    if metric_bundle in ("Stufe 2", "Stufe 3"):
        pairs = pairwise_from_statistics(
            statistics,
            grouping_name=grouping_name,
            group_columns=group_columns,
        )
        pairs = normalise_codes(
            pairs,
            ["reference_code_1", "reference_code_2", "comparison_code_1", "comparison_code_2"],
        )
        result["pairwise_metrics"] = json_records(pairs[PAIRWISE_COLUMNS])
    if metric_bundle == "Stufe 3":
        result["multigroup_metrics"] = [
            multigroup_from_statistics(
                statistics,
                grouping_name=grouping_name,
                group_columns=group_columns,
            ).to_dict()
        ]
    return result


# Abgleich von PostgreSQL und Pandas
def database_url():
    """Liest die Verbindungsangaben für PostgreSQL aus den Umgebungsvariablen und .env."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    required = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_PORT"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError("Database configuration missing. Set DATABASE_URL or: " + ", ".join(missing))
    return make_conninfo(
        host="localhost",
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def fetch_data(connection, query, parameters=()):
    """Führt eine parametrisierte SQL-Abfrage aus und liefert die Zeilen als DataFrame."""

    with connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [description.name for description in cursor.description]
    return pd.DataFrame(rows, columns=columns)

def normalise_codes(frame, columns):
    """Konvertiert Gruppencodes in den fehlwertfähigen Datentyp Int16."""

    return frame.astype({column: "Int16" for column in columns})


def compare_metrics(sql_frame, pandas_frame):
    """Prüft die Übereinstimmung zweier Ergebnistabellen mit der festgelegten Toleranz."""

    pd.testing.assert_frame_equal(
        sql_frame.reset_index(drop=True),
        pandas_frame.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=FLOAT_TOLERANCE,
        atol=FLOAT_TOLERANCE,
    )

def compare_groups(connection, raw, *, task_id, grouping_name, pandas_group_columns):
    """Vergleicht SQL- und Pandas-Gruppenstatistiken und liefert die Zahl geprüfter Gruppen."""

    pandas_statistics = group_statistics(raw, pandas_group_columns).rename(columns={pandas_group_columns[0]: "group_code_1"})

    if len(pandas_group_columns) == 2:
        pandas_statistics = pandas_statistics.rename(columns={pandas_group_columns[1]: "group_code_2"})
    else:
        pandas_statistics.insert(1, "group_code_2", pd.Series(pd.NA, index=pandas_statistics.index, dtype="Int16"))

    sql_statistics = fetch_data(
        connection,
        """
        SELECT
            group_code_1,
            group_code_2,
            group_size,
            positive_labels,
            negative_labels,
            representation_rate,
            positive_label_rate
        FROM fairness.group_statistics
        WHERE task_id = %s
          AND grouping_name = %s
        ORDER BY group_code_1 NULLS LAST, group_code_2 NULLS LAST
        """,
        (task_id, grouping_name),
    )

    columns = [
        "group_code_1",
        "group_code_2",
        "group_size",
        "positive_labels",
        "negative_labels",
        "representation_rate",
        "positive_label_rate",
    ]
    code_columns = ["group_code_1", "group_code_2"]
    sql_statistics = normalise_codes(sql_statistics[columns], code_columns)
    pandas_statistics = normalise_codes(pandas_statistics[columns], code_columns)
    compare_metrics(sql_statistics, pandas_statistics)
    return len(sql_statistics)


def compare_pairs(connection, raw, *, task_id, grouping_name, pandas_group_columns):
    """Vergleicht SQL- und Pandas-Paarmetriken und liefert die Zahl geprüfter Paare."""

    pandas_pairs = pairwise_metrics(raw, grouping_name=grouping_name, group_columns=pandas_group_columns)
    sql_pairs = fetch_data(connection, "SELECT * FROM fairness.all_pairwise_metrics(%s, %s)", (task_id, grouping_name))
    code_columns = ["reference_code_1", "reference_code_2", "comparison_code_1", "comparison_code_2"]
    sql_pairs = normalise_codes(sql_pairs[PAIRWISE_COLUMNS], code_columns)
    pandas_pairs = normalise_codes(pandas_pairs[PAIRWISE_COLUMNS], code_columns)
    compare_metrics(sql_pairs, pandas_pairs)
    return len(sql_pairs)


def compare_multigroup(connection, raw, *, task_id, grouping_name, pandas_group_columns):
    """Vergleicht die Mehrgruppenwerte aus SQL und Pandas und liefert die Pandas-Werte."""

    pandas_metrics = multigroup_metrics(raw, grouping_name=grouping_name, group_columns=pandas_group_columns)
    sql_metrics = fetch_data(connection, "SELECT * FROM fairness.multigroup_metrics(%s, %s)", (task_id, grouping_name))
    if len(sql_metrics) != 1:
        raise AssertionError("PostgreSQL multigroup_metrics must return exactly one row.")

    pandas_frame = pd.DataFrame([pandas_metrics.to_dict()], columns=MULTIGROUP_COLUMNS)
    compare_metrics(sql_metrics[MULTIGROUP_COLUMNS], pandas_frame)
    return pandas_metrics

def validate(connection, task_ids=TASK_IDS):
    """Vergleicht SQL und Pandas je ausgewählter Aufgabe."""

    selected = list(task_ids)
    unknown = sorted(set(selected).difference(TASK_DEFINITIONS))
    if not selected or unknown:
        raise MetricInputError(f"Validation requires at least one confirmed task; unknown tasks: {unknown}")

    task_results = {}
    total_rows = 0
    for task_id in selected:
        with connection.cursor() as cursor:
            cursor.execute("SELECT fairness.assert_training_data(%s)", (task_id,))

        raw = fetch_data(
            connection,
            """
            SELECT sample_rank, sex, race, label
            FROM fairness.training_data
            WHERE task_id = %s
            ORDER BY sample_rank
            """,
            (task_id,),
        )

        group_counts = {}
        pair_counts = {}
        multigroup_results = {}
        for grouping_name, group_columns in GROUPINGS.items():
            group_counts[grouping_name] = compare_groups(
                connection,
                raw,
                task_id=task_id,
                grouping_name=grouping_name,
                pandas_group_columns=group_columns,
            )
            pair_counts[grouping_name] = compare_pairs(
                connection,
                raw,
                task_id=task_id,
                grouping_name=grouping_name,
                pandas_group_columns=group_columns,
            )
            multigroup_results[grouping_name] = compare_multigroup(
                connection,
                raw,
                task_id=task_id,
                grouping_name=grouping_name,
                pandas_group_columns=group_columns,
            ).to_dict()

        first_sex_pair = json_records(
            pairwise_metrics(raw, grouping_name="sex", group_columns=["sex"], pairs=[(1, 2)])
        )[0]
        task_results[task_id] = {
            "rows_compared": len(raw),
            "groups_compared": group_counts,
            "pairwise_comparisons": pair_counts,
            "multigroup_metrics": multigroup_results,
            "documented_sex_pair": {
                "reference_group": first_sex_pair["reference_code_1"],
                "comparison_group": first_sex_pair["comparison_code_1"],
                "reference_group_size": first_sex_pair["reference_group_size"],
                "comparison_group_size": first_sex_pair["comparison_group_size"],
                "reference_positive_label_rate": first_sex_pair["reference_positive_label_rate"],
                "comparison_positive_label_rate": first_sex_pair["comparison_positive_label_rate"],
                "statistical_parity_difference": first_sex_pair["statistical_parity_difference"],
                "disparate_impact_ratio": first_sex_pair["disparate_impact_ratio"],
            },
        }
        total_rows += len(raw)

    return {
        "rows_compared": total_rows,
        "tasks": task_results,
        "status": "SQL and Pandas results agree",
        "float_tolerance": FLOAT_TOLERANCE,
    }

def json_records(frame):
    """Wandelt Ergebniszeilen ohne Rundung in JSON um."""

    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")

def collect_results(connection, task_ids=TASK_IDS):
    """Prüft und sammelt alle aktuellen Ergebnisse."""

    selected = list(task_ids)
    validation = validate(connection, selected)
    task_results = {}

    for task_id in selected:
        group_statistics = fetch_data(
            connection,
            """
            SELECT
                grouping_name,
                group_code_1,
                group_code_2,
                group_size,
                positive_labels,
                negative_labels,
                representation_rate,
                positive_label_rate
            FROM fairness.group_statistics
            WHERE task_id = %s
            ORDER BY
                CASE grouping_name
                    WHEN 'sex' THEN 1
                    WHEN 'race' THEN 2
                    WHEN 'sex_race' THEN 3
                END,
                group_code_1 NULLS LAST,
                group_code_2 NULLS LAST
            """,
            (task_id,),
        )
        pairwise_metrics = {}
        multigroup_metrics = {}
        for grouping_name in GROUPINGS:
            pairs = fetch_data(connection, "SELECT * FROM fairness.all_pairwise_metrics(%s, %s)", (task_id, grouping_name))
            multigroup = fetch_data(connection, "SELECT * FROM fairness.multigroup_metrics(%s, %s)", (task_id, grouping_name))
            if len(multigroup) != 1:
                raise AssertionError("PostgreSQL multigroup_metrics must return exactly one row.")
            pairwise_metrics[grouping_name] = json_records(pairs[PAIRWISE_COLUMNS])
            multigroup_metrics[grouping_name] = json_records(multigroup[MULTIGROUP_COLUMNS])[0]

        task_validation = validation["tasks"][task_id]
        task_results[task_id] = {
            "dataset": TASK_DEFINITIONS[task_id]["dataset_name"],
            "population_definition": TASK_DEFINITIONS[task_id]["population_definition"],
            "positive_label_definition": TASK_DEFINITIONS[task_id]["positive_label_definition"],
            "rows": task_validation["rows_compared"],
            "validated_group_counts": task_validation["groups_compared"],
            "validated_pair_counts": task_validation["pairwise_comparisons"],
            "group_statistics": json_records(group_statistics),
            "pairwise_metrics": pairwise_metrics,
            "multigroup_metrics": multigroup_metrics,
        }

    return {
        "status": validation["status"],
        "float_tolerance": validation["float_tolerance"],
        "rows_compared": validation["rows_compared"],
        "result_scope": {
            "data_level_before_model_training": True,
            "group_statistics_included": True,
            "pairwise_spd_dir_included": True,
            "multigroup_dpd_dpr_included": True,
            "benchmark_results_included": False,
        },
        "tasks": task_results,
    }

def report_provenance(report, metadata_path, *, database):
    """Prüft Metadaten und erstellt eine portable Berichtsherkunft."""

    if not metadata_path.is_file():
        raise ReportError(f"Preparation metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Preparation metadata cannot be read: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ReportError("Preparation metadata must contain one JSON object.")

    required = {
        "dataset", "source", "source_url", "survey_year", "horizon", "survey", "states", "seed",
        "analysis_weighting", "person_weight_retained", "task_ids", "tasks", "columns", "raw_rows",
        "prepared_rows", "folktables_version", "pandas_version", "python_version", "csv_path", "csv_sha256",
        "age_source_variable", "raw_missing_age",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ReportError(f"Preparation metadata is missing fields: {missing}")
    if metadata["age_source_variable"] != "AGEP" or "age" not in metadata["columns"]:
        raise ReportError("Preparation metadata must identify the retained AGEP/age source column.")

    report_task_ids = list(report["tasks"])
    metadata_task_ids = list(metadata["task_ids"])
    if len(metadata_task_ids) != len(set(metadata_task_ids)) or set(metadata_task_ids) != set(report_task_ids):
        raise ReportError("Preparation metadata and report contain different task IDs.")
    if int(metadata["prepared_rows"]) != int(report["rows_compared"]):
        raise ReportError("Preparation metadata and report contain different total row counts.")

    for task_id in report_task_ids:
        task_metadata = metadata["tasks"].get(task_id)
        task_report = report["tasks"][task_id]
        if not isinstance(task_metadata, dict):
            raise ReportError(f"Preparation metadata is missing task details for {task_id}.")
        task_rows = int(task_metadata.get("prepared_rows", -1))
        positive_labels = int(task_metadata.get("positive_labels", -1))
        negative_labels = int(task_metadata.get("negative_labels", -1))
        if task_rows != int(task_report["rows"]) or positive_labels + negative_labels != task_rows:
            raise ReportError(f"Preparation metadata and report disagree on rows or labels for {task_id}.")
        if task_metadata.get("dataset") != task_report["dataset"] or task_metadata.get("positive_label_definition") != task_report["positive_label_definition"]:
            raise ReportError(f"Preparation metadata and report disagree on the task contract for {task_id}.")
        if task_metadata.get("population_definition", task_report["population_definition"]) != task_report["population_definition"]:
            raise ReportError(f"Preparation metadata and report disagree on the population contract for {task_id}.")

        statistics = task_report["group_statistics"]
        for grouping, label in (("sex", "SEX"), ("race", "RAC1P")):
            rows = [row for row in statistics if row["grouping_name"] == grouping]
            if not rows:
                raise ReportError(f"Report lacks required group statistics for {task_id}.")
            for column, expected in (("group_size", task_rows), ("positive_labels", positive_labels), ("negative_labels", negative_labels)):
                if sum(int(row[column]) for row in rows) != expected:
                    raise ReportError(f"{label} {column} counts disagree with preparation metadata for {task_id}.")
            missing_count = sum(int(row["group_size"]) for row in rows if row["group_code_1"] is None)
            if missing_count != int(task_metadata.get(f"missing_{grouping}", -1)):
                raise ReportError(f"Missing {label} counts disagree with preparation metadata for {task_id}.")

    recorded_csv_path = Path(str(metadata["csv_path"]))
    sibling_csv_path = metadata_path.with_name(metadata_path.name.removesuffix(".metadata.json") + ".csv")
    candidates = [recorded_csv_path] if recorded_csv_path.is_absolute() else [PROJECT_ROOT / recorded_csv_path, metadata_path.parent / recorded_csv_path]
    candidates.append(sibling_csv_path)
    csv_path = next((path for path in candidates if path.is_file()), None)
    if csv_path is None:
        raise ReportError(f"Prepared CSV referenced by {metadata_path} was not found.")
    expected_hash = str(metadata["csv_sha256"]).lower()
    if file_hash(csv_path) != expected_hash:
        raise ReportError("Prepared CSV does not match the SHA-256 stored in its metadata.")

    def portable_path(path):
        """Liefert einen Pfad relativ zum Projekt oder andernfalls nur den Dateinamen."""

        try:
            return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return path.name

    task_metadata = {task_id: dict(metadata["tasks"][task_id]) for task_id in report_task_ids}
    for task_id in report_task_ids:
        task_metadata[task_id].setdefault("population_definition", report["tasks"][task_id]["population_definition"])

    return {
        "metadata_file": portable_path(metadata_path),
        "data_source": {
            "dataset": metadata["dataset"],
            "source": metadata["source"],
            "source_url": metadata["source_url"],
            "survey_year": metadata["survey_year"],
            "horizon": metadata["horizon"],
            "survey": metadata["survey"],
            "states": metadata["states"],
        },
        "preparation": {
            "seed": metadata["seed"],
            "analysis_weighting": metadata["analysis_weighting"],
            "person_weight_retained": metadata["person_weight_retained"],
            "raw_rows": metadata["raw_rows"],
            "raw_missing_age": metadata["raw_missing_age"],
            "age_source_variable": metadata["age_source_variable"],
            "prepared_rows": metadata["prepared_rows"],
            "task_ids": metadata_task_ids,
            "columns": metadata["columns"],
            "tasks": task_metadata,
        },
        "prepared_data_file": {"path": portable_path(csv_path), "sha256": expected_hash},
        "software": {
            "preparation": {
                "python": metadata["python_version"],
                "pandas": metadata["pandas_version"],
                "folktables": metadata["folktables_version"],
            },
            "reporting": {
                "python": platform.python_version(),
                "pandas": package_version("pandas"),
                "psycopg": package_version("psycopg"),
            },
        },
        "database": database,
        "verification": {
            "prepared_csv_sha256_verified": True,
            "report_aggregates_match_preparation_metadata": True,
            "database_content_hash_available": False,
            "database_linkage_limitation": "Database aggregates match the preparation metadata, but no content hash is stored inside the database.",
        },
    }


def database_info(connection):
    """Liefert PostgreSQL-Version und Bezeichnung der verwendeten Datentabelle."""

    with connection.cursor() as cursor:
        cursor.execute("SHOW server_version")
        return {
            "system": "PostgreSQL",
            "server_version": str(cursor.fetchone()[0]),
            "schema": "fairness",
            "relation": "fairness.training_data",
        }


def load_scaling_metadata():
    """Lädt und prüft den unveränderlichen Datenstand der Skalierung."""

    if not DEFAULT_SCALING_METADATA_PATH.is_file() or not DEFAULT_SCALING_CSV_PATH.is_file():
        raise BenchmarkError("Scaling data are missing. Run prepare-scaling first.")
    try:
        metadata = json.loads(DEFAULT_SCALING_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError("Scaling metadata cannot be read.") from exc

    required = {
        "benchmark_sizes", "csv_sha256", "folktables_version", "pandas_version", "prepared_rows",
        "python_version", "region_count", "regions", "reporting_groupings", "seed", "states", "task_ids", "tasks",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise BenchmarkError(f"Scaling metadata are missing fields: {missing}")
    if tuple(metadata["states"]) != BENCHMARK_STATES or int(metadata["region_count"]) != len(BENCHMARK_STATES):
        raise BenchmarkError("Scaling metadata contain a different Folktables region scope.")
    if tuple(metadata["reporting_groupings"]) != BENCHMARK_GROUPINGS:
        raise BenchmarkError("Scaling metadata and benchmark groupings disagree.")
    if tuple(metadata["task_ids"]) != TASK_IDS or set(metadata["tasks"]) != set(TASK_IDS):
        raise BenchmarkError("Scaling metadata and confirmed task order disagree.")

    sizes = metadata["benchmark_sizes"]
    if tuple(sizes.get("fixed_candidate_sizes", [])) != BENCHMARK_FIXED_SIZES:
        raise BenchmarkError("Scaling metadata and fixed benchmark sizes disagree.")
    task_sizes = {task_id: int(metadata["tasks"][task_id]["prepared_rows"]) for task_id in TASK_IDS}
    if {task_id: int(value) for task_id, value in sizes.get("full_sizes", {}).items()} != task_sizes:
        raise BenchmarkError("Scaling metadata contain inconsistent task sizes.")
    expected_plan = benchmark_size_plan(task_sizes)
    if sizes != expected_plan:
        raise BenchmarkError("Stored benchmark stages do not match the predetermined size rule.")
    if sum(task_sizes.values()) != int(metadata["prepared_rows"]):
        raise BenchmarkError("Scaling task sizes do not add up to prepared_rows.")
    if file_hash(DEFAULT_SCALING_CSV_PATH) != str(metadata["csv_sha256"]).lower():
        raise BenchmarkError("Scaling CSV and metadata SHA-256 disagree.")
    return metadata


def load_scaling_validation(metadata):
    """Prüft den gespeicherten SQL Pandas Nachweis des Gesamtbestands."""

    if not DEFAULT_SCALING_VALIDATION_PATH.is_file():
        raise BenchmarkError("Full-region validation evidence is missing. Run validate and preserve its JSON output first.")
    try:
        validation = json.loads(DEFAULT_SCALING_VALIDATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError("Full-region validation evidence cannot be read.") from exc
    if (
        validation.get("status") != "SQL and Pandas results agree"
        or int(validation.get("rows_compared", -1)) != int(metadata["prepared_rows"])
        or float(validation.get("float_tolerance", math.inf)) != FLOAT_TOLERANCE
        or set(validation.get("tasks", {})) != set(TASK_IDS)
    ):
        raise BenchmarkError("Full-region validation evidence does not match the scaling data.")
    for task_id in TASK_IDS:
        if int(validation["tasks"][task_id].get("rows_compared", -1)) != int(metadata["tasks"][task_id]["prepared_rows"]):
            raise BenchmarkError(f"Full-region validation row count differs for {task_id}.")
    return validation


def benchmark_metric_specs(metadata, *, pilot):
    """Erzeugt fachliche Konfigurationen und führt Überschneidungen zusammen."""

    sizes = metadata["benchmark_sizes"]
    if pilot:
        common_size = BENCHMARK_FIXED_SIZES[0]
        if any(int(sizes["full_sizes"][task_id]) < common_size for task_id in TASK_IDS):
            raise BenchmarkError("The technical pilot size is not available in every task.")
        z_entries = [(task_id, common_size) for task_id in TASK_IDS]
        comparison_tasks = ("acs_income",)
    else:
        common_size = int(sizes["n_common"])
        z_entries = [(task_id, size) for task_id in TASK_IDS for size in sizes["stages"][task_id]]
        comparison_tasks = TASK_IDS

    registrations = (
        [("Z", task_id, size, "sex_race", "Stufe 3") for task_id, size in z_entries]
        + [("G", task_id, common_size, grouping, "Stufe 3") for task_id in comparison_tasks for grouping in BENCHMARK_GROUPINGS]
        + [("M", task_id, common_size, "sex_race", bundle) for task_id in comparison_tasks for bundle in BENCHMARK_METRIC_BUNDLES]
    )

    specs = {}
    for experiment, task_id, dataset_size, grouping, metric_bundle in registrations:
        dataset_size = int(dataset_size)
        key = (task_id, dataset_size, grouping, metric_bundle)
        spec = specs.setdefault(key, {
            "task_id": task_id, "dataset_size": dataset_size, "grouping": grouping,
            "metric_bundle": metric_bundle, "experiments": [],
        })
        spec["experiments"].append(experiment)

    return list(specs.values())


def benchmark_configurations(specs):
    """Erweitert vorhandene fachliche Spezifikationen um Implementierung und Messgrenze."""

    configurations = [
        {
            **spec,
            "experiment": "|".join(spec["experiments"]),
            "implementation": implementation,
            "measurement_scope": scope,
            "configuration_id": (
                f"{implementation}_{scope}_{spec['task_id']}_n{spec['dataset_size']}_"
                f"{spec['grouping']}_{BENCHMARK_METRIC_BUNDLE_SLUGS[spec['metric_bundle']]}"
            ),
        }
        for spec in specs
        for implementation in ("sql", "pandas")
        for scope in ("compute_only", "end_to_end")
    ]
    identifiers = [configuration["configuration_id"] for configuration in configurations]
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkError("Benchmark configuration IDs are not unique.")
    return configurations


def benchmark_execution_schedule(configurations, *, warmups, measured_blocks, seed):
    """Plant Warm-ups und blockweise randomisierte Messungen reproduzierbar."""

    if warmups < 0 or measured_blocks <= 0 or not configurations:
        raise BenchmarkError("Benchmark schedule requires configurations and valid repetition counts.")
    schedule = []
    warmup_random = random.Random(seed + 1)
    measurement_random = random.Random(seed)
    phases = [("warmup", warmups, warmup_random), ("measured", measured_blocks, measurement_random)]
    for phase, repetitions, generator in phases:
        for round_number in range(1, repetitions + 1):
            current = list(configurations)
            generator.shuffle(current)
            schedule.extend(
                {
                    "warmup_or_measured": phase,
                    "round": round_number,
                    "execution_order": position,
                    "configuration": configuration,
                }
                for position, configuration in enumerate(current, start=1)
            )
    return schedule


def sql_benchmark_result(rows):
    """Ordnet die normalisierten SQL Ergebniszeilen den drei Umfängen zu."""

    result = {"group_statistics": [], "pairwise_metrics": [], "multigroup_metrics": []}
    for row in rows:
        result_type = row[5]
        payload = row[6]
        if result_type not in result or not isinstance(payload, dict):
            raise BenchmarkError(f"Unexpected SQL benchmark result type: {result_type!r}.")
        result[result_type].append(payload)
    return result


def assert_nested_close(actual, expected, *, path="result"):
    """Vergleicht verschachtelte Benchmarkobjekte mit der Metriktoleranz."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise BenchmarkError(f"{path} contains different keys: {set(actual) ^ set(expected)}")
        for key in actual:
            assert_nested_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise BenchmarkError(f"{path} contains {len(actual)} instead of {len(expected)} items.")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            assert_nested_close(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, int) and isinstance(expected, int):
            equal = int(actual) == int(expected)
        else:
            equal = math.isclose(float(actual), float(expected), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
        if not equal:
            raise BenchmarkError(f"{path} differs: {actual!r} != {expected!r}.")
        return
    if actual != expected:
        raise BenchmarkError(f"{path} differs: {actual!r} != {expected!r}.")


def benchmark_result_counts(result, *, grouping_name):
    """Ermittelt Ergebnisumfang und Zahl vollständiger beobachteter Gruppen."""

    complete_groups = 0
    for row in result["group_statistics"]:
        if row["group_code_1"] is not None and (grouping_name != "sex_race" or row["group_code_2"] is not None):
            complete_groups += 1
    return {
        "observed_group_count": complete_groups,
        "result_group_rows": len(result["group_statistics"]),
        "result_pair_rows": len(result["pairwise_metrics"]),
        "result_multigroup_rows": len(result["multigroup_metrics"]),
    }


def verify_benchmark_database(connection, metadata):
    """Gleicht Datenbankbestände und ANALYZE Status mit den Metadaten ab."""

    rows = fetch_data(
        connection,
        """
        SELECT
            task_id,
            COUNT(*)::BIGINT AS rows,
            COUNT(*) FILTER (WHERE label)::BIGINT AS positive_labels,
            COUNT(*) FILTER (WHERE NOT label)::BIGINT AS negative_labels,
            MIN(sample_rank)::BIGINT AS minimum_rank,
            MAX(sample_rank)::BIGINT AS maximum_rank
        FROM fairness.training_data
        GROUP BY task_id
        ORDER BY task_id
        """,
    )
    actual = {row.task_id: row for row in rows.itertuples(index=False)}
    if set(actual) != set(TASK_IDS):
        raise BenchmarkError("PostgreSQL does not contain exactly the three confirmed tasks.")
    for task_id in TASK_IDS:
        expected = metadata["tasks"][task_id]
        current = actual[task_id]
        if (
            int(current.rows) != int(expected["prepared_rows"])
            or int(current.positive_labels) != int(expected["positive_labels"])
            or int(current.negative_labels) != int(expected["negative_labels"])
            or int(current.minimum_rank) != 1
            or int(current.maximum_rank) != int(current.rows)
        ):
            raise BenchmarkError(f"PostgreSQL and scaling metadata disagree for {task_id}.")

    analyzed = fetch_data(
        connection,
        """
        SELECT last_analyze, n_live_tup
        FROM pg_stat_all_tables
        WHERE relid = 'fairness.training_data'::REGCLASS
        """,
    )
    if len(analyzed) != 1 or pd.isna(analyzed.iloc[0]["last_analyze"]):
        raise BenchmarkError("fairness.training_data has no recorded ANALYZE run.")
    return {
        "tasks": {
            task_id: {
                "rows": int(actual[task_id].rows),
                "positive_labels": int(actual[task_id].positive_labels),
                "negative_labels": int(actual[task_id].negative_labels),
                "sample_rank_range": [int(actual[task_id].minimum_rank), int(actual[task_id].maximum_rank)],
            }
            for task_id in TASK_IDS
        },
        "last_analyze": analyzed.iloc[0]["last_analyze"].isoformat(),
        "estimated_live_rows": int(analyzed.iloc[0]["n_live_tup"]),
    }


def load_benchmark_compute_frames(connection, specs):
    """Lädt je Aufgabe die größte benötigte Pandas Teilmenge vor der Messung."""

    maximum_sizes = {
        task_id: max(spec["dataset_size"] for spec in specs if spec["task_id"] == task_id)
        for task_id in TASK_IDS
    }
    frames = {}
    details = {}
    for task_id in TASK_IDS:
        start_ns = perf_counter_ns()
        frame = fetch_data(
            connection,
            """
            SELECT sample_rank, sex, race, label
            FROM fairness.training_data
            WHERE task_id = %s AND sample_rank <= %s
            ORDER BY sample_rank
            """,
            (task_id, maximum_sizes[task_id]),
        )
        if len(frame) != maximum_sizes[task_id]:
            raise BenchmarkError(f"Pandas compute-only preload is incomplete for {task_id}.")
        if int(frame["sample_rank"].iloc[0]) != 1 or int(frame["sample_rank"].iloc[-1]) != len(frame):
            raise BenchmarkError(f"Pandas compute-only ranks are not contiguous for {task_id}.")
        frame["sample_rank"] = frame["sample_rank"].astype("int64")
        for column in ("sex", "race"):
            frame[column] = frame[column].astype("Int16")
        frame["label"] = frame["label"].astype(bool)
        frames[task_id] = frame
        details[task_id] = {
            "rows": len(frame),
            "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
            "load_duration_ms": (perf_counter_ns() - start_ns) / 1_000_000,
        }
    return frames, details


def prepare_benchmark_relations(connection, specs):
    """Materialisiert jede SQL Teilmenge einmalig außerhalb der Messung."""

    unique_subsets = list(dict.fromkeys((spec["task_id"], spec["dataset_size"]) for spec in specs))
    relations = {}
    details = []
    for index, (task_id, dataset_size) in enumerate(unique_subsets, start=1):
        relation_name = f"benchmark_subset_{index:02d}"
        start_ns = perf_counter_ns()
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "CREATE TEMP TABLE {} ON COMMIT PRESERVE ROWS AS "
                    "SELECT sex, race, label FROM fairness.training_data "
                    "WHERE task_id = %s AND sample_rank <= %s"
                ).format(sql.Identifier(relation_name)),
                (task_id, dataset_size),
            )
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(relation_name)))
            if int(cursor.fetchone()[0]) != dataset_size:
                raise BenchmarkError(f"SQL benchmark subset is incomplete for {task_id} at {dataset_size} rows.")
            cursor.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(relation_name)))
            qualified_name = f"pg_temp.{relation_name}"
            cursor.execute("SELECT pg_total_relation_size(%s::REGCLASS)", (qualified_name,))
            relation_bytes = int(cursor.fetchone()[0])
        relations[(task_id, dataset_size)] = qualified_name
        details.append({
            "task_id": task_id,
            "dataset_size": dataset_size,
            "relation": qualified_name,
            "relation_bytes": relation_bytes,
            "setup_duration_ms": (perf_counter_ns() - start_ns) / 1_000_000,
        })
    return relations, details


def get_benchmark_query(connection, *, source_relation, grouping_name, metric_bundle):
    """Ruft die ausschließlich in der SQL Datei definierte Benchmarkabfrage ab."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT fairness.benchmark_query(%s::REGCLASS, %s, %s)",
            (source_relation, grouping_name, metric_bundle),
        )
        return cursor.fetchone()[0]


def execute_sql_benchmark_query(connection, query):
    """Führt eine Benchmarkabfrage aus und materialisiert alle Ergebniszeilen."""

    with connection.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()


def explain_sql_benchmark_query(connection, query):
    """Führt dieselbe Abfrage mit serverseitiger EXPLAIN Messung aus."""

    start_ns = perf_counter_ns()
    with connection.cursor() as cursor:
        cursor.execute("EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, SUMMARY ON, FORMAT JSON) " + query)
        plan = cursor.fetchone()[0][0]
    wall_duration_ms = (perf_counter_ns() - start_ns) / 1_000_000
    return plan, wall_duration_ms


def validate_benchmark_specs(connection, specs, relations, frames):
    """Vergleicht jeden eindeutigen Benchmarkumfang vor der ersten Zeitmessung."""

    queries = {}
    result_details = {}
    for spec in specs:
        key = (spec["task_id"], spec["dataset_size"], spec["grouping"], spec["metric_bundle"])
        source_relation = relations[(spec["task_id"], spec["dataset_size"])]
        query = get_benchmark_query(
            connection,
            source_relation=source_relation,
            grouping_name=spec["grouping"],
            metric_bundle=spec["metric_bundle"],
        )
        subset = frames[spec["task_id"]].iloc[:spec["dataset_size"]]
        pandas_result = benchmark_pandas_bundle(
            subset,
            grouping_name=spec["grouping"],
            metric_bundle=spec["metric_bundle"],
        )
        sql_result = sql_benchmark_result(execute_sql_benchmark_query(connection, query))
        assert_nested_close(sql_result, pandas_result, path="benchmark_result")
        queries[key] = query
        result_details[key] = benchmark_result_counts(pandas_result, grouping_name=spec["grouping"])
    return queries, result_details


def host_environment():
    """Erfasst die grundlegenden Angaben des Untersuchungsrechners."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def docker_environment():
    """Erfasst Version und Ressourcen der Docker VM."""

    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        info = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return {
        "server_version": info.get("ServerVersion"),
        "operating_system": info.get("OperatingSystem"),
        "architecture": info.get("Architecture"),
        "cpu_count": info.get("NCPU"),
        "memory_bytes": info.get("MemTotal"),
    }


def database_benchmark_environment(connection):
    """Erfasst Version und ausgewählte laufzeitrelevante PostgreSQL Parameter."""

    settings = [
        "shared_buffers",
        "work_mem",
        "maintenance_work_mem",
        "effective_cache_size",
        "max_parallel_workers",
        "max_parallel_workers_per_gather",
        "jit",
        "random_page_cost",
        "effective_io_concurrency",
    ]
    values = {}
    with connection.cursor() as cursor:
        for setting in settings:
            cursor.execute(sql.SQL("SHOW {}").format(sql.Identifier(setting)))
            values[setting] = str(cursor.fetchone()[0])
    return {**database_info(connection), "settings": values}


def benchmark_output_paths(*, pilot):
    """Bestimmt getrennte Ausgabepfade für Pilot und Hauptmessung."""

    if pilot:
        return {
            "raw": PROJECT_ROOT / "results" / "raw" / "benchmark_pilot_runs.csv",
            "environment": PROJECT_ROOT / "results" / "raw" / "benchmark_pilot_environment.json",
            "plans": DEFAULT_BENCHMARK_PLAN_ROOT / "pilot",
            "summary": PROJECT_ROOT / "results" / "tables" / "benchmark_pilot_summary.csv",
        }
    return {
        "raw": DEFAULT_BENCHMARK_RAW_PATH,
        "environment": DEFAULT_BENCHMARK_ENVIRONMENT_PATH,
        "plans": DEFAULT_BENCHMARK_PLAN_ROOT / "main",
        "summary": DEFAULT_BENCHMARK_SUMMARY_PATH,
    }


def prepare_benchmark_output_paths(paths):
    """Verhindert, dass vorhandene Rohmessungen unbemerkt überschrieben werden."""

    partial = paths["raw"].with_suffix(".partial.csv")
    occupied = [path for path in (paths["raw"], partial, paths["environment"], paths["summary"]) if path.exists()]
    if paths["plans"].exists() and any(paths["plans"].iterdir()):
        occupied.append(paths["plans"])
    if occupied:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in occupied)
        raise BenchmarkError(f"Benchmark output already exists and will not be overwritten: {names}")
    for path in (paths["raw"], paths["environment"], paths["summary"]):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["plans"].mkdir(parents=True, exist_ok=True)
    return partial


def plan_measurement(plan, *, client_wall_duration_ms):
    """Extrahiert die serverseitige Zeit aus dem PostgreSQL Plan."""

    return {
        "duration_ms": float(plan["Execution Time"]),
        "client_wall_duration_ms": float(client_wall_duration_ms),
        "sql_planning_time_ms": float(plan.get("Planning Time", 0.0)),
    }


def pandas_end_to_end(connection, configuration):
    """Misst Rohdatenabruf, DataFrame Bildung und vollständige Pandas Berechnung."""

    start_ns = perf_counter_ns()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT sex, race, label
            FROM fairness.training_data
            WHERE task_id = %s AND sample_rank <= %s
            ORDER BY sample_rank
            """,
            (configuration["task_id"], configuration["dataset_size"]),
        )
        rows = cursor.fetchall()
        columns = [description.name for description in cursor.description]
    frame = pd.DataFrame(rows, columns=columns)
    if len(frame) != configuration["dataset_size"]:
        raise BenchmarkError(f"Pandas end-to-end input is incomplete for {configuration['configuration_id']}.")
    result = benchmark_pandas_bundle(
        frame,
        grouping_name=configuration["grouping"],
        metric_bundle=configuration["metric_bundle"],
    )
    duration_ms = (perf_counter_ns() - start_ns) / 1_000_000
    return result, duration_ms


def execute_benchmark_configuration(connection, configuration, *, frames, queries, expected_details):
    """Führt genau eine vorab eingerichtete Messkonfiguration aus."""

    key = (
        configuration["task_id"],
        configuration["dataset_size"],
        configuration["grouping"],
        configuration["metric_bundle"],
    )
    expected = expected_details[key]
    plan = None

    if configuration["implementation"] == "sql":
        query = queries[key]
        if configuration["measurement_scope"] == "compute_only":
            plan, client_wall_duration_ms = explain_sql_benchmark_query(connection, query)
            measurement = plan_measurement(plan, client_wall_duration_ms=client_wall_duration_ms)
        else:
            start_ns = perf_counter_ns()
            rows = execute_sql_benchmark_query(connection, query)
            duration_ms = (perf_counter_ns() - start_ns) / 1_000_000
            actual_rows = len(rows)
            expected_rows = expected["result_group_rows"] + expected["result_pair_rows"] + expected["result_multigroup_rows"]
            if actual_rows != expected_rows:
                raise BenchmarkError(f"SQL result row count changed for {configuration['configuration_id']}.")
            measurement = {
                "duration_ms": duration_ms,
                "client_wall_duration_ms": duration_ms,
                "sql_planning_time_ms": None,
            }
    elif configuration["implementation"] == "pandas":
        if configuration["measurement_scope"] == "compute_only":
            subset = frames[configuration["task_id"]].iloc[:configuration["dataset_size"]]
            start_ns = perf_counter_ns()
            result = benchmark_pandas_bundle(
                subset,
                grouping_name=configuration["grouping"],
                metric_bundle=configuration["metric_bundle"],
            )
            duration_ms = (perf_counter_ns() - start_ns) / 1_000_000
        else:
            result, duration_ms = pandas_end_to_end(connection, configuration)
        actual = benchmark_result_counts(result, grouping_name=configuration["grouping"])
        if actual != expected:
            raise BenchmarkError(f"Pandas result shape changed for {configuration['configuration_id']}.")
        measurement = {
            "duration_ms": duration_ms,
            "client_wall_duration_ms": duration_ms,
            "sql_planning_time_ms": None,
        }
    else:
        raise BenchmarkError(f"Unknown benchmark implementation: {configuration['implementation']!r}.")

    return {**measurement, **expected}, plan


def benchmark_summary(raw_rows):
    """Aggregiert ausschließlich gemessene Wiederholungen mit Median und IQR."""

    frame = pd.DataFrame(raw_rows)
    measured = frame.loc[frame["warmup_or_measured"].eq("measured")].copy()
    grouping_columns = [
        "run_id",
        "experiment",
        "configuration_id",
        "task_id",
        "implementation",
        "measurement_scope",
        "dataset_size",
        "grouping",
        "observed_group_count",
        "metric_bundle",
    ]
    rows = []
    for key, current in measured.groupby(grouping_columns, sort=False, dropna=False):
        durations = current["duration_ms"].astype(float)
        first_quartile = float(durations.quantile(0.25))
        third_quartile = float(durations.quantile(0.75))
        row = dict(zip(grouping_columns, key, strict=True))
        row.update({
            "measured_runs": len(durations),
            "minimum_ms": float(durations.min()),
            "first_quartile_ms": first_quartile,
            "median_ms": float(durations.median()),
            "third_quartile_ms": third_quartile,
            "interquartile_range_ms": third_quartile - first_quartile,
            "maximum_ms": float(durations.max()),
            "mean_ms": float(durations.mean()),
            "standard_deviation_ms": float(durations.std(ddof=1)) if len(durations) > 1 else None,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(*, pilot):
    """Führt technischen Pilot oder unveränderte Baseline nach BENCHMARK.md aus."""

    mode = "pilot" if pilot else "main"
    warmups = BENCHMARK_PILOT_WARMUPS if pilot else BENCHMARK_WARMUPS
    measured_blocks = BENCHMARK_PILOT_BLOCKS if pilot else BENCHMARK_MEASURED_BLOCKS
    paths = benchmark_output_paths(pilot=pilot)
    partial_path = prepare_benchmark_output_paths(paths)
    run_id = datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ_{mode}")
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = load_scaling_metadata()
    validation = load_scaling_validation(metadata)
    specs = benchmark_metric_specs(metadata, pilot=pilot)
    configurations = benchmark_configurations(specs)
    schedule = benchmark_execution_schedule(
        configurations,
        warmups=warmups,
        measured_blocks=measured_blocks,
        seed=DEFAULT_SEED,
    )

    with psycopg.connect(database_url(), autocommit=True) as connection:
        database_state = verify_benchmark_database(connection, metadata)
        frames, frame_details = load_benchmark_compute_frames(connection, specs)
        relations, relation_details = prepare_benchmark_relations(connection, specs)
        queries, expected_details = validate_benchmark_specs(connection, specs, relations, frames)

        raw_rows = []
        saved_plans = set()
        with open(partial_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BENCHMARK_RAW_FIELDS)
            writer.writeheader()
            total = len(schedule)
            for position, scheduled in enumerate(schedule, start=1):
                configuration = scheduled["configuration"]
                measurement, plan = execute_benchmark_configuration(
                    connection,
                    configuration,
                    frames=frames,
                    queries=queries,
                    expected_details=expected_details,
                )
                row = {
                    "run_id": run_id,
                    "experiment": configuration["experiment"],
                    "configuration_id": configuration["configuration_id"],
                    "task_id": configuration["task_id"],
                    "implementation": configuration["implementation"],
                    "measurement_scope": configuration["measurement_scope"],
                    "dataset_size": configuration["dataset_size"],
                    "grouping": configuration["grouping"],
                    "observed_group_count": measurement["observed_group_count"],
                    "metric_bundle": configuration["metric_bundle"],
                    "warmup_or_measured": scheduled["warmup_or_measured"],
                    "round": scheduled["round"],
                    "execution_order": scheduled["execution_order"],
                    "duration_ms": measurement["duration_ms"],
                    "client_wall_duration_ms": measurement["client_wall_duration_ms"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": DEFAULT_SEED,
                    "result_group_rows": measurement["result_group_rows"],
                    "result_pair_rows": measurement["result_pair_rows"],
                    "result_multigroup_rows": measurement["result_multigroup_rows"],
                    "sql_planning_time_ms": measurement["sql_planning_time_ms"],
                }
                writer.writerow(row)
                handle.flush()
                raw_rows.append(row)

                if (
                    plan is not None
                    and scheduled["warmup_or_measured"] == "measured"
                    and configuration["configuration_id"] not in saved_plans
                ):
                    plan_path = paths["plans"] / f"{configuration['configuration_id']}.json"
                    key = (
                        configuration["task_id"],
                        configuration["dataset_size"],
                        configuration["grouping"],
                        configuration["metric_bundle"],
                    )
                    write_atomic(plan_path, {
                        "run_id": run_id,
                        "captured_at_utc": row["timestamp_utc"],
                        "configuration": configuration,
                        "query": queries[key],
                        "explain": plan,
                    })
                    saved_plans.add(configuration["configuration_id"])

                if position == 1 or position % max(1, total // 10) == 0 or position == total:
                    print(f"Benchmark {mode}: {position}/{total} Ausführungen", flush=True)

        partial_path.replace(paths["raw"])
        summary = benchmark_summary(raw_rows)
        write_atomic(paths["summary"], summary)
        environment = {
            "schema_version": "1.0",
            "run_id": run_id,
            "mode": mode,
            "status": "completed",
            "created_at_utc": created_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "seed": DEFAULT_SEED,
                "warmup_runs_per_configuration": warmups,
                "measured_blocks": measured_blocks,
                "cache_state": "warm cache; no artificial cache clearing",
                "configuration_count": len(configurations),
                "unique_metric_spec_count": len(specs),
                "parallel_benchmark_processes": 1,
                "pilot_is_not_a_main_result": pilot,
            },
            "host": host_environment(),
            "docker": docker_environment(),
            "software": {
                "python": platform.python_version(),
                "pandas": package_version("pandas"),
                "psycopg": package_version("psycopg"),
                "folktables": package_version("folktables"),
                "pytest": package_version("pytest"),
            },
            "database": database_benchmark_environment(connection),
            "database_state": database_state,
            "data": {
                "metadata_path": str(DEFAULT_SCALING_METADATA_PATH.relative_to(PROJECT_ROOT)),
                "metadata_sha256": file_hash(DEFAULT_SCALING_METADATA_PATH),
                "csv_path": str(DEFAULT_SCALING_CSV_PATH.relative_to(PROJECT_ROOT)),
                "csv_sha256": metadata["csv_sha256"],
                "prepared_rows": metadata["prepared_rows"],
                "region_count": metadata["region_count"],
                "task_sizes": metadata["benchmark_sizes"]["full_sizes"],
                "n_common": metadata["benchmark_sizes"]["n_common"],
            },
            "correctness_gate": {
                "validation_path": str(DEFAULT_SCALING_VALIDATION_PATH.relative_to(PROJECT_ROOT)),
                "validation_sha256": file_hash(DEFAULT_SCALING_VALIDATION_PATH),
                "status": validation["status"],
                "rows_compared": validation["rows_compared"],
                "float_tolerance": validation["float_tolerance"],
            },
            "code_sha256": {
                "python": file_hash(Path(__file__)),
                "sql": file_hash(PROJECT_ROOT / "fairness_datenbank.sql"),
                "benchmark_protocol": file_hash(PROJECT_ROOT / "BENCHMARK.md"),
            },
            "outputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()},
            "setup": {
                "pandas_compute_frames": frame_details,
                "sql_materialized_subsets": relation_details,
                "subset_selection_in_compute_only_timing": False,
                "metric_specs_validated_between_sql_and_pandas": len(specs),
            },
            "results": {
                "raw_rows": len(raw_rows),
                "measured_rows": int(sum(row["warmup_or_measured"] == "measured" for row in raw_rows)),
                "query_plans": len(saved_plans),
                "raw_sha256": file_hash(paths["raw"]),
                "summary_sha256": file_hash(paths["summary"]),
            },
        }
        write_atomic(paths["environment"], environment)

    print(f"Benchmark-{mode} abgeschlossen: {paths['raw']}")
    return 0

def run_validation():
    """Startet den SQL-Pandas-Abgleich und gibt das Prüfergebnis als JSON aus."""

    with psycopg.connect(database_url()) as connection:
        print(json.dumps(validate(connection), indent=2, sort_keys=True))
    return 0


def run_report():
    """Speichert die geprüften Analyseergebnisse samt Daten- und Softwareherkunft."""

    with psycopg.connect(database_url()) as connection:
        result = collect_results(connection)
        database = database_info(connection)

    result["report_schema_version"] = REPORT_SCHEMA_VERSION
    result["provenance"] = report_provenance(result, DEFAULT_REPORT_METADATA_PATH, database=database)

    write_atomic(DEFAULT_ANALYSIS_REPORT_PATH, result)
    print(f"Bericht gespeichert: {DEFAULT_ANALYSIS_REPORT_PATH}")
    return 0

def collect_python_results(prepared):
    """Berechnet alle Metriken in Pandas."""

    tasks = {}
    for task_id in TASK_IDS:
        current = prepared.loc[prepared["task_id"].eq(task_id)].copy()
        tasks[task_id] = {"group_statistics": [], "pairwise_metrics": [], "multigroup_metrics": []}
        for grouping in GROUPINGS:
            bundle = benchmark_pandas_bundle(current, grouping_name=grouping, metric_bundle="Stufe 3")
            tasks[task_id]["group_statistics"].extend(
                {"task_id": task_id, **row} for row in bundle["group_statistics"]
            )
            tasks[task_id]["pairwise_metrics"].extend(bundle["pairwise_metrics"])
            tasks[task_id]["multigroup_metrics"].extend(bundle["multigroup_metrics"])
    return tasks

def run_python():
    """Bereitet die kalifornischen Daten auf und speichert die Pandas-Ergebnisse."""

    print("1/2 Daten und CSV aufbereiten", flush=True)
    prepare_data()
    metadata = json.loads(DEFAULT_REPORT_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["csv_sha256"] != file_hash(DEFAULT_PREPARED_CSV_PATH):
        raise ValueError("CSV und Aufbereitungsmetadaten stimmen nicht überein.")
    prepared = pd.read_csv(DEFAULT_PREPARED_CSV_PATH)
    print("2/2 Pandas-Metriken berechnen", flush=True)
    report = {
        "schema_version": "1.1", "implementation": "python_pandas",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": metadata,
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
        "system": {"platform": platform.platform(), "machine": platform.machine()},
        "code_sha256": {"python": file_hash(Path(__file__))},
        "comparison_status": "SQL-Pandas-Abgleich separat auszuführen.",
        "tasks": collect_python_results(prepared),
    }
    write_atomic(DEFAULT_RESULT_PATH, report)
    print(f"Ergebnisse gespeichert: {DEFAULT_RESULT_PATH}")
    return 0


# Kommandozeile

def parse_args():
    """Liest den Unterbefehl und die Auswahl zwischen Benchmarkpilot und Hauptmessung ein."""

    parser = argparse.ArgumentParser(description="SQL und Pandas für Fairnessmetriken.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = (
        ("run", " Python ausführen und Pandas-Ergebnisse in einer JSON-Datei speichern.", run_python),
        ("prepare-scaling", "Vollständigen Regionsumfang für die Skalierbarkeit aufbereiten.", prepare_scaling_data),
        ("validate", "SQL und Pandas vergleichen.", run_validation),
        ("report", "Geprüfte Ergebnisse speichern.", run_report),
    )
    for command, help_text, handler in commands:
        subparsers.add_parser(command, help=help_text).set_defaults(handler=handler)

    benchmark_parser = subparsers.add_parser("benchmark", help="Vorab festgelegten technischen Pilot oder die Baseline ausführen.")
    benchmark_mode = benchmark_parser.add_mutually_exclusive_group(required=True)
    benchmark_mode.add_argument("--pilot", action="store_true", help="Kurzer technischer Trockenlauf; keine Hauptbenchmark-Ergebnisse.")
    benchmark_mode.add_argument("--main", action="store_true", help="Vollständige unveränderte Baseline mit 3 Warm-ups und 20 Messblöcken.")
    benchmark_parser.set_defaults(handler=lambda args: run_benchmark(pilot=args.pilot))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "benchmark":
        raise SystemExit(args.handler(args))
    raise SystemExit(args.handler())
