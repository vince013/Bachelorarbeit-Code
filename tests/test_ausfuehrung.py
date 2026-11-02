import json
import pandas as pd
import pytest
import fairness_auswertung as prototype

def test_result_file(tmp_path):
    """Prüft den unveränderten JSON-Export von Fehlwerten, Dezimalzahlen und Unicode-Text."""

    output = tmp_path / "results.json"
    result = {"undefined_ratio": None, "rate": 0.123456789012345, "group": "Geschlecht × Race"}
    prototype.write_atomic(output, result)
    assert json.loads(output.read_text()) == result


def test_atomic_csv_concatenation(tmp_path):
    """Prüft das Zusammenfügen mehrerer CSV-Dateien in der vorgegebenen Reihenfolge."""

    parts = []
    for index, values in enumerate(([1, 2], [3])):
        path = tmp_path / f"part_{index}.csv"
        prototype.write_atomic(path, pd.DataFrame({"value": values}))
        parts.append(path)

    output = tmp_path / "combined.csv"
    prototype.write_atomic(output, parts)
    assert pd.read_csv(output)["value"].tolist() == [1, 2, 3]


def test_atomic_write_preserves_existing_file_on_error(tmp_path):
    """Prüft, dass ein fehlgeschlagener Schreibvorgang die vorhandene Datei erhält."""

    output = tmp_path / "result.json"
    prototype.write_atomic(output, "existing\n")
    with pytest.raises(ValueError):
        prototype.write_atomic(output, {"invalid_rate": float("nan")})
    assert output.read_text(encoding="utf-8") == "existing\n"

def test_python_export():
    """Prüft gültiges JSON mit fehlenden Gruppencodes und undefinierten Paritätsverhältnissen."""

    frame = pd.DataFrame([
        {"task_id": task, "sex": sex, "race": 1, "label": False}
        for task in prototype.TASK_IDS for sex in (1, 2, None)
    ])
    result = prototype.collect_python_results(frame)
    encoded = json.dumps(result, allow_nan=False)
    assert "NaN" not in encoded
    for task in result.values():
        assert any(row["group_code_1"] is None for row in task["group_statistics"])
        assert all(row["disparate_impact_ratio"] is None for row in task["pairwise_metrics"])
        assert all(row["demographic_parity_ratio"] is None for row in task["multigroup_metrics"])


def test_global_task_ranking():
    """Prüft reproduzierbare Datensatzränge und die Ablehnung einer erneuten Rangvergabe."""

    frame = pd.DataFrame({
        "task_id": ["acs_income"] * 5,
        "source_order": range(5),
    })
    first = prototype.rank_task_frame(frame, seed=42)
    second = prototype.rank_task_frame(frame, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["sample_rank"].tolist() == [1, 2, 3, 4, 5]
    assert sorted(first["source_order"]) == list(range(5))
    with pytest.raises(prototype.PreparationError, match="sample_rank"):
        prototype.rank_task_frame(first, seed=42)


def test_benchmark_size_plan():
    """Prüft feste Größenstufen, Vollbestände und die gemeinsame Vergleichsstufe der Aufgaben."""

    plan = prototype.benchmark_size_plan({
        "acs_income": 1_500_000,
        "acs_employment": 2_400_000,
        "acs_public_coverage": 800_000,
    })

    assert plan["n_common"] == 500_000
    assert plan["stages"]["acs_income"][-1] == 1_500_000
    assert plan["stages"]["acs_employment"][-1] == 2_400_000
    assert plan["stages"]["acs_public_coverage"] == [10_000, 50_000, 100_000, 250_000, 500_000, 800_000]
    with pytest.raises(prototype.PreparationError, match="exactly"):
        prototype.benchmark_size_plan({"acs_income": 10_000})


def test_benchmark_configurations_and_schedule():
    """Prüft Konfigurationsumfang und reproduzierbare, vollständige Messblöcke."""

    metadata = {
        "benchmark_sizes": prototype.benchmark_size_plan({
            "acs_income": 1_664_500,
            "acs_employment": 3_236_107,
            "acs_public_coverage": 1_138_289,
        })
    }

    main_specs = prototype.benchmark_metric_specs(metadata, pilot=False)
    main_configurations = prototype.benchmark_configurations(main_specs)
    pilot_specs = prototype.benchmark_metric_specs(metadata, pilot=True)
    pilot_configurations = prototype.benchmark_configurations(pilot_specs)

    assert len(main_specs) == 34
    assert len(main_configurations) == 136
    assert len(pilot_specs) == 7
    assert len(pilot_configurations) == 28
    shared = next(
        spec for spec in main_specs
        if spec["task_id"] == "acs_income"
        and spec["dataset_size"] == 1_000_000
        and spec["grouping"] == "sex_race"
        and spec["metric_bundle"] == "Stufe 3"
    )
    assert shared["experiments"] == ["Z", "G", "M"]

    first = prototype.benchmark_execution_schedule(pilot_configurations, warmups=1, measured_blocks=2, seed=42)
    second = prototype.benchmark_execution_schedule(pilot_configurations, warmups=1, measured_blocks=2, seed=42)
    assert [item["configuration"]["configuration_id"] for item in first] == [
        item["configuration"]["configuration_id"] for item in second
    ]
    assert len(first) == 84
    expected_ids = {configuration["configuration_id"] for configuration in pilot_configurations}
    for round_number in (1, 2):
        measured_ids = {
            item["configuration"]["configuration_id"]
            for item in first
            if item["warmup_or_measured"] == "measured" and item["round"] == round_number
        }
        assert measured_ids == expected_ids


def test_benchmark_pandas_bundles():
    """Prüft die kumulativen Metrikumfänge und die Ablehnung ungültiger Gruppierungen."""

    frame = pd.DataFrame({
        "sex": pd.Series([1, 1, 2, 2], dtype="Int16"),
        "race": pd.Series([1, 2, 1, 2], dtype="Int16"),
        "label": [True, False, True, False],
    })

    stufe_1 = prototype.benchmark_pandas_bundle(frame, grouping_name="sex_race", metric_bundle="Stufe 1")
    stufe_2 = prototype.benchmark_pandas_bundle(frame, grouping_name="sex_race", metric_bundle="Stufe 2")
    stufe_3 = prototype.benchmark_pandas_bundle(frame, grouping_name="sex_race", metric_bundle="Stufe 3")

    assert len(stufe_1["group_statistics"]) == 4
    assert stufe_1["pairwise_metrics"] == []
    assert stufe_1["multigroup_metrics"] == []
    assert len(stufe_2["pairwise_metrics"]) == 6
    assert stufe_2["multigroup_metrics"] == []
    assert len(stufe_3["pairwise_metrics"]) == 6
    assert len(stufe_3["multigroup_metrics"]) == 1
    prototype.assert_nested_close(stufe_2["group_statistics"], stufe_1["group_statistics"])
    prototype.assert_nested_close(stufe_3["pairwise_metrics"], stufe_2["pairwise_metrics"])

    with pytest.raises(prototype.BenchmarkError, match="not enabled"):
        prototype.benchmark_pandas_bundle(frame, grouping_name="unknown", metric_bundle="Stufe 3")
