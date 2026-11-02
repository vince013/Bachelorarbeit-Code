import os
from pathlib import Path
import pandas as pd
import pytest

from fairness_auswertung import (
    FLOAT_TOLERANCE, GROUPINGS, PAIRWISE_COLUMNS, TASK_IDS, MetricInputError,
    PreparationError, compare_metrics, compare_pairs,
    compare_groups, compare_multigroup, fetch_data,
    normalise_codes, build_metadata, group_statistics,
    multigroup_metrics, pairwise_metrics,
    collect_results, prepare_tasks,
)

DATA_PATH = Path(__file__).parent / "data" / "manuelle_testfaelle.csv"
REFERENCE_GROUP = 1
COMPARISON_GROUP = 2

@pytest.fixture
def shared_raw():
    """Erzeugt synthetische ACS-Rohdaten für die drei Aufgaben."""

    columns = (
        "AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P PINCP PWGTP "
        "ESR PUBCOV DIS ESP CIT MIG MIL ANC NATIVITY DEAR DEYE DREM ST FER"
    ).split()
    frame = pd.DataFrame({column: [1] * 5 for column in columns})
    for column, values in {
        "AGEP": [30, 70, 15, 50, 40], "WKHP": [40, 30, 10, 0, 20],
        "SEX": [1, 2, 2, 1, 2], "RAC1P": [1, 2, 3, 4, 5],
        "PINCP": [60_000, 20_000, 1_000, 30_000, 40_000],
        "PWGTP": [10, 20, 30, 40, 50], "ESR": [1, 2, 1, 6, 1],
        "PUBCOV": [1, 2, 1, 2, 1], "ST": [6] * 5,
    }.items():
        frame[column] = values
    return frame


@pytest.mark.parametrize("task_id,expected_rows,expected_positive_labels", [
    ("acs_income", 3, 1), ("acs_employment", 5, 3), ("acs_public_coverage", 2, 1),
])
def test_task_preparation(
    shared_raw,
    task_id,
    expected_rows,
    expected_positive_labels,
):
    """Prüft Aufgabenfilter, positive Labels und die deterministische Aufbereitung."""

    first = prepare_tasks(shared_raw, task_ids=(task_id,), seed=42)
    second = prepare_tasks(shared_raw, task_ids=(task_id,), seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["task_id", "sample_rank", "sex", "race", "label", "person_weight", "age"]
    assert first["task_id"].eq(task_id).all()
    assert first["sample_rank"].tolist() == list(range(1, expected_rows + 1))
    assert len(first) == expected_rows
    assert int(first["label"].astype(bool).sum()) == expected_positive_labels


def test_task_preparation_without_global_ranks(shared_raw):
    """Prüft die gemeinsame Aufgabenaufbereitung vor der globalen Rangvergabe."""

    prepared = prepare_tasks(shared_raw, seed=42, assign_sample_rank=False)

    assert list(prepared.columns) == ["task_id", "sex", "race", "label", "person_weight", "age"]
    assert len(prepared) == 10
    assert set(prepared["task_id"]) == set(TASK_IDS)


def test_task_ranks(shared_raw, tmp_path):
    """Prüft aufgabenlokale Ränge und Metadaten bei gemeinsamer Aufbereitung."""

    combined = prepare_tasks(shared_raw, seed=42)

    assert len(combined) == 10
    assert set(combined["task_id"]) == {"acs_income", "acs_employment", "acs_public_coverage"}
    assert not combined.duplicated(["task_id", "sample_rank"]).any()
    for task_id, task_frame in combined.groupby("task_id", sort=False):
        assert task_frame["sample_rank"].tolist() == list(range(1, len(task_frame) + 1))
        single_task = prepare_tasks(shared_raw, task_ids=(task_id,), seed=42)
        pd.testing.assert_frame_equal(task_frame.reset_index(drop=True), single_task)

    metadata = build_metadata(
        state="CA",
        seed=42,
        raw_rows=len(shared_raw),
        raw_missing_age=int(shared_raw["AGEP"].isna().sum()),
        prepared=combined,
        output_path=tmp_path / "tasks.csv",
    )
    assert metadata["raw_rows"] == 5
    assert metadata["prepared_rows"] == 10
    assert {task: details["prepared_rows"] for task, details in metadata["tasks"].items()} == {
        "acs_income": 3, "acs_employment": 5, "acs_public_coverage": 2,
    }


def test_invalid_tasks(shared_raw):
    """Prüft die Ablehnung unbekannter oder mehrfach ausgewählter Aufgaben."""

    with pytest.raises(PreparationError, match="Unknown task"):
        prepare_tasks(shared_raw, task_ids=("unknown",), seed=42)
    with pytest.raises(PreparationError, match="only once"):
        prepare_tasks(shared_raw, task_ids=["acs_income", "acs_income"], seed=42)


@pytest.fixture(scope="module")
def cases():
    """Lädt die gemeinsamen synthetischen Eingaben für Paar- und Mehrgruppentests."""

    return pd.read_csv(DATA_PATH, dtype={"sex": "Int16", "race": "Int16"})


def case(frame, case_id):
    """Wählt die Eingabezeilen eines synthetischen Testfalls aus."""

    return frame.loc[frame["case_id"].eq(case_id)].reset_index(drop=True)


@pytest.mark.parametrize("case_id,reference_size,comparison_size,reference_rate,comparison_rate,spd,dir_ratio", [
    ("equal_rates", 2, 2, 0.5, 0.5, 0.0, 1.0),
    ("different_rates", 4, 4, 0.75, 0.25, -0.5, 1.0 / 3.0),
    ("comparison_no_positive", 2, 2, 0.5, 0.0, -0.5, 0.0),
    ("missing_group_values", 2, 2, 0.5, 0.5, 0.0, 1.0),
    ("missing_reference", 0, 2, None, 0.5, None, None),
    ("zero_reference_rate", 2, 2, 0.0, 0.5, 0.5, None),
])
def test_pairwise_values(
    cases,
    case_id,
    reference_size,
    comparison_size,
    reference_rate,
    comparison_rate,
    spd,
    dir_ratio,
):
    """Vergleicht Gruppengrößen, Labelraten, SPD und DIR mit den festgelegten Sollwerten."""

    result = pairwise_metrics(
        case(cases, case_id), grouping_name="sex", group_columns=["sex"],
        pairs=[(REFERENCE_GROUP, COMPARISON_GROUP)],
    ).iloc[0]
    assert (result["reference_group_size"], result["comparison_group_size"]) == (reference_size, comparison_size)
    for column, expected in (
        ("reference_positive_label_rate", reference_rate), ("comparison_positive_label_rate", comparison_rate),
        ("statistical_parity_difference", spd), ("disparate_impact_ratio", dir_ratio),
    ):
        if expected is None:
            assert pd.isna(result[column])
        else:
            assert result[column] == pytest.approx(expected, abs=FLOAT_TOLERANCE)


def test_missing_group(cases):
    """Prüft fehlende Gruppencodes in der Statistik und ihren Ausschluss aus Paarvergleichen."""

    current = case(cases, "missing_group_values")
    statistics = group_statistics(current, ["sex"])

    missing = statistics.loc[statistics["sex"].isna()].iloc[0]
    assert len(statistics) == 3
    assert missing["group_size"] == 2
    assert missing["positive_labels"] == 1
    assert missing["negative_labels"] == 1
    assert missing["positive_label_rate"] == pytest.approx(0.5)
    pairs = pairwise_metrics(current, grouping_name="sex", group_columns=["sex"])
    assert [(row.reference_code_1, row.comparison_code_1) for row in pairs.itertuples()] == [(1, 2)]


def test_group_statistics(cases):
    """Prüft Gruppengrößen, positive und negative Labelzahlen sowie Repräsentationsanteile."""

    statistics = group_statistics(
        case(cases, "different_rates"),
        ["sex"],
    )

    reference = statistics.loc[statistics["sex"].eq(1)].iloc[0]
    comparison = statistics.loc[statistics["sex"].eq(2)].iloc[0]
    assert reference["group_size"] == 4
    assert reference["positive_labels"] == 3
    assert reference["negative_labels"] == 1
    assert reference["representation_rate"] == pytest.approx(0.5)
    assert comparison["group_size"] == 4
    assert comparison["positive_labels"] == 1
    assert comparison["negative_labels"] == 3
    assert comparison["representation_rate"] == pytest.approx(0.5)
    assert statistics["representation_rate"].sum() == pytest.approx(1.0)


def test_small_intersections(cases):
    """Prüft kleine intersektionale Gruppen, gebundene Extremgruppen und Mehrgruppenwerte."""

    current = case(cases, "small_intersections")
    statistics = group_statistics(
        current,
        ["sex", "race"],
    )
    multigroup = multigroup_metrics(
        current,
        grouping_name="sex_race",
        group_columns=["sex", "race"],
    )

    sizes = {
        (int(row.sex), int(row.race)): int(row.group_size)
        for row in statistics.itertuples()
    }
    assert sizes == {(1, 1): 1, (1, 2): 1, (2, 1): 1, (2, 2): 2}
    assert multigroup.included_group_count == 4
    assert multigroup.minimum_groups == ((1, 2), (2, 1))
    assert multigroup.maximum_groups == ((1, 1),)
    assert multigroup.demographic_parity_difference == pytest.approx(1.0)
    assert multigroup.demographic_parity_ratio == pytest.approx(0.0)


def test_all_pairs(cases):
    """Prüft Vollständigkeit, kanonische Reihenfolge und Auswahl intersektionaler Gruppenpaare."""

    current = case(cases, "small_intersections")
    pairs = pairwise_metrics(
        current,
        grouping_name="sex_race",
        group_columns=["sex", "race"],
    )

    assert len(pairs) == 6
    assert list(pairs.iloc[0][
        [
            "reference_code_1",
            "reference_code_2",
            "comparison_code_1",
            "comparison_code_2",
        ]
    ]) == [1, 1, 1, 2]
    selected = pairwise_metrics(
        current,
        grouping_name="sex_race",
        group_columns=["sex", "race"],
        pairs=[((1, 1), (1, 2))],
    )
    pd.testing.assert_frame_equal(selected, pairs.iloc[[0]].reset_index(drop=True))


def test_missing_pair_codes(cases):
    """Prüft die Ablehnung ausdrücklich angeforderter Paare mit fehlenden Gruppencodes."""

    with pytest.raises(MetricInputError, match="must not be missing"):
        pairwise_metrics(
            case(cases, "missing_group_values"),
            grouping_name="sex",
            group_columns=["sex"],
            pairs=[(pd.NA, 2)],
        )


def test_invalid_labels():
    """Prüft die Ablehnung eines Labels außerhalb der binären Werte 0 und 1."""

    frame = pd.DataFrame({"sex": [1, 2], "label": [0, 2]})
    with pytest.raises(MetricInputError, match="0/1"):
        group_statistics(frame, ["sex"])


@pytest.mark.parametrize("case_id,included_groups,minimum_rate,maximum_rate,minimum_groups,maximum_groups,dpd,dpr", [
    ("multi_different_rates", 3, 0.25, 0.75, ((1,),), ((3,),), 0.5, 1.0 / 3.0),
    ("multi_tied_extremes", 4, 0.0, 1.0, ((1,), (2,)), ((3,), (4,)), 1.0, 0.0),
    ("multi_all_zero", 3, 0.0, 0.0, ((1,), (2,), (3,)), ((1,), (2,), (3,)), 0.0, None),
    ("multi_single_group", 1, 0.5, 0.5, ((1,),), ((1,),), None, None),
    ("multi_missing_codes", 2, 0.5, 1.0, ((1,),), ((2,),), 0.5, 0.5),
])
def test_multigroup_values(
    cases,
    case_id,
    included_groups,
    minimum_rate,
    maximum_rate,
    minimum_groups,
    maximum_groups,
    dpd,
    dpr,
):
    """Vergleicht Extremraten, Extremgruppen, DPD und DPR mit den festgelegten Sollwerten."""

    result = multigroup_metrics(
        case(cases, case_id),
        grouping_name="race",
        group_columns=["race"],
    )

    assert result.included_group_count == included_groups
    assert result.minimum_positive_label_rate == pytest.approx(minimum_rate)
    assert result.maximum_positive_label_rate == pytest.approx(maximum_rate)
    assert result.minimum_groups == minimum_groups
    assert result.maximum_groups == maximum_groups
    if dpd is None:
        assert result.demographic_parity_difference is None
    else:
        assert result.demographic_parity_difference == pytest.approx(dpd)
    if dpr is None:
        assert result.demographic_parity_ratio is None
    else:
        assert result.demographic_parity_ratio == pytest.approx(dpr)


def test_missing_group_multigroup(cases):
    """Prüft den Erhalt fehlender Gruppen in der Statistik und ihren Ausschluss aus DPD und DPR."""

    current = case(cases, "multi_missing_codes")
    statistics = group_statistics(current, ["race"])
    result = multigroup_metrics(
        current,
        grouping_name="race",
        group_columns=["race"],
    )

    missing = statistics.loc[statistics["race"].isna()].iloc[0]
    assert missing["group_size"] == 2
    assert missing["positive_label_rate"] == 0
    assert result.included_group_count == 2
    assert result.minimum_groups == ((1,),)


def test_only_missing_groups():
    """Prüft ausschließlich fehlende Gruppen sowie die Ablehnung ungültiger Gruppendefinitionen."""

    frame = pd.DataFrame(
        {
            "race": pd.Series([pd.NA, pd.NA], dtype="Int16"),
            "label": [True, False],
        }
    )
    result = multigroup_metrics(
        frame,
        grouping_name="race",
        group_columns=["race"],
    )

    assert result.included_group_count == 0
    assert result.minimum_positive_label_rate is None
    assert result.maximum_positive_label_rate is None
    assert result.minimum_groups == ()
    assert result.maximum_groups == ()
    assert result.demographic_parity_difference is None
    assert result.demographic_parity_ratio is None

    with pytest.raises(MetricInputError, match="Unknown grouping"):
        multigroup_metrics(
            frame,
            grouping_name="unknown",
            group_columns=["race"],
        )
    with pytest.raises(MetricInputError, match="requires columns"):
        multigroup_metrics(
            frame,
            grouping_name="race",
            group_columns=["sex"],
        )
