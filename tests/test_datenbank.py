import os
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from fairness_auswertung import (
    GROUPINGS,
    assert_nested_close,
    benchmark_pandas_bundle,
    compare_groups,
    compare_multigroup,
    compare_pairs,
    sql_benchmark_result,
)


pytestmark = pytest.mark.integration
SQL_PATH = Path(__file__).parents[1] / "fairness_datenbank.sql"


def test_sql_and_pandas_use_the_same_three_groupings():
    """Prüft SQL und Pandas für die drei Gruppierungen, Paarrichtungen und den Benchmarkumfang."""

    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    frame = pd.DataFrame(
        {
            "sample_rank": [1, 2, 3, 4],
            "sex": pd.Series([1, 1, 2, 2], dtype="Int16"),
            "race": pd.Series([1, 2, 1, 2], dtype="Int16"),
            "label": [True, True, True, False],
        }
    )

    with psycopg.connect(database_url) as connection:
        connection.execute("DROP SCHEMA IF EXISTS fairness CASCADE")
        connection.execute(SQL_PATH.read_text(encoding="utf-8"))
        connection.execute("TRUNCATE fairness.training_data")
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO fairness.training_data
                    (task_id, sample_rank, sex, race, label, person_weight, age)
                VALUES ('acs_employment', %s, %s, %s, %s, 1, 30)
                """,
                frame.itertuples(index=False, name=None),
            )

        for grouping_name, group_columns in GROUPINGS.items():
            arguments = {
                "task_id": "acs_employment",
                "grouping_name": grouping_name,
                "pandas_group_columns": group_columns,
            }
            compare_groups(connection, frame, **arguments)
            compare_pairs(connection, frame, **arguments)
            compare_multigroup(connection, frame, **arguments)

        direct = connection.execute(
            "SELECT * FROM fairness.pairwise_metrics('acs_employment', 'sex', 1::SMALLINT, NULL, 2::SMALLINT, NULL)"
        ).fetchone()
        reverse = connection.execute(
            "SELECT * FROM fairness.pairwise_metrics('acs_employment', 'sex', 2::SMALLINT, NULL, 1::SMALLINT, NULL)"
        ).fetchone()
        assert direct[-2:] == pytest.approx((-0.5, 0.5))
        assert reverse[-2:] == pytest.approx((0.5, 2.0))

        query = connection.execute(
            "SELECT fairness.benchmark_query('fairness.training_data'::REGCLASS, 'sex_race', 'Stufe 3')"
        ).fetchone()[0]
        sql_result = sql_benchmark_result(connection.execute(query).fetchall())
        pandas_result = benchmark_pandas_bundle(frame, grouping_name="sex_race", metric_bundle="Stufe 3")
        assert_nested_close(sql_result, pandas_result)

        connection.rollback()
