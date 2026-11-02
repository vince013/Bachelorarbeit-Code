CREATE SCHEMA IF NOT EXISTS fairness;

CREATE TABLE IF NOT EXISTS fairness.training_data (
    task_id TEXT NOT NULL,
    sample_rank BIGINT NOT NULL,
    sex SMALLINT,
    race SMALLINT,
    label BOOLEAN NOT NULL,
    person_weight DOUBLE PRECISION,
    age SMALLINT,

    CONSTRAINT training_data_task_id
        CHECK (task_id IN ('acs_income', 'acs_employment', 'acs_public_coverage')),
    CONSTRAINT training_data_task_rank_unique
        UNIQUE (task_id, sample_rank),
    CONSTRAINT training_data_sample_rank_positive
        CHECK (sample_rank > 0),
    CONSTRAINT training_data_sex_code
        CHECK (sex IS NULL OR sex BETWEEN 1 AND 2),
    CONSTRAINT training_data_race_code
        CHECK (race IS NULL OR race BETWEEN 1 AND 9),
    CONSTRAINT training_data_age_code CHECK (age BETWEEN 0 AND 99),
    CONSTRAINT training_data_person_weight_positive
        CHECK (person_weight IS NULL OR person_weight >= 1)
);

COMMENT ON TABLE fairness.training_data IS 'Trainingslabels aus ACS PUMS ohne Modellvorhersagen.';
COMMENT ON COLUMN fairness.training_data.task_id IS 'Feste Kennung der Analyseaufgabe.';
COMMENT ON COLUMN fairness.training_data.sample_rank IS 'Feste Rangfolge je Aufgabe nach einer dokumentierten Mischung.';
COMMENT ON COLUMN fairness.training_data.sex IS 'ACS PUMS SEX: 1 männlich, 2 weiblich. Fehlende Werte bleiben sichtbar.';
COMMENT ON COLUMN fairness.training_data.race IS 'ACS PUMS RAC1P von 1 bis 9. Fehlende Werte bleiben sichtbar.';
COMMENT ON COLUMN fairness.training_data.age IS 'ACS 2018 AGEP von 0 bis 99. Fehlende Werte bleiben sichtbar.';
COMMENT ON COLUMN fairness.training_data.label IS 'Vorhandenes binäres Label der jeweiligen Aufgabe.';
COMMENT ON COLUMN fairness.training_data.person_weight IS 'ACS PUMS Personengewicht. Im Kern ungewichtet.';


-- Prüft die Aufgabe und die lückenlose Reihenfolge der Datensätze.
CREATE OR REPLACE FUNCTION fairness.assert_training_data(p_task_id TEXT)
RETURNS VOID
LANGUAGE plpgsql
STABLE
AS $function$
DECLARE
    v_count BIGINT;
    v_min_rank BIGINT;
    v_max_rank BIGINT;
BEGIN
    IF p_task_id IS NULL OR p_task_id NOT IN ('acs_income', 'acs_employment', 'acs_public_coverage') THEN
        RAISE EXCEPTION 'Unknown task: %', p_task_id;
    END IF;

    SELECT COUNT(*), MIN(sample_rank), MAX(sample_rank)
    INTO v_count, v_min_rank, v_max_rank
    FROM fairness.training_data
    WHERE task_id = p_task_id;

    IF v_count = 0 THEN
        RAISE EXCEPTION 'fairness.training_data is empty for task %.', p_task_id;
    END IF;

    IF v_min_rank <> 1 OR v_max_rank <> v_count THEN
        RAISE EXCEPTION 'sample_rank must form the complete range 1..row_count for task %.', p_task_id;
    END IF;

    IF EXISTS (
        SELECT 1 FROM fairness.training_data WHERE task_id = p_task_id
          AND ((p_task_id = 'acs_income' AND (age IS NULL OR age <= 16))
            OR (p_task_id = 'acs_public_coverage' AND (age IS NULL OR age >= 65)))
    ) THEN
        RAISE EXCEPTION 'Age violates task filter for %; reprepare and import data with the original AGEP column.', p_task_id;
    END IF;

END
$function$;


CREATE OR REPLACE FUNCTION fairness.import_training_data(p_csv_path TEXT, p_expected_rows BIGINT DEFAULT NULL)
RETURNS TABLE (
    imported_rows BIGINT,
    imported_tasks BIGINT
)
LANGUAGE plpgsql
VOLATILE
AS $function$
DECLARE
    v_staged_rows BIGINT;
    v_task_id TEXT;
BEGIN
    IF p_csv_path IS NULL OR BTRIM(p_csv_path) = '' THEN
        RAISE EXCEPTION 'CSV path must not be empty.';
    END IF;

    DROP TABLE IF EXISTS training_data_stage;
    CREATE TEMP TABLE training_data_stage (
        task_id TEXT,
        sample_rank BIGINT,
        sex SMALLINT,
        race SMALLINT,
        label BOOLEAN,
        person_weight DOUBLE PRECISION,
        age SMALLINT
    ) ON COMMIT DROP;

    EXECUTE FORMAT(
        'COPY training_data_stage '
        '(task_id, sample_rank, sex, race, label, person_weight, age) '
        'FROM %L WITH (FORMAT CSV, HEADER TRUE)',
        p_csv_path
    );

    SELECT COUNT(*) INTO v_staged_rows
    FROM training_data_stage;

    IF v_staged_rows = 0 THEN
        RAISE EXCEPTION 'The task import file contains no data rows.';
    END IF;

    IF p_expected_rows IS NOT NULL AND v_staged_rows <> p_expected_rows THEN
        RAISE EXCEPTION 'Expected % rows, but staged %.', p_expected_rows, v_staged_rows;
    END IF;

    TRUNCATE TABLE fairness.training_data RESTART IDENTITY;

    INSERT INTO fairness.training_data (task_id, sample_rank, sex, race, label, person_weight, age)
    SELECT
        task_id,
        sample_rank,
        sex,
        race,
        label,
        person_weight,
        age
    FROM training_data_stage;

    FOR v_task_id IN
        SELECT DISTINCT task_id
        FROM training_data_stage
        ORDER BY task_id
    LOOP
        PERFORM fairness.assert_training_data(v_task_id);
    END LOOP;

    EXECUTE 'ANALYZE fairness.training_data';

    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT,
        COUNT(DISTINCT task_id)::BIGINT
    FROM fairness.training_data;
END
$function$;


-- Gemeinsame Gruppenstatistik für alle Gruppierungen.

CREATE OR REPLACE VIEW fairness.group_statistics AS
WITH grouped AS (
    SELECT
        task_id,
        'sex'::TEXT AS grouping_name,
        sex AS group_code_1,
        NULL::SMALLINT AS group_code_2,
        COUNT(*)::BIGINT AS group_size,
        COUNT(*) FILTER (WHERE label)::BIGINT AS positive_labels,
        COUNT(*) FILTER (WHERE NOT label)::BIGINT AS negative_labels
    FROM fairness.training_data
    GROUP BY task_id, sex

    UNION ALL

    SELECT
        task_id,
        'race'::TEXT,
        race,
        NULL::SMALLINT,
        COUNT(*)::BIGINT,
        COUNT(*) FILTER (WHERE label)::BIGINT,
        COUNT(*) FILTER (WHERE NOT label)::BIGINT
    FROM fairness.training_data
    GROUP BY task_id, race

    UNION ALL

    SELECT
        task_id,
        'sex_race'::TEXT,
        sex,
        race,
        COUNT(*)::BIGINT,
        COUNT(*) FILTER (WHERE label)::BIGINT,
        COUNT(*) FILTER (WHERE NOT label)::BIGINT
    FROM fairness.training_data
    GROUP BY task_id, sex, race
)
SELECT
    task_id,
    grouping_name,
    group_code_1,
    group_code_2,
    group_size,
    positive_labels,
    negative_labels,
    group_size::DOUBLE PRECISION
        / NULLIF(SUM(group_size) OVER (PARTITION BY task_id, grouping_name), 0)
        AS representation_rate,
    positive_labels::DOUBLE PRECISION
        / NULLIF(group_size, 0)::DOUBLE PRECISION
        AS positive_label_rate
FROM grouped;

CREATE OR REPLACE FUNCTION fairness.pairwise_metrics(
    p_task_id TEXT,
    p_grouping_name TEXT,
    p_reference_code_1 SMALLINT,
    p_reference_code_2 SMALLINT,
    p_comparison_code_1 SMALLINT,
    p_comparison_code_2 SMALLINT
)
RETURNS TABLE (
    grouping_name TEXT,
    reference_code_1 SMALLINT,
    reference_code_2 SMALLINT,
    comparison_code_1 SMALLINT,
    comparison_code_2 SMALLINT,
    reference_group_size BIGINT,
    comparison_group_size BIGINT,
    reference_positive_label_rate DOUBLE PRECISION,
    comparison_positive_label_rate DOUBLE PRECISION,
    statistical_parity_difference DOUBLE PRECISION,
    disparate_impact_ratio DOUBLE PRECISION
)
LANGUAGE plpgsql
STABLE
AS $function$
BEGIN
    IF p_task_id IS NULL OR p_task_id NOT IN ('acs_income', 'acs_employment', 'acs_public_coverage') THEN
        RAISE EXCEPTION 'Unknown task: %', p_task_id;
    END IF;

    IF p_grouping_name IS NULL
       OR p_grouping_name NOT IN ('sex', 'race', 'sex_race') THEN
        RAISE EXCEPTION 'Unknown grouping: %', p_grouping_name;
    END IF;

    IF p_reference_code_1 IS NULL OR p_comparison_code_1 IS NULL THEN
        RAISE EXCEPTION 'Primary reference and comparison codes must not be NULL.';
    END IF;

    IF p_grouping_name = 'sex_race' AND (
        p_reference_code_2 IS NULL OR p_comparison_code_2 IS NULL
    ) THEN
        RAISE EXCEPTION '% comparisons require two codes per group.', p_grouping_name;
    END IF;

    IF p_grouping_name <> 'sex_race' AND (
        p_reference_code_2 IS NOT NULL OR p_comparison_code_2 IS NOT NULL
    ) THEN
        RAISE EXCEPTION '% comparisons require exactly one code per group.', p_grouping_name;
    END IF;

    IF p_reference_code_1 = p_comparison_code_1
       AND p_reference_code_2 IS NOT DISTINCT FROM p_comparison_code_2 THEN
        RAISE EXCEPTION 'Reference and comparison group must be different.';
    END IF;

    RETURN QUERY
    WITH matched_pair AS (
        SELECT
            metrics.*,
            ROW(metrics.reference_code_1, metrics.reference_code_2)
                IS NOT DISTINCT FROM ROW(p_reference_code_1, p_reference_code_2) AS same_direction
        FROM fairness.all_pairwise_metrics(p_task_id, p_grouping_name) AS metrics
        WHERE (
            ROW(metrics.reference_code_1, metrics.reference_code_2)
                IS NOT DISTINCT FROM ROW(p_reference_code_1, p_reference_code_2)
            AND ROW(metrics.comparison_code_1, metrics.comparison_code_2)
                IS NOT DISTINCT FROM ROW(p_comparison_code_1, p_comparison_code_2)
        ) OR (
            ROW(metrics.reference_code_1, metrics.reference_code_2)
                IS NOT DISTINCT FROM ROW(p_comparison_code_1, p_comparison_code_2)
            AND ROW(metrics.comparison_code_1, metrics.comparison_code_2)
                IS NOT DISTINCT FROM ROW(p_reference_code_1, p_reference_code_2)
        )
    )
    SELECT
        p_grouping_name,
        p_reference_code_1,
        p_reference_code_2,
        p_comparison_code_1,
        p_comparison_code_2,
        CASE WHEN pair.same_direction THEN pair.reference_group_size ELSE pair.comparison_group_size END,
        CASE WHEN pair.same_direction THEN pair.comparison_group_size ELSE pair.reference_group_size END,
        CASE WHEN pair.same_direction THEN pair.reference_positive_label_rate ELSE pair.comparison_positive_label_rate END,
        CASE WHEN pair.same_direction THEN pair.comparison_positive_label_rate ELSE pair.reference_positive_label_rate END,
        CASE WHEN pair.same_direction THEN pair.statistical_parity_difference ELSE -pair.statistical_parity_difference END,
        CASE
            WHEN pair.same_direction THEN pair.disparate_impact_ratio
            ELSE pair.reference_positive_label_rate / NULLIF(pair.comparison_positive_label_rate, 0)
        END
    FROM matched_pair AS pair;
END
$function$;


-- Gibt jedes vollständige Gruppenpaar einmal zurück.
CREATE OR REPLACE FUNCTION fairness.all_pairwise_metrics(p_task_id TEXT, p_grouping_name TEXT)
RETURNS TABLE (
    grouping_name TEXT,
    reference_code_1 SMALLINT,
    reference_code_2 SMALLINT,
    comparison_code_1 SMALLINT,
    comparison_code_2 SMALLINT,
    reference_group_size BIGINT,
    comparison_group_size BIGINT,
    reference_positive_label_rate DOUBLE PRECISION,
    comparison_positive_label_rate DOUBLE PRECISION,
    statistical_parity_difference DOUBLE PRECISION,
    disparate_impact_ratio DOUBLE PRECISION
)
LANGUAGE plpgsql
STABLE
AS $function$
BEGIN
    IF p_task_id IS NULL OR p_task_id NOT IN ('acs_income', 'acs_employment', 'acs_public_coverage') THEN
        RAISE EXCEPTION 'Unknown task: %', p_task_id;
    END IF;

    IF p_grouping_name IS NULL
       OR p_grouping_name NOT IN ('sex', 'race', 'sex_race') THEN
        RAISE EXCEPTION 'Unknown grouping: %', p_grouping_name;
    END IF;

    RETURN QUERY
    SELECT
        p_grouping_name,
        reference.group_code_1,
        reference.group_code_2,
        comparison.group_code_1,
        comparison.group_code_2,
        reference.group_size,
        comparison.group_size,
        reference.positive_label_rate,
        comparison.positive_label_rate,
        comparison.positive_label_rate - reference.positive_label_rate,
        comparison.positive_label_rate
            / NULLIF(reference.positive_label_rate, 0)
    FROM fairness.group_statistics AS reference
    JOIN fairness.group_statistics AS comparison
      ON comparison.task_id = reference.task_id
     AND comparison.grouping_name = reference.grouping_name
     AND (
            reference.group_code_1 < comparison.group_code_1
         OR (
                reference.group_code_1 = comparison.group_code_1
            AND reference.group_code_2 < comparison.group_code_2
         )
     )
    WHERE reference.task_id = p_task_id
      AND reference.grouping_name = p_grouping_name
      AND reference.group_code_1 IS NOT NULL
      AND comparison.group_code_1 IS NOT NULL
      AND (
            p_grouping_name <> 'sex_race'
         OR (
                reference.group_code_2 IS NOT NULL
            AND comparison.group_code_2 IS NOT NULL
         )
      )
    ORDER BY
        reference.group_code_1,
        reference.group_code_2 NULLS FIRST,
        comparison.group_code_1,
        comparison.group_code_2 NULLS FIRST;
END
$function$;


-- Berechnet Mehrgruppenextreme aus vollständigen Gruppencodes.
-- Vergleicht Raten vor der Ausgabe exakt.
CREATE OR REPLACE FUNCTION fairness.multigroup_metrics(p_task_id TEXT, p_grouping_name TEXT)
RETURNS TABLE (
    grouping_name TEXT,
    included_group_count BIGINT,
    minimum_positive_label_rate DOUBLE PRECISION,
    maximum_positive_label_rate DOUBLE PRECISION,
    minimum_groups JSONB,
    maximum_groups JSONB,
    demographic_parity_difference DOUBLE PRECISION,
    demographic_parity_ratio DOUBLE PRECISION
)
LANGUAGE plpgsql
STABLE
AS $function$
BEGIN
    IF p_task_id IS NULL OR p_task_id NOT IN ('acs_income', 'acs_employment', 'acs_public_coverage') THEN
        RAISE EXCEPTION 'Unknown task: %', p_task_id;
    END IF;

    IF p_grouping_name IS NULL
       OR p_grouping_name NOT IN ('sex', 'race', 'sex_race') THEN
        RAISE EXCEPTION 'Unknown grouping: %', p_grouping_name;
    END IF;

    RETURN QUERY
    WITH included AS (
        SELECT
            stats.group_code_1,
            stats.group_code_2,
            stats.positive_labels::NUMERIC
                / stats.group_size::NUMERIC AS exact_rate,
            CASE
                WHEN p_grouping_name = 'sex_race' THEN JSONB_BUILD_ARRAY(stats.group_code_1, stats.group_code_2)
                ELSE JSONB_BUILD_ARRAY(stats.group_code_1)
            END AS group_key
        FROM fairness.group_statistics AS stats
        WHERE stats.task_id = p_task_id
          AND stats.grouping_name = p_grouping_name
          AND stats.group_code_1 IS NOT NULL
          AND (
                p_grouping_name <> 'sex_race'
             OR stats.group_code_2 IS NOT NULL
          )
    ),
    bounds AS (
        SELECT
            COUNT(*)::BIGINT AS group_count,
            MIN(exact_rate) AS minimum_rate,
            MAX(exact_rate) AS maximum_rate
        FROM included
    ),
    summarised AS (
        SELECT
            bounds.group_count,
            bounds.minimum_rate,
            bounds.maximum_rate,
            COALESCE(
                JSONB_AGG(included.group_key ORDER BY included.group_code_1, included.group_code_2 NULLS FIRST) FILTER (WHERE included.exact_rate = bounds.minimum_rate),
                '[]'::JSONB
            ) AS minimum_group_keys,
            COALESCE(
                JSONB_AGG(included.group_key ORDER BY included.group_code_1, included.group_code_2 NULLS FIRST) FILTER (WHERE included.exact_rate = bounds.maximum_rate),
                '[]'::JSONB
            ) AS maximum_group_keys
        FROM bounds
        LEFT JOIN included ON TRUE
        GROUP BY
            bounds.group_count,
            bounds.minimum_rate,
            bounds.maximum_rate
    )
    SELECT
        p_grouping_name,
        summarised.group_count,
        summarised.minimum_rate::DOUBLE PRECISION,
        summarised.maximum_rate::DOUBLE PRECISION,
        summarised.minimum_group_keys,
        summarised.maximum_group_keys,
        CASE
            WHEN summarised.group_count < 2 THEN NULL
            ELSE (summarised.maximum_rate - summarised.minimum_rate)::DOUBLE PRECISION
        END,
        CASE
            WHEN summarised.group_count < 2 THEN NULL
            ELSE (summarised.minimum_rate / NULLIF(summarised.maximum_rate, 0))::DOUBLE PRECISION
        END
    FROM summarised;
END
$function$;


-- Erzeugt die vollständig materialisierende SQL Abfrage für den Benchmark.
-- Die Quellrelation enthält bereits genau die außerhalb der Messung gebildete
-- Teilmenge und muss sex, race und label bereitstellen.
CREATE OR REPLACE FUNCTION fairness.benchmark_query(
    p_source_relation REGCLASS,
    p_grouping_name TEXT,
    p_metric_bundle TEXT
)
RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $function$
DECLARE
    v_group_code_1 TEXT;
    v_group_code_2 TEXT;
    v_group_by TEXT;
    v_pair_result TEXT := '';
    v_multigroup_result TEXT := '';
BEGIN
    IF p_source_relation IS NULL THEN
        RAISE EXCEPTION 'Benchmark source relation must not be NULL.';
    END IF;
    IF p_grouping_name IS NULL OR p_grouping_name NOT IN ('sex', 'race', 'sex_race') THEN
        RAISE EXCEPTION 'Unknown benchmark grouping: %', p_grouping_name;
    END IF;
    IF p_metric_bundle IS NULL OR p_metric_bundle NOT IN ('Stufe 1', 'Stufe 2', 'Stufe 3') THEN
        RAISE EXCEPTION 'Unknown benchmark metric bundle: %', p_metric_bundle;
    END IF;
    CASE p_grouping_name
        WHEN 'sex' THEN
            v_group_code_1 := 'source.sex';
            v_group_code_2 := 'NULL::SMALLINT';
            v_group_by := 'source.sex';
        WHEN 'race' THEN
            v_group_code_1 := 'source.race';
            v_group_code_2 := 'NULL::SMALLINT';
            v_group_by := 'source.race';
        WHEN 'sex_race' THEN
            v_group_code_1 := 'source.sex';
            v_group_code_2 := 'source.race';
            v_group_by := 'source.sex, source.race';
    END CASE;

    IF p_metric_bundle IN ('Stufe 2', 'Stufe 3') THEN
        v_pair_result := $query$
UNION ALL
SELECT
    2,
    pairs.reference_code_1,
    pairs.reference_code_2,
    pairs.comparison_code_1,
    pairs.comparison_code_2,
    'pairwise_metrics',
    JSONB_BUILD_OBJECT(
        'grouping_name', pairs.grouping_name,
        'reference_code_1', pairs.reference_code_1,
        'reference_code_2', pairs.reference_code_2,
        'comparison_code_1', pairs.comparison_code_1,
        'comparison_code_2', pairs.comparison_code_2,
        'reference_group_size', pairs.reference_group_size,
        'comparison_group_size', pairs.comparison_group_size,
        'reference_positive_label_rate', pairs.reference_positive_label_rate,
        'comparison_positive_label_rate', pairs.comparison_positive_label_rate,
        'statistical_parity_difference', pairs.statistical_parity_difference,
        'disparate_impact_ratio', pairs.disparate_impact_ratio
    )
FROM pairs
$query$;
    END IF;

    IF p_metric_bundle = 'Stufe 3' THEN
        v_multigroup_result := $query$
UNION ALL
SELECT
    3,
    NULL::SMALLINT,
    NULL::SMALLINT,
    NULL::SMALLINT,
    NULL::SMALLINT,
    'multigroup_metrics',
    JSONB_BUILD_OBJECT(
        'grouping_name', summarised.grouping_name,
        'included_group_count', summarised.group_count,
        'minimum_positive_label_rate', summarised.minimum_rate::DOUBLE PRECISION,
        'maximum_positive_label_rate', summarised.maximum_rate::DOUBLE PRECISION,
        'minimum_groups', summarised.minimum_groups,
        'maximum_groups', summarised.maximum_groups,
        'demographic_parity_difference', CASE
            WHEN summarised.group_count < 2 THEN NULL
            ELSE (summarised.maximum_rate - summarised.minimum_rate)::DOUBLE PRECISION
        END,
        'demographic_parity_ratio', CASE
            WHEN summarised.group_count < 2 THEN NULL
            ELSE (summarised.minimum_rate / NULLIF(summarised.maximum_rate, 0))::DOUBLE PRECISION
        END
    )
FROM summarised
$query$;
    END IF;

    RETURN FORMAT(
        $query$
WITH parameters AS (
    SELECT %1$L::TEXT AS grouping_name
),
grouped AS MATERIALIZED (
    SELECT
        %2$s::SMALLINT AS group_code_1,
        %3$s::SMALLINT AS group_code_2,
        COUNT(*)::BIGINT AS group_size,
        COUNT(*) FILTER (WHERE source.label)::BIGINT AS positive_labels,
        COUNT(*) FILTER (WHERE NOT source.label)::BIGINT AS negative_labels
    FROM %4$s AS source
    GROUP BY %5$s
),
statistics AS MATERIALIZED (
    SELECT
        parameters.grouping_name,
        grouped.group_code_1,
        grouped.group_code_2,
        grouped.group_size,
        grouped.positive_labels,
        grouped.negative_labels,
        grouped.group_size::DOUBLE PRECISION
            / NULLIF(SUM(grouped.group_size) OVER (), 0) AS representation_rate,
        grouped.positive_labels::DOUBLE PRECISION
            / NULLIF(grouped.group_size, 0)::DOUBLE PRECISION AS positive_label_rate
    FROM grouped
    CROSS JOIN parameters
),
pairs AS (
    SELECT
        reference.grouping_name,
        reference.group_code_1 AS reference_code_1,
        reference.group_code_2 AS reference_code_2,
        comparison.group_code_1 AS comparison_code_1,
        comparison.group_code_2 AS comparison_code_2,
        reference.group_size AS reference_group_size,
        comparison.group_size AS comparison_group_size,
        reference.positive_label_rate AS reference_positive_label_rate,
        comparison.positive_label_rate AS comparison_positive_label_rate,
        comparison.positive_label_rate - reference.positive_label_rate
            AS statistical_parity_difference,
        comparison.positive_label_rate
            / NULLIF(reference.positive_label_rate, 0) AS disparate_impact_ratio
    FROM statistics AS reference
    JOIN statistics AS comparison
      ON comparison.grouping_name = reference.grouping_name
     AND (
            reference.group_code_1 < comparison.group_code_1
         OR (
                reference.group_code_1 = comparison.group_code_1
            AND reference.group_code_2 < comparison.group_code_2
         )
     )
    WHERE reference.group_code_1 IS NOT NULL
      AND comparison.group_code_1 IS NOT NULL
      AND (
            reference.grouping_name <> 'sex_race'
         OR (
                reference.group_code_2 IS NOT NULL
            AND comparison.group_code_2 IS NOT NULL
         )
      )
),
included AS (
    SELECT
        statistics.grouping_name,
        statistics.group_code_1,
        statistics.group_code_2,
        statistics.positive_labels::NUMERIC
            / statistics.group_size::NUMERIC AS exact_rate,
        CASE
            WHEN statistics.grouping_name = 'sex_race'
                THEN JSONB_BUILD_ARRAY(statistics.group_code_1, statistics.group_code_2)
            ELSE JSONB_BUILD_ARRAY(statistics.group_code_1)
        END AS group_key
    FROM statistics
    WHERE statistics.group_code_1 IS NOT NULL
      AND (
            statistics.grouping_name <> 'sex_race'
         OR statistics.group_code_2 IS NOT NULL
      )
),
bounds AS (
    SELECT
        parameters.grouping_name,
        COUNT(included.group_code_1)::BIGINT AS group_count,
        MIN(included.exact_rate) AS minimum_rate,
        MAX(included.exact_rate) AS maximum_rate
    FROM parameters
    LEFT JOIN included ON included.grouping_name = parameters.grouping_name
    GROUP BY parameters.grouping_name
),
summarised AS (
    SELECT
        bounds.grouping_name,
        bounds.group_count,
        bounds.minimum_rate,
        bounds.maximum_rate,
        COALESCE(
            JSONB_AGG(included.group_key ORDER BY included.group_code_1, included.group_code_2 NULLS FIRST)
                FILTER (WHERE included.exact_rate = bounds.minimum_rate),
            '[]'::JSONB
        ) AS minimum_groups,
        COALESCE(
            JSONB_AGG(included.group_key ORDER BY included.group_code_1, included.group_code_2 NULLS FIRST)
                FILTER (WHERE included.exact_rate = bounds.maximum_rate),
            '[]'::JSONB
        ) AS maximum_groups
    FROM bounds
    LEFT JOIN included ON included.grouping_name = bounds.grouping_name
    GROUP BY bounds.grouping_name, bounds.group_count, bounds.minimum_rate, bounds.maximum_rate
)
SELECT
    1 AS result_order,
    statistics.group_code_1 AS sort_code_1,
    statistics.group_code_2 AS sort_code_2,
    NULL::SMALLINT AS sort_code_3,
    NULL::SMALLINT AS sort_code_4,
    'group_statistics'::TEXT AS result_type,
    JSONB_BUILD_OBJECT(
        'grouping_name', statistics.grouping_name,
        'group_code_1', statistics.group_code_1,
        'group_code_2', statistics.group_code_2,
        'group_size', statistics.group_size,
        'positive_labels', statistics.positive_labels,
        'negative_labels', statistics.negative_labels,
        'representation_rate', statistics.representation_rate,
        'positive_label_rate', statistics.positive_label_rate
    ) AS payload
FROM statistics
%6$s
%7$s
ORDER BY
    result_order,
    sort_code_1 NULLS LAST,
    sort_code_2 NULLS LAST,
    sort_code_3 NULLS LAST,
    sort_code_4 NULLS LAST
        $query$,
        p_grouping_name,
        v_group_code_1,
        v_group_code_2,
        p_source_relation,
        v_group_by,
        v_pair_result,
        v_multigroup_result
    );
END
$function$;

COMMENT ON FUNCTION fairness.benchmark_query(REGCLASS, TEXT, TEXT) IS
'Erzeugt die SQL Baseline für die vorab festgelegten Berechnungsstufen 1 bis 3.';


-- Prüft Metadaten und liefert SQL Ergebnisse.
CREATE OR REPLACE FUNCTION fairness.run_analysis(p_csv_path TEXT, p_metadata_path TEXT)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $function$
DECLARE
    v_metadata JSONB;
    v_import RECORD;
    v_task_id TEXT;
    v_count BIGINT;
    v_positive BIGINT;
    v_tasks JSONB;
BEGIN
    v_metadata := pg_read_file(p_metadata_path)::JSONB;
    IF (v_metadata->>'csv_sha256') IS DISTINCT FROM encode(sha256(pg_read_binary_file(p_csv_path)), 'hex') THEN
        RAISE EXCEPTION 'CSV checksum does not match preparation metadata.';
    END IF;
    IF NOT (v_metadata->'tasks' ?& ARRAY['acs_income', 'acs_employment', 'acs_public_coverage'])
       OR (SELECT COUNT(*) FROM jsonb_object_keys(v_metadata->'tasks')) <> 3
       OR v_metadata->>'prepared_rows' IS NULL THEN
        RAISE EXCEPTION 'Preparation metadata must describe all three tasks.';
    END IF;

    SELECT * INTO v_import
    FROM fairness.import_training_data(p_csv_path, (v_metadata->>'prepared_rows')::BIGINT);
    IF v_import.imported_tasks <> 3 THEN
        RAISE EXCEPTION 'All three tasks are required.';
    END IF;
    FOREACH v_task_id IN ARRAY ARRAY['acs_income', 'acs_employment', 'acs_public_coverage'] LOOP
        SELECT COUNT(*), COUNT(*) FILTER (WHERE label)
        INTO v_count, v_positive
        FROM fairness.training_data WHERE task_id = v_task_id;
        IF v_count IS DISTINCT FROM (v_metadata->'tasks'->v_task_id->>'prepared_rows')::BIGINT
           OR v_positive IS DISTINCT FROM (v_metadata->'tasks'->v_task_id->>'positive_labels')::BIGINT THEN
            RAISE EXCEPTION 'Task counts do not match preparation metadata: %', v_task_id;
        END IF;
    END LOOP;

    SELECT jsonb_object_agg(t.task_id, jsonb_build_object(
        'group_statistics', (
            SELECT jsonb_agg(to_jsonb(s) ORDER BY s.grouping_name, s.group_code_1, s.group_code_2)
            FROM fairness.group_statistics AS s WHERE s.task_id = t.task_id
        ),
        'pairwise_metrics', (
            SELECT COALESCE(jsonb_agg(to_jsonb(p) ORDER BY p.grouping_name, p.reference_code_1,
                p.reference_code_2, p.comparison_code_1, p.comparison_code_2), '[]'::JSONB)
            FROM (VALUES ('sex'), ('race'), ('sex_race')) AS g(grouping_name)
            CROSS JOIN LATERAL fairness.all_pairwise_metrics(t.task_id, g.grouping_name) AS p
        ),
        'multigroup_metrics', (
            SELECT jsonb_agg(to_jsonb(m) ORDER BY m.grouping_name)
            FROM (VALUES ('sex'), ('race'), ('sex_race')) AS g(grouping_name)
            CROSS JOIN LATERAL fairness.multigroup_metrics(t.task_id, g.grouping_name) AS m
        )
    )) INTO v_tasks
    FROM (VALUES ('acs_income'), ('acs_employment'), ('acs_public_coverage')) AS t(task_id);

    RETURN jsonb_build_object(
        'schema_version', '1.1', 'implementation', 'postgresql',
        'created_at_utc', to_char(clock_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'),
        'database', current_database(), 'software', jsonb_build_object('postgresql', version()),
        'data_provenance', v_metadata,
        'input_checks', 'CSV checksum, task counts and label counts agree with preparation metadata.',
        'comparison_status', 'SQL-Pandas-Abgleich separat auszuführen.',
        'tasks', v_tasks
    );
END
$function$;
