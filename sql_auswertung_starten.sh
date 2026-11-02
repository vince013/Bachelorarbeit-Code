#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SQL_FILE="${PROJECT_DIR}/fairness_datenbank.sql"

CSV_PATH="/data/processed/acs_tasks_ca_2018_1year.csv"
METADATA_PATH="/data/processed/acs_tasks_ca_2018_1year.metadata.json"

RESULT_DIR="${PROJECT_DIR}/results"
RESULT_FILE="${RESULT_DIR}/sql_ergebnisse.json"

if [[ ! -f "$SQL_FILE" ]]; then
    echo "Fehler: SQL-Datei nicht gefunden: $SQL_FILE" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"

TEMP_FILE="$(mktemp "${RESULT_DIR}/sql_ergebnisse.XXXXXX.json")"

cleanup() {
    rm -f "$TEMP_FILE"
}
trap cleanup EXIT

cd "$PROJECT_DIR"

echo "Starte PostgreSQL ..."
docker compose up -d --wait postgres

echo "Führe Datenbankskript und Analyse aus ..."

{
    printf '%s\n' 'BEGIN;'
    cat "$SQL_FILE"
    cat <<SQL
SELECT fairness.run_analysis(
    '${CSV_PATH}',
    '${METADATA_PATH}'
)::text;

COMMIT;
SQL
} | docker compose exec -T postgres sh -c '
    exec psql -X -qAt -v ON_ERROR_STOP=1 \
        -U "$POSTGRES_USER" -d "$POSTGRES_DB"
' > "$TEMP_FILE"

if [[ ! -s "$TEMP_FILE" ]]; then
    echo "Fehler: fairness.run_analysis hat kein Ergebnis erzeugt." >&2
    exit 1
fi

python3 -m json.tool "$TEMP_FILE" >/dev/null

mv "$TEMP_FILE" "$RESULT_FILE"
trap - EXIT

echo "Analyse erfolgreich abgeschlossen."
echo "Ergebnis gespeichert unter: $RESULT_FILE"
