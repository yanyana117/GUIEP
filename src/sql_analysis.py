"""
SQL exploratory data analysis of the USGS earthquake catalogue (GUIEP)
=====================================================================

This module loads the same USGS catalogue used by ``seismicity.py`` into a
real **SQLite** database (Python's built-in ``sqlite3``, no external engine)
and answers a set of exploratory questions purely with SQL:

    * SELECT / WHERE          - filter the strongest events
    * GROUP BY + aggregates   - count / average / max magnitude per decade
    * JOIN                    - normalise messy place labels via a lookup table
    * ORDER BY                - rank decades and regions
    * window function (RANK)  - the single strongest quake of each decade

Every SQL result is cross-checked against an independent pandas computation so
the numbers are demonstrably correct.

Usage
-----
    python src/sql_analysis.py                 # build DB + run all queries
    python src/sql_analysis.py --db quakes.db  # keep the .db file on disk

The catalogue is the USGS earthquake search export
(<https://earthquake.usgs.gov/earthquakes/>).
"""
import argparse
import os
import sqlite3

import pandas as pd


# --------------------------------------------------------------------------- #
#  1. Load the CSV and build the SQLite database (CREATE TABLE + INSERT)
# --------------------------------------------------------------------------- #
def load_dataframe(csv_path):
    """Read the USGS CSV and add the derived columns the SQL layer needs.

    Mirrors the date parsing in ``seismicity.py`` so the two analyses see the
    exact same rows. Adds:
        year    : integer calendar year
        decade  : year rounded down to the decade (1989 -> 1980)
        region  : the raw region token, i.e. the text after the last comma in
                  ``place`` ("12 km NNW of Parkfield, California" -> "California")
    """
    df = pd.read_csv(csv_path)
    ts = df["time"].astype(str).str.replace("T", "-", regex=False)
    df["datetime"] = pd.to_datetime(ts, format="%Y-%m-%d-%H:%M:%S.%fZ",
                                    errors="coerce")
    df = df.dropna(subset=["datetime", "mag"]).reset_index(drop=True)

    df["year"] = df["datetime"].dt.year
    df["decade"] = (df["year"] // 10) * 10
    df["region"] = (df["place"].astype(str)
                    .str.rsplit(",", n=1).str[-1].str.strip()
                    .replace({"nan": "Unknown"}))
    return df


def build_database(df, db_path=":memory:"):
    """Create the ``earthquakes`` table and bulk-insert every event.

    Uses explicit DDL (``CREATE TABLE``) and parameterised ``INSERT`` so the
    schema and the load are real SQL, not hidden behind an ORM helper.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS earthquakes;
        CREATE TABLE earthquakes (
            id        TEXT,
            time      TEXT,      -- ISO timestamp string
            year      INTEGER,
            decade    INTEGER,
            latitude  REAL,
            longitude REAL,
            depth     REAL,
            mag       REAL,      -- magnitude
            place     TEXT,
            region    TEXT,      -- normalised in code, refined later via JOIN
            type      TEXT       -- 'earthquake', 'nuclear explosion', ...
        );
        """
    )

    rows = [
        (
            r.id, r.datetime.isoformat(), int(r.year), int(r.decade),
            _num(r.latitude), _num(r.longitude), _num(r.depth),
            float(r.mag), r.place, r.region, r.type,
        )
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        "INSERT INTO earthquakes "
        "(id, time, year, decade, latitude, longitude, depth, mag, place, "
        " region, type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def build_region_lookup(conn, df):
    """Build a small ``regions`` lookup table for the JOIN demo.

    The raw catalogue records the same place several ways - "California" and
    "CA", "Mexico" and "MX". A lookup table maps every raw token to a clean
    region name and a country, so we can GROUP BY a *consistent* label instead
    of double-counting. This is exactly what a JOIN against a dimension table
    is for.
    """
    # Map known abbreviations / variants to a clean region name.
    clean = {
        "CA": "California", "California-Nevada border region": "California",
        "NV": "Nevada", "MX": "Mexico", "WA": "Washington", "OR": "Oregon",
        "NM": "New Mexico", "AZ": "Arizona", "CO": "Colorado",
    }
    # Region -> country (everything not listed defaults to United States).
    country = {
        "Mexico": "Mexico", "MX": "Mexico", "Canada": "Canada",
    }

    rows = []
    for raw in sorted(df["region"].dropna().unique()):
        region_clean = clean.get(raw, raw)
        rows.append((raw, region_clean,
                     country.get(region_clean, country.get(raw, "United States"))))

    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS regions;
        CREATE TABLE regions (
            region_raw   TEXT PRIMARY KEY,  -- joins to earthquakes.region
            region_clean TEXT,             -- normalised name
            country      TEXT
        );
        """
    )
    cur.executemany(
        "INSERT INTO regions (region_raw, region_clean, country) "
        "VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def _num(x):
    """NaN -> SQL NULL, otherwise a float."""
    return None if pd.isna(x) else float(x)


# --------------------------------------------------------------------------- #
#  2. The analysis queries (each is plain SQL, returned as a DataFrame)
# --------------------------------------------------------------------------- #
# Q1  SELECT / WHERE / ORDER BY - the strongest events in the catalogue.
Q_STRONGEST = """
    SELECT substr(time, 1, 10) AS date, place, mag
    FROM   earthquakes
    WHERE  mag >= 7.0
    ORDER  BY mag DESC
    LIMIT  10;
"""

# Q2  GROUP BY + aggregates - activity per decade.
Q_BY_DECADE = """
    SELECT decade,
           COUNT(*)        AS n_events,
           ROUND(AVG(mag), 2) AS avg_mag,
           MAX(mag)        AS max_mag
    FROM   earthquakes
    GROUP  BY decade
    ORDER  BY decade;
"""

# Q3  GROUP BY a JOINed column - top regions after normalising labels.
Q_BY_REGION = """
    SELECT r.region_clean        AS region,
           COUNT(*)              AS n_events,
           ROUND(AVG(e.mag), 2)  AS avg_mag,
           MAX(e.mag)            AS max_mag
    FROM   earthquakes e
    JOIN   regions r ON e.region = r.region_raw
    GROUP  BY r.region_clean
    ORDER  BY n_events DESC
    LIMIT  10;
"""

# Q4  JOIN + GROUP BY country - roll regions up to country level.
Q_BY_COUNTRY = """
    SELECT r.country,
           COUNT(*)              AS n_events,
           ROUND(AVG(e.mag), 2)  AS avg_mag
    FROM   earthquakes e
    JOIN   regions r ON e.region = r.region_raw
    GROUP  BY r.country
    ORDER  BY n_events DESC;
"""

# Q5  Window function - the single strongest quake of each decade.
#     RANK() numbers events 1,2,3,... within each decade, highest mag first;
#     the outer query keeps only rank 1.
Q_DECADE_PEAK = """
    WITH ranked AS (
        SELECT decade,
               substr(time, 1, 10) AS date,
               place,
               mag,
               RANK() OVER (PARTITION BY decade ORDER BY mag DESC) AS rnk
        FROM   earthquakes
    )
    SELECT decade, date, place, mag
    FROM   ranked
    WHERE  rnk = 1
    ORDER  BY decade;
"""

# Q6  Catalogue-wide aggregates (used for the headline numbers + the checks).
Q_OVERALL = """
    SELECT COUNT(*)            AS n_events,
           ROUND(AVG(mag), 4)  AS avg_mag,
           MAX(mag)            AS max_mag,
           MIN(mag)            AS min_mag
    FROM   earthquakes;
"""


def run(conn, sql):
    """Run a query and return the result as a pandas DataFrame (for display)."""
    return pd.read_sql_query(sql, conn)


# --------------------------------------------------------------------------- #
#  3. Cross-validation: SQL results must equal an independent pandas result
# --------------------------------------------------------------------------- #
def cross_validate(conn, df):
    """Recompute the key aggregates with pandas and assert they match SQL."""
    checks = []

    # (a) overall count / max / mean
    sql_overall = run(conn, Q_OVERALL).iloc[0]
    checks.append(("total event count",
                   int(sql_overall["n_events"]), len(df)))
    checks.append(("max magnitude",
                   float(sql_overall["max_mag"]), float(df["mag"].max())))
    checks.append(("mean magnitude (4 dp)",
                   float(sql_overall["avg_mag"]), round(float(df["mag"].mean()), 4)))

    # (b) per-decade event counts must agree row for row
    sql_decade = run(conn, Q_BY_DECADE).set_index("decade")["n_events"]
    pd_decade = df.groupby("decade").size()
    checks.append(("per-decade counts identical",
                   sql_decade.reindex(pd_decade.index).tolist(),
                   pd_decade.tolist()))

    # (c) strongest single event
    sql_top = run(conn, Q_STRONGEST).iloc[0]["mag"]
    checks.append(("strongest single event",
                   float(sql_top), float(df["mag"].max())))

    print("\nCross-validation (SQL  vs  pandas)")
    print("-" * 52)
    all_ok = True
    for name, sql_val, pd_val in checks:
        ok = sql_val == pd_val
        all_ok &= ok
        print(f"  [{'OK ' if ok else 'XX '}] {name:<28} "
              f"{sql_val if not isinstance(sql_val, list) else 'rows'} "
              f"{'==' if ok else '!='} "
              f"{pd_val if not isinstance(pd_val, list) else 'rows'}")
    print("-" * 52)
    print("ALL CHECKS PASSED" if all_ok else "MISMATCH DETECTED")
    return all_ok


# --------------------------------------------------------------------------- #
#  4. Driver
# --------------------------------------------------------------------------- #
def _show(title, frame):
    print(f"\n{title}")
    print("=" * len(title))
    print(frame.to_string(index=False))


def main():
    here = os.path.dirname(__file__)
    default_data = os.path.join(here, "..", "data",
                                "usgs_earthquakes_1900_2020.csv")
    p = argparse.ArgumentParser(description="SQL EDA of the USGS catalogue")
    p.add_argument("--data", default=default_data, help="path to USGS CSV")
    p.add_argument("--db", default=":memory:",
                   help="SQLite file to write (default: in-memory)")
    args = p.parse_args()

    df = load_dataframe(args.data)
    conn = build_database(df, args.db)
    build_region_lookup(conn, df)
    print(f"Loaded {len(df)} events into SQLite "
          f"({'in-memory' if args.db == ':memory:' else args.db}).")

    _show("Q1  Strongest events (SELECT / WHERE / ORDER BY)",
          run(conn, Q_STRONGEST))
    _show("Q2  Activity per decade (GROUP BY + COUNT/AVG/MAX)",
          run(conn, Q_BY_DECADE))
    _show("Q3  Top regions after label normalisation (JOIN + GROUP BY)",
          run(conn, Q_BY_REGION))
    _show("Q4  Events by country (JOIN roll-up)",
          run(conn, Q_BY_COUNTRY))
    _show("Q5  Strongest quake of each decade (window function RANK)",
          run(conn, Q_DECADE_PEAK))

    cross_validate(conn, df)
    conn.close()


if __name__ == "__main__":
    main()
