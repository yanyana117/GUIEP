"""
Earthquake seismicity analysis (GUIEP)
======================================

Quantitative seismicity metrics for a US earthquake catalogue (1900-2020,
sourced from the USGS). Implements the **seismicity index S** of Gu Jicheng,
a single number summarising the activity level of a region/time window from the
event count, the magnitude distribution, and the largest event:

    S = 1.17 * log10(N + 1)
      + 0.29 * log10( (1/N) * sum_i 10^(1.5 * M_i) )
      + 0.15 * M_max

and a spatial-temporal index ST4 based on the great-circle separation of events.

Usage
-----
    python seismicity.py --data ../data/usgs_earthquakes_1900_2020.csv
    python seismicity.py --start 1989-01-01 --end 1989-12-31
    python seismicity.py --plot ../results/magnitude_time.png

Reference: Gu Jicheng, "On the quantification of seismicity (seismicity index
S)"; catalogue from https://earthquake.usgs.gov/earthquakes/
"""
import argparse
import math
import os

import numpy as np
import pandas as pd


def load_catalog(path):
    """Load a USGS earthquake CSV and return it indexed by event datetime.

    Robust to both the standard USGS time format (``...T...Z``) and the
    hyphenated variant used in this dataset (``YYYY-MM-DD-HH:MM:SS.fffZ``).
    """
    df = pd.read_csv(path)
    ts = df["time"].astype(str).str.replace("T", "-", regex=False)
    df["datetime"] = pd.to_datetime(ts, format="%Y-%m-%d-%H:%M:%S.%fZ",
                                    errors="coerce")
    df = df.dropna(subset=["datetime", "mag"]).set_index("datetime").sort_index()
    return df


def seismicity_index(df):
    """Gu's seismicity index S for the events in ``df`` (range 0-10)."""
    N = len(df)
    if N == 0:
        raise ValueError("No events in the selected window.")
    mags = df["mag"].to_numpy(dtype=float)
    m_max = mags.max()
    energy_term = np.sum(10 ** (1.5 * mags))
    S = (1.17 * math.log10(N + 1)
         + 0.29 * math.log10((1.0 / N) * energy_term)
         + 0.15 * m_max)
    return S, N, m_max


def spatial_index_st4(df):
    """Spatial-temporal index ST4 from the first/last events' great-circle
    separation. Faithful to the original GUIEP research formulation."""
    N = len(df)
    if N < 2:
        return float("nan")
    phi = df["latitude"].to_numpy(dtype=float)
    lam = df["longitude"].to_numpy(dtype=float)
    phi_i, phi_j = phi[0], phi[-1]
    lam_i, lam_j = lam[0], lam[-1]

    A = math.cos(np.deg2rad(phi_i))
    B = math.cos(np.deg2rad(phi_j))
    C = math.cos(np.deg2rad(lam_i - lam_j))
    D = math.sin(np.deg2rad(phi_i))
    E = math.sin(np.deg2rad(phi_j))
    theta_ij = 1 / math.sqrt(2) * ((1 - A * B * C - D * E)) ** 0.5

    R0 = 6370.0  # km
    d_ij = 2 * R0 * np.arcsin(np.deg2rad(theta_ij))
    d = 1 / (N * (N - 1)) * d_ij
    k = 0.01  # /km
    return 0.375 * 10 ** (-k * d)


def plot_magnitude_time(df, out_path):
    """Magnitude-vs-time scatter for the catalogue (a stated project goal)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(df.index, df["mag"], s=8, c=df["mag"], cmap="inferno", alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"US earthquakes {df.index.min():%Y}-{df.index.max():%Y} "
                 f"(N = {len(df)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    return out_path


def main():
    here = os.path.dirname(__file__)
    default_data = os.path.join(here, "..", "data",
                                "usgs_earthquakes_1900_2020.csv")
    p = argparse.ArgumentParser(description="Seismicity index analysis")
    p.add_argument("--data", default=default_data, help="path to USGS CSV")
    p.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    p.add_argument("--plot", nargs="?", const=os.path.join(
        here, "..", "results", "magnitude_time.png"), default=None,
        help="save a magnitude-time plot (optional path)")
    args = p.parse_args()

    df = load_catalog(args.data)
    window = df.loc[args.start:args.end]

    S, N, m_max = seismicity_index(window)
    st4 = spatial_index_st4(window)

    lo = window.index.min()
    hi = window.index.max()
    print(f"Window         : {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}")
    print(f"Event count N  : {N}")
    print(f"M_max          : {m_max:.2f}")
    print(f"Seismicity S   : {S:.4f}")
    print(f"Spatial ST4    : {st4:.4f}")

    if args.plot:
        out = plot_magnitude_time(df, args.plot)
        print(f"Saved plot     : {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
