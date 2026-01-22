#!/usr/bin/env python3
"""
Plot predictions and uncertainty bands from CSV files produced by B-ODIL.

Expected inputs:
  - laplace_pred_*.csv
  - laplace_data_*.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot B-ODIL predictions and data")
    parser.add_argument(
        "pred",
        type=str,
        help="Prediction CSV file (laplace_pred_*.csv)",
    )
    parser.add_argument(
        "data",
        type=str,
        help="Data CSV file (laplace_data_*.csv)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output file (e.g. pred_plot.pdf)",
    )
    parser.add_argument(
        "--ylim-x",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-limits for x plot",
    )
    parser.add_argument(
        "--ylim-v",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-limits for v plot",
    )
    args = parser.parse_args()

    # load CSVs
    df_pred = pd.read_csv(args.pred)
    df_data = pd.read_csv(args.data)

    t = df_pred["t"].values

    # create plot
    fig, axes = plt.subplots(ncols=2, figsize=(9.6, 4.8))

    # ---- x(t) ----
    ax = axes[0]
    ax.fill_between(
        t,
        df_pred["x05"],
        df_pred["x95"],
        alpha=0.2,
        lw=0,
        color="r",
        label="5–95% posterior",
    )
    ax.plot(t, df_pred["xmap"], "-r", label="MAP")
    if "xexact" in df_pred.columns:
        ax.plot(t, df_pred["xexact"], "--k", label="exact")
    ax.plot(df_data["t"], df_data["x"], "+k", label="data")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.legend(frameon=False)

    if args.ylim_x is not None:
        ax.set_ylim(*args.ylim_x)

    # ---- v(t) ----
    ax = axes[1]
    ax.fill_between(
        t,
        df_pred["v05"],
        df_pred["v95"],
        alpha=0.2,
        lw=0,
        color="r",
        label="5–95% posterior",
    )
    ax.plot(t, df_pred["vmap"], "-r", label="MAP")
    if "vexact" in df_pred.columns:
        ax.plot(t, df_pred["vexact"], "--k", label="exact")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v$")

    if args.ylim_v is not None:
        ax.set_ylim(*args.ylim_v)

    plt.tight_layout()

    if args.save is not None:
        plt.savefig(args.save, dpi=300)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
