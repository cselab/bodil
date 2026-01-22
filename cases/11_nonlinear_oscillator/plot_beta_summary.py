#!/usr/bin/env python3
"""
Plot beta selection results from beta_selection_summary.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot beta selection results")
    parser.add_argument(
        "csv",
        type=str,
        default="beta_selection_summary.csv",
        help="CSV file produced during beta selection",
    )
    parser.add_argument(
        "--show-coverage",
        action="store_true",
        help="Overlay 90%% coverage on a secondary axis",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output filename (e.g. beta_selection.pdf)",
    )
    args = parser.parse_args()

    # load data
    df = pd.read_csv(args.csv)

    beta = df["beta"].values
    nlpd = df["val_nlpd"].values
    cov90 = df["val_cov90"].values

    # sort by beta
    idx = np.argsort(beta)
    beta = beta[idx]
    nlpd = nlpd[idx]
    cov90 = cov90[idx]

    # identify best beta
    best_idx = np.argmin(nlpd)
    best_beta = beta[best_idx]

    # plot
    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))

    ax1.plot(beta, nlpd, "-o", label="Validation NLPD")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel("Validation NLPD")
    ax1.set_title("Beta selection by validation NLPD")

    ax1.axvline(best_beta, ls="--", color="k", alpha=0.5)
    ax1.scatter(best_beta, nlpd[best_idx], zorder=3)
    ax1.legend(frameon=False)

    if args.show_coverage:
        ax2 = ax1.twinx()
        ax2.plot(beta, cov90, "--s", alpha=0.6, label="90% coverage")
        ax2.set_ylabel("90% coverage")
        ax2.axhline(0.9, color="k", lw=0.5, alpha=0.3)

    plt.tight_layout()

    if args.save is not None:
        plt.savefig(args.save, dpi=300)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
