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

    # sort by beta
    idx = np.argsort(beta)
    beta = beta[idx]
    nlpd = nlpd[idx]

    # identify best beta
    best_idx = np.argmin(nlpd)
    best_beta = beta[best_idx]

    # plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(beta, nlpd, "-o", label="Validation NLPD")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Validation NLPD")

    ax.axvline(best_beta, ls="--", color="k", alpha=0.5)
    ax.scatter(best_beta, nlpd[best_idx], zorder=3)
    ax.legend(frameon=False)

    plt.tight_layout()

    if args.save is not None:
        plt.savefig(args.save, dpi=300)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
