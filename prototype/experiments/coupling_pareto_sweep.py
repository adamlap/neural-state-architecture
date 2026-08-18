"""Thin wrapper → dynamic_nsa_tradeoff α sweep."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from prototype.experiments.dynamic_nsa_tradeoff import run_tradeoff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-samples", type=int, default=600)
    args = p.parse_args()
    run_tradeoff(epochs=args.epochs, lr=args.lr, n_samples=args.n_samples, do_alpha_sweep=True)

if __name__ == "__main__":
    main()
