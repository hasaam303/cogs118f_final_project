"""Recover the generating family as a function of planning weight and task length."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analysis import FAMILIES, GENERATING_RANGES, _map, design_seed, fit_families
from .models import PARAMS, simulate
from .study import OUT, SEED, digest


WEIGHTS = (0., .1, .25, .5, .75, .9, 1.)
LENGTHS = (125, 250, 500, 1000)
AGENTS = 100
STARTS = 10
GRID_SEED = SEED + 5


def fit_prefixes(agent, weight):
    """Match nuisance parameters, reward paths and random streams across weights."""
    rng = np.random.default_rng(design_seed(GRID_SEED, agent, 0))
    params = {name: rng.uniform(*GENERATING_RANGES[name])
              for name in PARAMS["Hybrid"] if name != "w"}
    generator = "MF" if weight == 0 else "MB" if weight == 1 else "Hybrid"
    simulation_seed = design_seed(GRID_SEED, agent, 1)
    trials = simulate(generator, dict(params, w=weight), max(LENGTHS), simulation_seed)
    rows = []
    for length in LENGTHS:
        fit_seed = design_seed(GRID_SEED, agent, 2, length)
        fits = fit_families(trials[:length], fit_seed, STARTS)
        if not all(np.isfinite(f["bic"]) and f["success"] for f in fits):
            raise RuntimeError(f"Unresolved fit for agent {agent}, w={weight}, n={length}")
        winner = min(fits, key=lambda f: (f["bic"], f["k"], f["model"]))
        rows.append(dict(agent=agent, weight=weight, generator=generator, trials=length,
                         winner=winner["model"], **{f"bic_{f['model']}": f["bic"] for f in fits}))
    return rows


def run_grid(jobs=6, refresh=False):
    """Load verified decisions or regenerate the complete, seeded experiment."""
    folder = Path(__file__).parent
    sources = [Path(__file__), folder/"analysis.py", folder/"models.py", folder/"study.py"]
    fingerprint = {p.name: digest(p) for p in sources}
    path = OUT/"hybrid_recovery_grid.csv"
    manifest = OUT/"hybrid_recovery_grid_run.json"
    if not refresh and path.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["inputs"] == fingerprint and saved["output"] == digest(path):
            return pd.read_csv(path)
    tasks = [(agent, w) for agent in range(AGENTS) for w in WEIGHTS]
    rows = _map(fit_prefixes, tasks, jobs)
    frame = pd.DataFrame([row for agent_rows in rows for row in agent_rows])
    OUT.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)
    record = dict(inputs=fingerprint, output=digest(path), weights=WEIGHTS, lengths=LENGTHS,
                  agents_per_weight=AGENTS, starts=STARTS, seed=GRID_SEED)
    manifest.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return frame


def selection_rates(frame):
    """Pointwise Wilson intervals across independent agents within each cell."""
    rows = []
    for (weight, length), group in frame.groupby(["weight", "trials"]):
        for family in FAMILIES:
            wins = int(group.winner.eq(family).sum())
            total = len(group)
            ci = binomtest(wins, total).proportion_ci(method="wilson")
            rows.append(dict(weight=weight, trials=length, selected=family, wins=wins,
                             agents=total, rate=wins/total, lower=ci.low, upper=ci.high))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    grid = run_grid(args.jobs, args.refresh)
    print(selection_rates(grid).query("weight == 0.5").to_string(index=False))
