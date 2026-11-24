"""Run the expensive stages from the notebook or with python -m replication.study."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .analysis import (fit_participants, fixed_agents, load_trials, model_comparison,
                       recovery_experiment)
from .predictive import posterior_predictive


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "trials.csv.gz"
OUT = ROOT / "outputs"
SEED = 1182020
DESIGN = {
    "human": dict(seed=SEED+1, starts=10),
    "figure2": dict(seed=SEED+2, n_agents=1000, n_refit=1000, n_trials=1000, starts=10),
    "recovery": dict(seed=SEED+3, n_agents=100, n_trials=250, starts=10),
    "predictive": dict(seed=SEED+4, n_rep=500, n_particles=2048, min_ess=200, max_particles=65536),
}
OUTPUTS = {"human": ["human_fits"], "figure2": ["simulated_agents"],
           "recovery": ["recovery_fits"], "predictive": ["predictive_cells", "posterior_diagnostics"]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage(name, *, settings=None, jobs=6, refresh=False):
    """Reuse results only if code, input data, settings and every output hash agree."""
    settings = dict(DESIGN[name] if settings is None else settings)
    OUT.mkdir(exist_ok=True)
    sources = [Path(__file__), ROOT/"replication/models.py", ROOT/"replication/analysis.py"]
    if name in ("human", "predictive"):
        sources.append(DATA)
    if name == "predictive":
        sources.extend([ROOT/"replication/predictive.py", OUT/"human_fits.csv"])
    fingerprint = {"settings": settings, "inputs": {p.name: digest(p) for p in sources}}
    manifest = OUT / (name + "_run.json")
    paths = [OUT / (key + ".csv") for key in OUTPUTS[name]]
    if not refresh and manifest.exists() and all(p.exists() for p in paths):
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["run"] == fingerprint and all(previous["outputs"][p.name] == digest(p) for p in paths):
            return tuple(pd.read_csv(p, dtype={"participant": str}, float_precision="round_trip") for p in paths)

    if name == "human":
        tables = (fit_participants(load_trials(DATA), jobs=jobs, **settings),)
    elif name == "figure2":
        tables = (fixed_agents(jobs=jobs, **settings),)
    elif name == "recovery":
        tables = (recovery_experiment(jobs=jobs, **settings),)
    else:
        fits = pd.read_csv(OUT/"human_fits.csv", dtype={"participant": str}, float_precision="round_trip")
        winners = model_comparison(fits).drop_duplicates("cohort").set_index("cohort").model.to_dict()
        tables = posterior_predictive(load_trials(DATA), fits, winners, n_jobs=jobs, **settings)
    for p, frame in zip(paths, tables, strict=True):
        tmp = p.with_suffix(".tmp")
        frame.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(p)
    metadata = {"run": fingerprint, "outputs": {p.name: digest(p) for p in paths}}
    manifest.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*DESIGN, "all"])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for stage in (DESIGN if args.stage == "all" else [args.stage]):
        print(f"Running {stage}", flush=True)
        results = run_stage(stage, jobs=args.jobs, refresh=args.refresh)
        print(f"{stage} output rows: {[len(t) for t in results]}", flush=True)
