"""Study design, participant-level summaries and model recovery for the notebook."""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.special import logit
from scipy.stats import spearmanr

from .models import BOUNDS, PARAMS, fit, simulate


COHORTS = ("Common instructions", "Spaceship", "Magic carpet")
FAMILIES = ("MF", "MB", "Hybrid", "Unlucky", "TDLR")
OBS = ["s1", "a1", "s2", "a2", "reward"]
# Cell order is rewarded/common, rewarded/rare, unrewarded/common, unrewarded/rare.
CELLS = ((1, 1), (1, 0), (0, 1), (0, 0))
CELL_NAMES = tuple(f"stay_r{r}_c{c}" for r, c in CELLS)
PAPER_WEIGHTS = {"MB": 0.94, "Unlucky": 0.36, "TDLR": 0.80}
FIG2_PARAMETERS = {
    "MF": dict(alpha1=.5, alpha2=.5, lam=.7, beta1=5., beta2=5., p=0.),
    "Hybrid": dict(alpha1=.5, alpha2=.5, lam=.7, beta1=5., beta2=5., w=.5, p=0.),
    "MB": dict(alpha2=.5, beta1=5., beta2=5., p=0.),
    "Unlucky": dict(alpha2=.5, beta1=5., beta2=5., eta=.5, p=0.),
    "TDLR": dict(alpha_common=.8, alpha_rare=.2, beta1=5., beta2=5., p=0.),
}
# Independent Uniform draws over these intervals define the additional recovery study.
GENERATING_RANGES = {
    "alpha1": (.15, .85), "alpha2": (.15, .85), "lam": (.15, .85),
    "beta1": (2., 8.), "beta2": (2., 8.), "w": (.1, .9), "p": (-.25, .25),
    "eta": (.25, .80), "alpha_common": (.55, .90), "alpha_rare": (.10, .45),
}


def load_trials(path=Path("data/trials.csv.gz")):
    df = pd.read_csv(path, dtype={"participant": str})
    required = {"cohort", "participant", "trial", "valid", "common", *OBS}
    if not required.issubset(df.columns) or set(df.cohort) != set(COHORTS):
        raise ValueError("Unexpected published trial schema or cohort names")
    if df.duplicated(["cohort", "participant", "trial"]).any():
        raise ValueError("Duplicate trial number within a participant")
    ok = df[df.valid.eq(1)]
    if (not ok[["a1", "s2", "a2", "reward", "common"]].isin([0, 1]).all().all()
            or not ok.s1.isin(range(4)).all() or not ok.common.eq(ok.a1.eq(ok.s2)).all()):
        raise ValueError("Invalid decision, state, reward or transition encoding")
    return df.sort_values(["cohort", "participant", "trial"]).reset_index(drop=True)


def participant_audit(data):
    people = data.groupby("cohort").participant.nunique().rename("participants")
    trials = data.groupby("cohort").valid.agg(["size", "sum"]).rename(
        columns={"size": "main_trials", "sum": "completed"})
    return pd.concat([people, trials.assign(omissions=trials.main_trials-trials.completed)],
                     axis=1).loc[list(COHORTS)]


def stay_counts(trials, keep=None):
    """Counts for two successive first choices, labelled by the EARLIER trial."""
    a, s, r = trials[:, 1], trials[:, 2], trials[:, 4]
    cell = 2 * (1 - r[:-1]) + (a[:-1] != s[:-1]).astype(int)
    stay = a[1:] == a[:-1]
    if keep is not None:
        cell, stay = cell[keep], stay[keep]
    totals = np.bincount(cell, minlength=4)
    repeats = np.bincount(cell, weights=stay, minlength=4)
    return repeats, totals


def human_signatures(data, adjacent_only=False):
    """Per-person empirical cells and saturated logistic contrasts on ±1 predictors."""
    rows = []
    for (cohort, subject), group in data[data.valid.eq(1)].groupby(["cohort", "participant"]):
        x = group[OBS].to_numpy(dtype=np.int64)
        keep = np.diff(group.trial.to_numpy()) == 1 if adjacent_only else None
        repeat, total = stay_counts(x, keep)
        prob = np.divide(repeat, total, out=np.full(4, np.nan), where=total > 0)
        # A fixed 1/2 count per outcome keeps log odds finite for separation.
        logits = logit((repeat + .5) / (total + 1))
        if (total == 0).any():
            logits[:] = np.nan
        rows.append(dict(cohort=cohort, participant=subject, pairs=int(total.sum()),
                         **dict(zip(CELL_NAMES, prob)),
                         reward_coef=(logits[0]+logits[1]-logits[2]-logits[3])/4,
                         interaction_coef=(logits[0]-logits[1]-logits[2]+logits[3])/4))
    return pd.DataFrame(rows)


def bootstrap_means(values, seed, draws=3000):
    """Participant (or independent simulated agent) is the resampling unit."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    indices = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    means = np.nanmean(values[indices], axis=1)
    return np.nanmean(values, axis=0), *np.quantile(means, [.025, .975], axis=0)


def contrast_table(primary, strict, seed):
    rows = []
    for i, cohort in enumerate(COHORTS):
        for j, variable in enumerate(["reward_coef", "interaction_coef"]):
            a = primary.loc[primary.cohort.eq(cohort), variable]
            b = strict.loc[strict.cohort.eq(cohort), variable]
            m, lo, hi = bootstrap_means(a, seed + 2*i+j)
            rows.append(dict(cohort=cohort, effect=variable, estimate=m[0], lower=lo[0],
                             upper=hi[0], adjacent_only=b.mean(), people=a.notna().sum()))
    return pd.DataFrame(rows)


def design_seed(seed, *index):
    return int(np.random.SeedSequence([seed, *index]).generate_state(1)[0])


def _map(function, tasks, jobs):
    # Process workers keep SciPy's outer optimizers independent. Restrict each BLAS pool.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=jobs)(delayed(function)(*args) for args in tasks)


def fixed_agents(n_agents, n_refit, n_trials, starts, seed, jobs):
    """Fixed-parameter Figure 2 generators plus illustrative MF and hybrid agents."""
    if not 1 <= n_refit <= n_agents:
        raise ValueError("The hybrid refit subset must be between 1 and n_agents")
    tasks = [(family, i, n_refit, n_trials, starts, design_seed(seed, j, i))
             for j, family in enumerate(FAMILIES) for i in range(n_agents)]
    return pd.DataFrame(_map(_fixed_agent, tasks, jobs))


def _fixed_agent(family, agent, n_refit, n_trials, starts, seed):
    x = simulate(family, FIG2_PARAMETERS[family], n_trials, seed)
    num, den = stay_counts(x)
    row = dict(generator=family, agent=agent, trials=n_trials, **dict(zip(CELL_NAMES, num/den)))
    if family in PAPER_WEIGHTS and agent < n_refit:
        fitted = fit("Hybrid", x, seed=design_seed(seed, 1), starts=starts, paper_prior=True)
        row.update(w=fitted["w"], nll=fitted["nll"], success=fitted["success"])
    return row


def fixed_weight_table(agents):
    fitted = agents.dropna(subset=["w"]).groupby("generator").w
    out = fitted.agg(n="count", reproduced_median="median").loc[list(PAPER_WEIGHTS)]
    out["replicate_2.5%"] = fitted.quantile(.025)
    out["replicate_97.5%"] = fitted.quantile(.975)
    out["published_median"] = pd.Series(PAPER_WEIGHTS)
    out["difference"] = out.reproduced_median - out.published_median
    return out


def fit_families(trials, seed, starts):
    """Fit all families on the same choices. Nest simpler solutions into flexible fits."""
    rows = {}
    for idx, name in enumerate(("MB", "MF", "Hybrid", "Unlucky", "TDLR")):
        initial = []
        if name in {"Hybrid", "Unlucky", "TDLR"}:
            mb = rows["MB"]
            base = {key: mb[key] for key in ("beta1", "beta2", "p")}
            if name == "Hybrid":
                initial = [dict(base, alpha2=mb["alpha2"], alpha1=.5, lam=.5, w=1.),
                           {**{key: rows["MF"][key] for key in PARAMS["MF"]}, "w": 0.}]
            elif name == "Unlucky":
                initial = [dict(base, alpha2=mb["alpha2"], eta=1.)]
            else:
                initial = [dict(base, alpha_common=mb["alpha2"], alpha_rare=mb["alpha2"])]
        row = fit(name, trials, seed=design_seed(seed, idx), starts=starts, initial_points=initial)
        k = len(PARAMS[name])
        rows[name] = dict(row, model=name, k=k, n_trials=len(trials),
                          bic=2*row["nll"]+k*np.log(2*len(trials)), aic=2*row["nll"]+2*k)
    if (rows["Hybrid"]["nll"] > min(rows["MF"]["nll"], rows["MB"]["nll"])+1e-4
            or any(rows[m]["nll"] > rows["MB"]["nll"]+1e-4 for m in ("TDLR", "Unlucky"))):
        raise RuntimeError("An optimizer missed an explicitly nested solution")
    return list(rows.values())


def _fit_person(cohort, subject, trials, seed, starts):
    return [dict(r, cohort=cohort, participant=subject) for r in fit_families(trials, seed, starts)]


def fit_participants(data, seed, starts, jobs):
    groups = data[data.valid.eq(1)].groupby(["cohort", "participant"], sort=True)
    tasks = [(cohort, person, group[OBS].to_numpy(dtype=np.int64),
              design_seed(seed, i), starts) for i, ((cohort, person), group) in enumerate(groups)]
    return pd.DataFrame([r for rows in _map(_fit_person, tasks, jobs) for r in rows])


def _recover_one(family, agent, n_trials, seed, starts):
    rng = np.random.default_rng(seed)
    truth = {param: rng.uniform(*GENERATING_RANGES[param]) for param in PARAMS[family]}
    x = simulate(family, truth, n_trials, design_seed(seed, 1))
    truth = {f"true_{param}": value for param, value in truth.items()}
    return [dict(row, generator=family, agent=agent, **truth)
            for row in fit_families(x, design_seed(seed, 2), starts)]


def recovery_experiment(n_agents, n_trials, seed, starts, jobs):
    tasks = [(fam, i, n_trials, design_seed(seed, j, i), starts)
             for j, fam in enumerate(FAMILIES) for i in range(n_agents)]
    return pd.DataFrame([r for rows in _map(_recover_one, tasks, jobs) for r in rows])


def parameter_recovery(fits):
    out = []
    for name in FAMILIES:
        group = fits.loc[fits.generator.eq(name) & fits.model.eq(name)]
        for param in PARAMS[name]:
            truth, estimate = group["true_" + param], group[param]
            low, high = BOUNDS[param]
            out.append(dict(generator=name, parameter=param, n=len(group),
                            spearman_r=spearmanr(truth, estimate).statistic,
                            bias=(estimate-truth).mean(), rmse=np.sqrt(((estimate-truth)**2).mean()),
                            at_bound=np.mean(np.isclose(estimate, low, atol=1e-5) |
                                             np.isclose(estimate, high, atol=1e-5))))
    return pd.DataFrame(out)


def model_recovery(fits):
    # Sort in a declared order to resolve exact BIC ties deterministically.
    best = fits.sort_values(["bic", "k", "model"]).drop_duplicates(["generator", "agent"])
    return pd.crosstab(best.generator, best.model).reindex(index=FAMILIES, columns=FAMILIES, fill_value=0)


def model_comparison(fits):
    cols = ["cohort", "participant"]
    best = fits.sort_values(["bic", "k", "model"]).drop_duplicates(cols)
    out = fits.groupby(["cohort", "model"])[["bic", "aic"]].sum()
    out["individual_wins"] = best.groupby(["cohort", "model"]).size().reindex(out.index, fill_value=0)
    out["delta_bic"] = out.bic - out.groupby(level=0).bic.transform("min")
    out["delta_aic"] = out.aic - out.groupby(level=0).aic.transform("min")
    return out.reset_index().sort_values(["cohort", "bic"]).reset_index(drop=True)
