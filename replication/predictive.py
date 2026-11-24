"""Participant-level importance sampling and forward posterior predictive checks."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.vq import kmeans2
from scipy.special import expit, log_expit, logit, logsumexp
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

from .analysis import CELLS, OBS, stay_counts
from .models import BOUNDS, PARAMS, nll_batch, reward_walk, simulate


PROB_COLUMNS = ["p00", "p01", "p10", "p11"]


def _log_prior(z):
    # A uniform density on each original interval transforms to a logistic
    # density in its logit coordinate. Interval widths cancel the Jacobian.
    return (log_expit(z) + log_expit(-z)).sum(axis=-1)


class _Coordinates:
    """Logits of bounded parameters, with repeat-choice bias on its log-odds scale."""

    def __init__(self, model):
        names = PARAMS[model]
        bounds = np.array([BOUNDS[name] for name in names])
        self.lower = bounds[:, 0]
        self.width = bounds[:, 1] - bounds[:, 0]
        self.beta = names.index("beta1")
        self.perseveration = names.index("p")

    def _bias_scale(self, unit_beta):
        # First-stage logits contain the product beta1 * p. Scaling the p
        # coordinate straightens this ridge; hypot keeps it regular near zero.
        beta = self.lower[self.beta] + self.width[self.beta] * unit_beta
        return np.hypot(1, beta) / 5

    def encode(self, fractions):
        z = logit(np.maximum(fractions, np.finfo(float).tiny))
        z[..., self.perseveration] *= self._bias_scale(fractions[..., self.beta])
        return z

    def decode(self, z):
        scale = self._bias_scale(expit(z[..., self.beta]))
        native = z.copy()
        native[..., self.perseveration] /= scale
        theta = self.lower + self.width * expit(native)
        # The off-diagonal derivative of the bias scale does not affect this
        # triangular map's determinant.
        return theta, _log_prior(native) - np.log(scale)


def _tempered_pilot(model, trials, coords, rng, n):
    """Explore posterior regions from independent prior draws before proposal fitting."""
    dim = len(coords.lower)
    z = coords.encode(rng.random((n, dim)))
    theta, prior = coords.decode(z)
    ll = -nll_batch(model, theta, trials)
    power = 0.
    steps = accepted = attempted = 0
    jitter = 1e-6 * np.eye(dim)
    while power < 1.:
        # Choose the next likelihood power to retain 70% incremental ESS.
        def weights_at(next_power):
            return _normalized((next_power - power) * ll)

        next_power = 1.
        w = weights_at(next_power)
        if 1 / (w @ w) < .7 * n:
            lo, hi = power, 1.
            for _ in range(24):
                mid = (lo + hi) / 2
                w = weights_at(mid)
                if 1 / (w @ w) < .7 * n:
                    hi = mid
                else:
                    lo = mid
            next_power = hi
            w = weights_at(hi)
        centered = z - w @ z
        chol = np.linalg.cholesky((centered.T * w) @ centered + jitter)
        positions = (rng.random() + np.arange(n)) / n
        ancestors = np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)
        z, ll, prior = z[ancestors], ll[ancestors], prior[ancestors]
        power = next_power
        steps += 1
        for _ in range(6 if power == 1. else 4):
            # The pilot particles use symmetric Metropolis moves at this power.
            scale = np.where(rng.random(n) < .2, .35, 1.) * (2.38 / np.sqrt(dim))
            proposal = z + scale[:, None] * (rng.normal(size=z.shape) @ chol.T)
            theta, new_prior = coords.decode(proposal)
            new_ll = -nll_batch(model, theta, trials)
            keep = np.log(rng.random(n)) < (new_prior - prior + power * (new_ll - ll))
            z[keep], ll[keep], prior[keep] = proposal[keep], new_ll[keep], new_prior[keep]
            accepted += keep.sum()
            attempted += n
    return z, steps, accepted / attempted


def _fit_proposal(pilot, rng):
    """Fit broad and local Student-t components to a tempered pilot cloud."""
    n, dim = pilot.shape
    cov = np.cov(pilot, rowvar=False) + 1e-6 * np.eye(dim)
    center = pilot.mean(axis=0)
    standardized = np.linalg.solve(np.linalg.cholesky(cov), (pilot - center).T).T
    _, labels = kmeans2(standardized, min(8, dim + 2), iter=15, minit="++", rng=rng)
    students = [multivariate_t(center, cov, df=7)]
    counts = []
    for label in np.unique(labels):
        cluster = pilot[labels == label]
        if len(cluster) <= dim:
            continue
        shape = .9 * np.cov(cluster, rowvar=False) + .1 * cov
        students.append(multivariate_t(cluster.mean(axis=0), shape, df=7))
        counts.append(len(cluster))
    proportions = [.04, .22, *(.74 * np.asarray(counts) / sum(counts))]
    return students, proportions


def _proposal_sample(rng, n, coords, students, proportions):
    """IID draws and the full mixture log density, including a defensive prior."""
    choices = rng.choice(len(proportions), n, p=proportions)
    dim = len(coords.lower)
    z = np.empty((n, dim))
    prior_mask = choices == 0
    uniform = rng.random((prior_mask.sum(), dim))
    z[prior_mask] = coords.encode(uniform)
    for i, student in enumerate(students, start=1):
        mask = choices == i
        if mask.any():
            z[mask] = student.rvs(size=mask.sum(), random_state=rng)
    densities = [coords.decode(z)[1]] + [student.logpdf(z) for student in students]
    log_q = logsumexp(np.log(proportions)[:, None] + np.vstack(densities), axis=0)
    return z, log_q


def _normalized(log_weights):
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise ValueError("No finite importance weights. Check trial coding and parameter bounds.")
    return np.exp(log_weights - normalizer)


def _sample_posterior(model, trials, rng, n_rep, n_particles, min_ess, max_particles):
    names = PARAMS[model]
    coords = _Coordinates(model)

    def log_target(z):
        theta, prior = coords.decode(z)
        return prior - nll_batch(model, theta, trials)

    pilot_n = min(2048, max(256, n_particles // 2))
    pilot, temperatures, acceptance = _tempered_pilot(model, trials, coords, rng, pilot_n)
    # Freeze the proposal before collecting the independent production draws.
    students, production_mix = _fit_proposal(pilot, rng)
    chunks, log_weights = [], []
    n_total, size = 0, n_particles
    while n_total < max_particles:
        size = min(size, max_particles - n_total)
        z, log_q = _proposal_sample(rng, size, coords, students, production_mix)
        chunks.append(coords.decode(z)[0])
        log_weights.append(log_target(z) - log_q)
        n_total += size
        theta = np.concatenate(chunks)
        all_logw = np.concatenate(log_weights)
        weights = _normalized(all_logw)
        ess = 1 / (weights @ weights)
        mean = weights @ theta
        sd = np.sqrt(weights @ (theta - mean)**2)
        even = _normalized(all_logw[::2]) @ theta[::2]
        odd = _normalized(all_logw[1::2]) @ theta[1::2]
        split_diff = np.max(np.abs(even - odd) / np.maximum(sd, 1e-12))
        max_weight = weights.max()
        ok = ess >= min_ess and max_weight <= .05 and split_diff <= .25
        if ok:
            break
        size = n_total

    diagnostics = dict(particles=n_total, ess=ess, ess_fraction=ess / n_total,
                       max_weight=max_weight, split_mean_diff_sd=split_diff,
                       posterior_ok=bool(ok), pilot_particles=pilot_n,
                       pilot_temperatures=temperatures, pilot_acceptance=acceptance,
                       proposal_components=len(students) + 1)
    diagnostics.update({f"post_mean_{k}": v for k, v in zip(names, mean)})
    diagnostics.update({f"post_sd_{k}": v for k, v in zip(names, sd)})
    selected = rng.choice(n_total, size=n_rep, replace=True, p=weights)
    return theta[selected], diagnostics


def _stay_cells(trials, pair_mask):
    repeats, n = stay_counts(trials, pair_mask)
    return np.divide(repeats, n, out=np.full(len(CELLS), np.nan), where=n > 0)


def _participant(cohort, participant, group, fit, model, seed, n_rep,
                 n_particles, min_ess, max_particles, adjacent_only):
    group = group.sort_values("trial")
    ids = group["trial"].to_numpy(dtype=float)
    if (not np.isfinite(ids).all() or (ids != np.trunc(ids)).any()
            or (np.diff(ids) <= 0).any()):
        raise ValueError(f"{cohort}/{participant} has invalid or duplicate trial numbers.")
    if not group["valid"].dropna().isin([True, False]).all():
        raise ValueError(f"{cohort}/{participant} has a non-binary validity flag.")
    completed = group.loc[group["valid"].eq(True).fillna(False)]
    if completed.empty:
        raise ValueError(f"{cohort}/{participant} has no completed trials.")
    raw = completed[OBS + ["common"]].to_numpy(dtype=float)
    if not np.isin(raw[:, 0], range(4)).all() or not np.isin(raw[:, 1:], [0, 1]).all():
        raise ValueError(f"{cohort}/{participant} has an invalid completed trial.")
    trials = np.ascontiguousarray(raw[:, :5], dtype=np.int64)
    if not np.array_equal(raw[:, 5], trials[:, 1] == trials[:, 2]):
        raise ValueError(f"{cohort}/{participant} has inconsistent common-transition coding.")
    trial_pos = completed["trial"].to_numpy(dtype=np.int64) - int(ids[0])
    pair_mask = (np.diff(trial_pos) == 1 if adjacent_only
                 else np.ones(len(trials) - 1, dtype=bool))
    observed = _stay_cells(trials, pair_mask)

    probs = completed.reindex(columns=PROB_COLUMNS).to_numpy(dtype=float, na_value=np.nan)
    probs = probs.reshape(-1, 2, 2)
    if np.isinf(probs).any() or ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"{cohort}/{participant} has an invalid reward probability.")
    known = ~np.isnan(probs)
    complete_schedule = bool(known.all())
    seed_fit, seed_reps = seed.spawn(2)
    theta, diag = _sample_posterior(model, trials, np.random.default_rng(seed_fit),
                                    n_rep, n_particles, min_ess, max_particles)
    rng = np.random.default_rng(seed_reps)
    predicted = np.empty((n_rep, 4))
    for i, draw in enumerate(theta):
        schedule = probs
        if not complete_schedule:
            fallback = reward_walk(int(trial_pos[-1]) + 1, rng)[trial_pos]
            schedule = np.where(known, probs, fallback)
        sim_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        simulated = simulate(model, dict(zip(PARAMS[model], draw)), len(trials),
                             sim_seed, states=trials[:, 0], reward_probs=schedule)
        predicted[i] = _stay_cells(simulated, pair_mask)
    diag.update(cohort=cohort, participant=participant, model=model,
                n_valid=len(trials), n_stay_pairs=int(pair_mask.sum()),
                reward_prob_fraction=float(known.mean()), fit_success=fit["success"])
    return cohort, observed, predicted, diag


def _mean_and_count(values):
    n = np.isfinite(values).sum(axis=0)
    mean = np.divide(np.nansum(values, axis=0), n,
                     out=np.full(n.shape, np.nan), where=n > 0)
    return mean, n


def posterior_predictive(data, fits, winners, seed=118, *, n_rep=500,
                         n_particles=2048, min_ess=200, max_particles=16384,
                         n_jobs=-1, adjacent_only=False, return_draws=False):
    """Forward posterior predictive check of the selected family in each cohort.

    The posterior is proportional to the joint two-stage choice likelihood
    times independent Uniform(BOUNDS[name]) priors for each participant. A
    tempered sequential Monte Carlo pilot starts at the full prior and fits a
    mixture of Student-t proposals over bounded-parameter logit coordinates.
    Independent production draws are weighted by the exact target/proposal
    ratio. Sampling doubles to max_particles until importance ESS >= min_ess,
    largest normalized weight <= .05 and odd/even posterior means differ by
    <= .25 posterior SD on every parameter. The diagnostics table flags any
    participant reaching the cap before satisfying all three checks.

    Each replication resamples parameters for every participant and generates
    new choices, transitions and rewards. Learning follows each simulated
    history. Starting states and recorded reward probabilities are fixed.
    Missing probabilities use independent Gaussian walks (SD .025, reflected
    at .25 and .75) over the original trial timeline. Only completed trials
    drive learning. Stay compares successive completed first-stage choices,
    including across omissions, as in the source regression. adjacent_only=True
    restricts comparisons to original consecutive trial numbers.

    Returns (cells, diagnostics). The four cells per cohort compare observed
    participant-mean rates with replicate means and pointwise central 95%
    predictive intervals. Each participant with a defined cell has equal
    weight in that dataset. With return_draws=True the third return value is
    a long DataFrame of the replicate group rates and contributing counts.
    """
    if n_rep < 2 or n_particles < 2 or min_ess <= 0 or max_particles < n_particles:
        raise ValueError("Require n_rep >= 2, 2 <= n_particles <= max_particles, min_ess > 0.")
    groups = list(data.groupby(["cohort", "participant"], sort=True, observed=True))
    if not groups or any(data[c].isna().any() for c in ["cohort", "participant"]):
        raise ValueError("Provide at least one participant and nonmissing cohort/participant IDs.")
    if set(winners) != {cohort for (cohort, _), _ in groups}:
        raise ValueError("winners must specify exactly one model for each supplied cohort.")
    indexed = fits.set_index(["cohort", "participant", "model"], verify_integrity=True)
    seeds = np.random.SeedSequence(seed).spawn(len(groups))
    jobs = []
    for ((cohort, person), group), child in zip(groups, seeds):
        model = winners[cohort]
        if model not in PARAMS:
            raise ValueError(f"Unknown winner {model!r} in cohort {cohort!r}.")
        fit = indexed.loc[(cohort, person, model)]
        jobs.append(delayed(_participant)(cohort, person, group, fit, model, child,
                      n_rep, n_particles, min_ess, max_particles, adjacent_only))
    with threadpool_limits(limits=1, user_api="blas"):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)

    cells, group_draws = [], []
    for cohort, model in winners.items():
        subset = [r for r in results if r[0] == cohort]
        observed, n_obs = _mean_and_count(np.stack([r[1] for r in subset]))
        prediction, n_pred = _mean_and_count(np.stack([r[2] for r in subset]))
        for i, (reward, common) in enumerate(CELLS):
            valid = prediction[:, i][np.isfinite(prediction[:, i])]
            interval = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
            keys = dict(cohort=cohort, model=model, reward=reward, common=common)
            cells.append(dict(**keys, observed=observed[i],
                              pred_mean=valid.mean() if len(valid) else np.nan,
                              lower=interval[0], upper=interval[1],
                              observed_n=int(n_obs[i]), min_pred_n=int(n_pred[:, i].min()),
                              max_pred_n=int(n_pred[:, i].max()), n_rep=len(valid)))
            if return_draws:
                group_draws.append(pd.DataFrame(dict(**keys, draw=np.arange(n_rep),
                                       stay=prediction[:, i], n=n_pred[:, i])))
    diagnostics = pd.DataFrame([r[3] for r in results])
    tables = pd.DataFrame(cells), diagnostics
    return (*tables, pd.concat(group_draws, ignore_index=True)) if return_draws else tables
