"""Choice likelihoods and simulators for Feher da Silva & Hare's two-step task.

Trials have columns (s1, a1, s2, a2, reward). All codes are zero-based.
The first action is common when a1 == s2. A single participant's complete
decision trials must be passed in temporal order.
"""

import math
from collections.abc import Mapping
from operator import index

import numpy as np
from numba import njit
from scipy.optimize import minimize


PARAMS = {
    "MF": ("alpha1", "alpha2", "lam", "beta1", "beta2", "p"),
    "MB": ("alpha2", "beta1", "beta2", "p"),
    "Hybrid": ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p"),
    "Unlucky": ("alpha2", "beta1", "beta2", "eta", "p"),
    "TDLR": ("alpha_common", "alpha_rare", "beta1", "beta2", "p"),
}
BOUNDS = {
    "alpha1": (0.0, 1.0), "alpha2": (0.0, 1.0), "lam": (0.0, 1.0),
    "beta1": (0.01, 20.0), "beta2": (0.01, 20.0), "w": (0.0, 1.0),
    "p": (-2.0, 2.0), "eta": (0.0, 1.0),
    "alpha_common": (0.0, 1.0), "alpha_rare": (0.0, 1.0),
}

_ALL = tuple(BOUNDS)
_A1, _A2, _LAM, _B1, _B2, _W, _P, _ETA, _COMMON, _RARE = range(len(_ALL))
_MF, _MB, _HYBRID, _UNLUCKY, _TDLR = range(5)
_FAMILY = {"MF": _MF, "MB": _MB, "Hybrid": _HYBRID, "Unlucky": _UNLUCKY, "TDLR": _TDLR}
_FREE = {m: np.array([_ALL.index(p) for p in names]) for m, names in PARAMS.items()}
_LIMITS = {m: np.array([BOUNDS[p] for p in names]) for m, names in PARAMS.items()}
_BASE = {
    "alpha1": 0.4, "alpha2": 0.45, "lam": 0.55, "w": 0.5, "eta": 0.55,
    "alpha_common": 0.65, "alpha_rare": 0.35, "beta1": 4.0, "beta2": 4.0, "p": 0.0,
}


def _parameter_array(model, values):
    if isinstance(values, Mapping):
        values = [values[name] for name in PARAMS[model]]
    values = np.asarray(values, dtype=float)
    if values.shape != (len(PARAMS[model]),):
        raise ValueError(f"{model} parameters must follow {PARAMS[model]}")
    return values


def _observations(trials):
    data = np.asarray(trials)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("trials must have shape (n_trials, 5)")
    if not np.issubdtype(data.dtype, np.integer):
        if not np.issubdtype(data.dtype, np.floating) or not np.all(data == np.floor(data)):
            raise ValueError("trials must contain integer state, action and reward codes")
    if len(data) and (np.any(data < 0) or np.any(data > [3, 1, 1, 1, 1])):
        raise ValueError("s1 must be 0..3 and all other columns must be 0 or 1")
    return np.ascontiguousarray(data, dtype=np.int64)


@njit(cache=True, nogil=True)
def _pack(family, values, free):
    """Use one internal parameter layout. Pure MB families fix w=1, pure MF fixes w=0."""
    full = np.zeros(10)
    full[_W] = 0.0 if family == _MF else 1.0
    full[_ETA] = 1.0
    full[free] = values
    return full


@njit(inline="always")
def _first_stage(family, p, best0, best1, mf_difference, previous_sign):
    mb_difference = 0.4 * (best1 - best0)
    if family == _UNLUCKY:
        # Action 0 devalues both possible successor states by the factor eta.
        action0_value = 0.7 * best0 + 0.3 * best1
        mb_difference += (1.0 - p[_ETA]) * action0_value
    mixed = p[_W] * mb_difference + (1.0 - p[_W]) * mf_difference
    return mixed + p[_P] * previous_sign, mb_difference


@njit(inline="always")
def _second_stage_scale(family, p, a1):
    return p[_ETA] if family == _UNLUCKY and a1 == 0 else 1.0


@njit(inline="always")
def _value_rate(family, p, common):
    if family == _TDLR:
        return p[_COMMON] if common else p[_RARE]
    return p[_A2]


@njit(inline="always")
def _learning_errors(q, v, reward, lam):
    reward_error = reward - v
    return v - q + lam * reward_error, reward_error


@njit(inline="always")
def _bernoulli(logit, choice):
    """Return stable negative log probability and its derivative w.r.t. the logit."""
    if logit >= 0.0:
        e = math.exp(-logit)
        return math.log1p(e) + (1 - choice) * logit, 1.0 / (1.0 + e) - choice
    e = math.exp(logit)
    return math.log1p(e) - choice * logit, e / (1.0 + e) - choice


@njit(cache=True, nogil=True)
def _replay(family, p, trials, with_gradient):
    """Replay both choices before learning; propagate derivatives in physical units."""
    q = np.zeros((4, 2))
    v = np.zeros((2, 2))
    # Only the learning parameters affect stored values. dv indexes the single
    # learning rate, or the common rate for TDLR. dr indexes TDLR's rare rate.
    dv = np.zeros((2, 2))
    dr = np.zeros((2, 2))
    dq_alpha1 = np.zeros((4, 2))
    dq_alpha2 = np.zeros((4, 2))
    dq_lambda = np.zeros((4, 2))
    grad = np.zeros(10)
    uses_q = family == _MF or family == _HYBRID
    rate_parameter = _COMMON if family == _TDLR else _A2
    previous_sign = 0
    total = 0.0

    for t in range(trials.shape[0]):
        s1, a1, s2, a2, reward = trials[t]
        best0 = int(v[0, 1] > v[0, 0])
        best1 = int(v[1, 1] > v[1, 0])
        mf_difference = q[s1, 1] - q[s1, 0]
        first, mb_difference = _first_stage(
            family, p, v[0, best0], v[1, best1], mf_difference, previous_sign
        )
        scale = _second_stage_scale(family, p, a1)
        v_difference = v[s2, 1] - v[s2, 0]
        loss1, residual1 = _bernoulli(p[_B1] * first, a1)
        loss2, residual2 = _bernoulli(p[_B2] * scale * v_difference, a2)
        total += loss1 + loss2

        common = a1 == s2
        rate = _value_rate(family, p, common)
        q_error, v_error = _learning_errors(q[s1, a1], v[s2, a2], reward, p[_LAM])

        if with_gradient:
            first_slope = residual1 * p[_B1]
            second_slope = residual2 * p[_B2]
            grad[_B1] += residual1 * first
            grad[_B2] += residual2 * scale * v_difference
            grad[_P] += first_slope * previous_sign

            dmb = 0.4 * (dv[1, best1] - dv[0, best0])
            if family == _UNLUCKY:
                dmb += (1.0 - p[_ETA]) * (0.7 * dv[0, best0] + 0.3 * dv[1, best1])
                grad[_ETA] -= first_slope * (0.7 * v[0, best0] + 0.3 * v[1, best1])
                if a1 == 0:
                    grad[_ETA] += second_slope * v_difference
            grad[rate_parameter] += (
                first_slope * p[_W] * dmb + second_slope * scale * (dv[s2, 1] - dv[s2, 0])
            )
            if family == _TDLR:
                grad[_RARE] += first_slope * 0.4 * (dr[1, best1] - dr[0, best0])
                grad[_RARE] += second_slope * (dr[s2, 1] - dr[s2, 0])
                dr[s2, a2] = (1.0 - rate) * dr[s2, a2] + (0.0 if common else v_error)
            if uses_q:
                mf_slope = first_slope * (1.0 - p[_W])
                grad[_A1] += mf_slope * (dq_alpha1[s1, 1] - dq_alpha1[s1, 0])
                grad[_A2] += mf_slope * (dq_alpha2[s1, 1] - dq_alpha2[s1, 0])
                grad[_LAM] += mf_slope * (dq_lambda[s1, 1] - dq_lambda[s1, 0])
                grad[_W] += first_slope * (mb_difference - mf_difference)
                decay = 1.0 - p[_A1]
                dq_alpha1[s1, a1] = decay * dq_alpha1[s1, a1] + q_error
                dq_alpha2[s1, a1] = decay * dq_alpha2[s1, a1] + p[_A1] * (1.0 - p[_LAM]) * dv[s2, a2]
                dq_lambda[s1, a1] = decay * dq_lambda[s1, a1] + p[_A1] * v_error
            dv[s2, a2] = (1.0 - rate) * dv[s2, a2]
            if family != _TDLR or common:
                dv[s2, a2] += v_error

        if uses_q:
            q[s1, a1] += p[_A1] * q_error
        v[s2, a2] += rate * v_error
        previous_sign = 2 * a1 - 1
    return total, grad


@njit(cache=True, nogil=True)
def _batch_kernel(family, parameters, free, limits, trials):
    result = np.empty(len(parameters))
    for i in range(len(parameters)):
        values = parameters[i]
        if np.any(~np.isfinite(values)) or np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
            result[i] = np.inf
        else:
            result[i] = _replay(family, _pack(family, values, free), trials, False)[0]
    return result


def nll_batch(model, theta_matrix, trials):
    """Joint choice NLL for each parameter row in PARAMS order; invalid rows get +inf."""
    data = _observations(trials)
    matrix = np.asarray(theta_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(PARAMS[model]):
        raise ValueError(f"theta_matrix must have shape (n_draws, {len(PARAMS[model])})")
    return _batch_kernel(_FAMILY[model], np.ascontiguousarray(matrix), _FREE[model], _LIMITS[model], data)


def nll(model, theta, trials):
    """Negative joint conditional log likelihood of both decisions, summed over trials."""
    return float(nll_batch(model, _parameter_array(model, theta)[None, :], trials)[0])


@njit(cache=True, nogil=True)
def _fit_objective(coordinates, family, free, trials, paper_prior):
    values = coordinates.copy()
    for i, parameter in enumerate(free):
        if parameter == _B1 or parameter == _B2:
            values[i] = math.exp(values[i])
    full = _pack(family, values, free)
    loss, grad = _replay(family, full, trials, True)
    if paper_prior:
        # These are the beta and perseveration priors in hybrid_no_log_lik.stan.
        for parameter in (_B1, _B2):
            beta = full[parameter]
            loss += math.log1p((beta / 5.0) ** 2)
            grad[parameter] += 2.0 * beta / (25.0 + beta * beta)
        perseverance = full[_P]
        loss += math.log1p(perseverance * perseverance)
        grad[_P] += 2.0 * perseverance / (1.0 + perseverance * perseverance)
    for parameter in (_B1, _B2):
        grad[parameter] *= full[parameter]  # optimize in log-beta coordinates
    return loss, grad[free]


def _start_points(model, starts, rng, initial_points):
    names = PARAMS[model]
    if initial_points is not None:
        for point in initial_points:
            yield _parameter_array(model, point).copy()
    yield np.array([_BASE[name] for name in names])
    for _ in range(starts - 1):
        point = []
        for name in names:
            if name in ("beta1", "beta2"):
                point.append(math.exp(rng.uniform(math.log(0.7), math.log(13))))
            elif name == "p":
                point.append(rng.uniform(-0.5, 0.5))
            else:
                point.append(rng.uniform(0.05, 0.95))
        yield np.array(point)


def fit(model, trials, seed=0, starts=5, paper_prior=False, initial_points=None):
    """Bounded multi-start fit. Returned parameters and nll are in physical units.

    starts counts the seeded attempts. Optional initial_points vectors are
    tried as additional starts. paper_prior adds the published Figure 2 hybrid
    simulation penalties to the optimized objective. nll always excludes them.
    """
    data = _observations(trials)
    if not len(data) or index(starts) < 1:
        raise ValueError("fit requires observed trials and at least one start")
    if paper_prior and model != "Hybrid":
        raise ValueError("the published simulation prior is defined for the Hybrid model")
    names = PARAMS[model]
    limits = _LIMITS[model]
    beta_columns = np.array([names.index("beta1"), names.index("beta2")])
    optimizer_bounds = limits.copy()
    optimizer_bounds[beta_columns] = np.log(optimizer_bounds[beta_columns])
    args = (_FAMILY[model], _FREE[model], data, bool(paper_prior))
    best_loss = np.inf
    best = None
    best_success = False
    converged = 0
    attempted = 0
    for start in _start_points(model, starts, np.random.default_rng(seed), initial_points):
        if not np.all(np.isfinite(start)) or np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
            raise ValueError(f"initial parameters must be within the bounds for {model}")
        start[beta_columns] = np.log(start[beta_columns])
        result = minimize(
            _fit_objective, start, args=args, method="L-BFGS-B", jac=True,
            bounds=optimizer_bounds, options={"maxiter": 300, "maxls": 45, "ftol": 1e-10},
        )
        attempted += 1
        converged += int(result.success)
        # Some line-search exits report an objective at a discarded trial point.
        loss = _fit_objective(result.x, *args)[0]
        # At a boundary, two starts can differ only by floating-point noise.
        # Keep the converged solution when the objective is indistinguishable.
        tied = abs(loss - best_loss) <= 1e-8
        if (best is None or loss < best_loss - 1e-8
                or (tied and result.success and (not best_success or loss < best_loss))):
            best, best_loss, best_success = result.x.copy(), loss, bool(result.success)
    if best is None:
        raise RuntimeError(f"No finite objective after {attempted} starts for {model}")
    if not best_success:
        retry = minimize(_fit_objective, best.copy(), args=args, method="L-BFGS-B", jac=True,
                         bounds=optimizer_bounds, options={"maxiter": 300, "maxls": 80, "ftol": 1e-10})
        attempted += 1
        converged += int(retry.success)
        retry_loss = _fit_objective(retry.x, *args)[0]
        if retry.success and retry_loss <= best_loss + 1e-8:
            best, best_loss, best_success = retry.x.copy(), retry_loss, True
    best[beta_columns] = np.exp(best[beta_columns])
    best = np.clip(best, limits[:, 0], limits[:, 1])
    full = _pack(_FAMILY[model], best, _FREE[model])
    raw_nll = _replay(_FAMILY[model], full, data, False)[0]
    return {
        **dict(zip(names, best.tolist())), "nll": float(raw_nll), "success": best_success,
        "objective": float(best_loss), "n_starts": attempted, "n_converged": converged,
    }


@njit(cache=True, nogil=True)
def _reflected_schedule(start, increments):
    schedule = np.empty((len(increments) + 1, 2, 2))
    schedule[0] = start
    for t in range(len(increments)):
        for state in range(2):
            for action in range(2):
                candidate = schedule[t, state, action] + increments[t, state, action]
                phase = (candidate - 0.25) % 1.0
                schedule[t + 1, state, action] = 0.75 - abs(phase - 0.5)
    return schedule


def reward_walk(n_trials, rng):
    """Four independent reflected Gaussian walks as in the authors' utils.py.

    Each starts uniformly on [.25,.75] and moves by N(0,.025²) per trial.
    Reflection on this interval is periodic with period 1, so reducing a
    Gaussian increment modulo 1 before reflection gives the same next value.
    """
    n_trials = index(n_trials)
    if n_trials < 0:
        raise ValueError("n_trials must be nonnegative")
    if not n_trials:
        return np.empty((0, 2, 2))
    rng = np.random.default_rng(rng)
    return _reflected_schedule(
        rng.uniform(0.25, 0.75, size=(2, 2)),
        rng.normal(0, 0.025, size=(n_trials - 1, 2, 2)),
    )


@njit(inline="always")
def _choose(logit, uniform):
    return int(uniform < 1.0 / (1.0 + math.exp(-logit)))


@njit(cache=True, nogil=True)
def _simulate(family, p, states, schedule, uniforms):
    q = np.zeros((4, 2))
    v = np.zeros((2, 2))
    data = np.empty((len(states), 5), dtype=np.int64)
    previous_sign = 0
    for t in range(len(states)):
        s1 = states[t]
        first, _ = _first_stage(
            family, p, max(v[0, 0], v[0, 1]), max(v[1, 0], v[1, 1]),
            q[s1, 1] - q[s1, 0], previous_sign,
        )
        a1 = _choose(p[_B1] * first, uniforms[t, 0])
        common = uniforms[t, 1] < 0.7
        s2 = a1 if common else 1 - a1
        scale = _second_stage_scale(family, p, a1)
        a2 = _choose(p[_B2] * scale * (v[s2, 1] - v[s2, 0]), uniforms[t, 2])
        reward = int(uniforms[t, 3] < schedule[t, s2, a2])
        q_error, v_error = _learning_errors(q[s1, a1], v[s2, a2], reward, p[_LAM])
        if family == _MF or family == _HYBRID:
            q[s1, a1] += p[_A1] * q_error
        v[s2, a2] += _value_rate(family, p, common) * v_error
        data[t, 0], data[t, 1], data[t, 2], data[t, 3], data[t, 4] = s1, a1, s2, a2, reward
        previous_sign = 2 * a1 - 1
    return data


def simulate(model, params, n_trials, seed, states=None, reward_probs=None):
    """Generate both choices, transitions and rewards for one participant.

    states defaults to a single first-stage state, as in the published agent
    simulations. reward_probs[t,s2,a2] defaults to four drifting probabilities.
    Passing an observed schedule conditions the simulated environment on it.
    """
    values = _parameter_array(model, params)
    limits = _LIMITS[model]
    if not np.all(np.isfinite(values)) or np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
        raise ValueError(f"simulation parameters must be within the bounds for {model}")
    n_trials = index(n_trials)
    if n_trials < 0:
        raise ValueError("n_trials must be nonnegative")
    rng = np.random.default_rng(seed)
    if states is None:
        states = np.zeros(n_trials, dtype=np.int64)
    else:
        states = np.asarray(states)
        if states.shape != (n_trials,) or not np.isin(states, [0, 1, 2, 3]).all():
            raise ValueError("states must contain one first-stage state (0..3) per trial")
        states = np.ascontiguousarray(states, dtype=np.int64)
    if reward_probs is None:
        schedule = reward_walk(n_trials, rng)
    else:
        schedule = np.ascontiguousarray(reward_probs, dtype=float)
        if schedule.shape != (n_trials, 2, 2) or not np.isfinite(schedule).all() or np.any((schedule < 0) | (schedule > 1)):
            raise ValueError("reward_probs must have shape (n_trials, 2, 2) and values in [0,1]")
    full = _pack(_FAMILY[model], values, _FREE[model])
    return _simulate(_FAMILY[model], full, states, schedule, rng.random((n_trials, 4)))
