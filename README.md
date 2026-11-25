# Two-stage choice and model recovery

Open the [executed replication notebook](two_stage_replication.ipynb) for a reproduction of Feher da Silva and Hare’s two-stage decision study. It explains the five learning models, simulates their behavior, tests parameter and model recovery, and compares the fits to 34,824 completed trials from 251 participants. [Published paper](https://doi.org/10.1038/s41562-020-0905-y)

The three simulated planning agents received median fitted hybrid weights of **0.943**, **0.355**, and **0.804** for the correct task model, unlucky-symbol belief, and transition-dependent learning rates. The paper reports **0.94**, **0.36**, and **0.80**. The human data also reproduce the stronger reward-by-transition interaction in the Spaceship and Magic carpet cohorts. [Simulation comparison](outputs/figure2_weight_comparison.csv) · [Human estimates](outputs/human_stay_contrasts.csv)

Summed participant BIC favors the simple model-based family in all three cohorts. AIC favors TDLR for common instructions and the hybrid for both story tasks. In 500 replicated datasets per cohort, all 12 observed stay rates fall outside the BIC winners’ pointwise 95% posterior predictive intervals. The largest discrepancy is in Spaceship after an unrewarded common transition, where participants repeat 42.0% of choices and the model predicts 52.9%. [Model comparison](outputs/model_comparison.csv) · [Predictive check](outputs/posterior_predictive.png)

Section 6 tests how often BIC identifies a hybrid learner as the task gets longer. For the same 100 learners with equal planning and model-free weights, BIC selected the hybrid **3, 16, 34, and 57 times** after 125, 250, 500, and 1,000 trials. With a true planning weight of 0.9, it selected the pure planner for **99 of 100** learners even at 1,000 trials. These are individual model-selection rates under the simulated parameter ranges. [Heatmaps with 95% intervals](outputs/hybrid_recovery_grid.png) · [Rates for all five candidate models](outputs/hybrid_recovery_rates.csv)

## Reproduce the analysis

Use Python 3.11 and run the following from the project root in an activated environment.

```sh
python -m pip install -r requirements.txt
```

Open the notebook in Jupyter or VS Code, select that Python environment, and run all cells. Set `REFIT = True` in the first code cell to regenerate the expensive stages. By default, the notebook verifies the saved results against digests of the inputs, algorithms and settings. For a fresh computation from a terminal, run:

```sh
python -m replication.study all --refresh --jobs 6
python -m replication.hybrid_recovery --refresh --jobs 6
```

The [data guide](data/README.md) documents the two pinned upstream sources, inclusion rules, trial coding, and independent rebuild. The included trial table supports offline use. [Analysis code](replication/) · [Figures and fit tables](outputs/)
