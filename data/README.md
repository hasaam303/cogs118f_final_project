# Published two-stage task trials

`trials.csv.gz` contains one row per main-task attempt for the three cohorts in Feher da Silva and Hare (2020). It includes partial trials so the original order can be reconstructed. Each row is identified by `(cohort, participant, trial)`. [Target paper](https://doi.org/10.1038/s41562-020-0905-y) · [Authors’ data loader](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/analyses_simulations/utils.py)

| Cohort | Participants | Main attempts | Complete | Missed stage 1 | Missed stage 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common instructions | 206 | 25,750 | 24,750 | 722 | 278 |
| Spaceship | 21 | 5,273 | 5,250 | 14 | 9 |
| Magic carpet | 24 | 4,934 | 4,824 | 56 | 54 |
| **Total** | **251** | **35,957** | **34,824** | **792** | **341** |

## Sources and exclusions

- **Common instructions.** Kool, Cushman and Gershman (2016) released the standard Daw-paradigm arm in `data/daw paradigm/data.mat`. It contains 206 participant IDs, each with 25 practice and 125 main trials. We remove the 5,150 practice rows using the source practice flag and retain all 206 participants, as Feher da Silva and Hare did. Kool et al. report 207 participants completed that arm. Their separate 2016 analysis excluded participants exceeding 25 missed trials. Nine of the 206 released records meet that criterion, yielding 197 under their rule. Their MATLAB code uses a strict “fewer than 25” inclusion test and nobody in this file missed exactly 25. The table uses the original `subject_1` through `subject_206` IDs. The target authors numbered them 0 through 205. [Kool et al. paper](https://doi.org/10.1371/journal.pcbi.1005090) · [Pinned dataset](https://github.com/wkool/tradeoffs/blob/6f849e13a8fe32c0b62ad768d3578068b93ee69e/data/daw%20paradigm/data.mat) · [2016 analysis code](https://github.com/wkool/tradeoffs/blob/6f849e13a8fe32c0b62ad768d3578068b93ee69e/data/daw%20paradigm/groupanalysis.m)
- **Spaceship.** All 21 main choice CSVs in `results/spaceship/choices` supply 250 completed trials each. Attempts that timed out generated replacement trials, so some files contain more than 250 rows. The 21 separate `*_practice.csv` files are omitted. Four other identifiers have information files only. [Pinned choice files](https://github.com/carolfs/muddled_models/tree/4567763780a2c596fd6510af720ec468a8214a8f/results/spaceship/choices) · [Task program](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/spaceship/exp.py#L763-L884)
- **Magic carpet.** All 24 `*_game.csv` files in `results/magic_carpet/choices` supply 201 completed trials each. This is the task program’s stopping rule. It also generated replacement trials after timeouts. The 24 `*_tutorial.csv` files are omitted. Twelve other identifiers have configuration files only. Both Zurich cohorts retain the five-digit ID from the choice filename, including leading zeroes. [Pinned choice files](https://github.com/carolfs/muddled_models/tree/4567763780a2c596fd6510af720ec468a8214a8f/results/magic_carpet/choices) · [Task configuration](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/magic_carpet/exp.py#L72-L84)

## Column dictionary

| Column | Meaning |
| --- | --- |
| `cohort`, `participant` | Published condition and source participant ID. Read the ID as a string. |
| `trial`, `source_trial` | `trial` counts all main attempts from 1. The original counter starts at 0 for Common instructions and Magic carpet and at 1 for Spaceship. |
| `block` | One-based block reconstructed from scheduled breaks in the task programs. Common instructions has one main block. Spaceship offers a break after every 84 attempts if fewer than 240 trials are complete. Magic carpet offers one after every 67 attempts if at most 191 trials are complete. |
| `s1` | First-stage configuration. It is 0 for Common instructions and Magic carpet. In Spaceship, `2 * symbol0 + symbol1` codes the four possible announcements, where `symbol0` is the displayed ship and `symbol1` the displayed planet. |
| `a1` | Relative first action, indexed by the state it usually reaches. A choice of 0 usually reaches state 0. For Spaceship the same button can target different planets under different announcements. |
| `a1_left`, `a1_side` | Relative action offered on the left and the participant’s physical first-stage key (0 left, 1 right). The two symbol-based tasks identify the left option in the raw log. In Spaceship the announcement determines it. |
| `s2`, `a2` | Reached second-stage state and the option chosen there, each 0 or 1. `a2` identifies a symbol within its state for Common instructions and Magic carpet. In Spaceship it identifies the left or right second-stage action. |
| `reward` | 1 for a reward, 0 for a completed trial without reward, blank for a timeout. |
| `common` | 1 for common, 0 for rare. Among trials that reached stage two, `common == (a1 == s2)`. On first-stage timeouts the Zurich experiments logged the preselected transition flag, while Kool et al. stored a missing transition. |
| `valid`, `omission_stage` | `valid` is 1 for a completed trial and 0 for an omission. `omission_stage` is 0 for complete, 1 if the first response timed out, or 2 if only the second response timed out. Observed first-stage choices and reached states survive a stage-two timeout. |
| `p00`, `p01`, `p10`, `p11` | Scheduled reward probabilities indexed by `(s2, a2)` in that order. All four are present on every attempt, including omissions. |

The reward schedule columns come from `ps1a1, ps1a2, ps2a1, ps2a2` in Kool et al., `rwrd_prob0` through `rwrd_prob3` in Spaceship, and `reward.1.1, reward.1.2, reward.2.1, reward.2.2` in Magic carpet. The Spaceship runtime indexes probabilities by `2 * final_state + choice2`. The Magic carpet runtime writes each probability using its reached state and symbol within that state. [Kool variable definitions](https://github.com/wkool/tradeoffs/blob/6f849e13a8fe32c0b62ad768d3578068b93ee69e/data/daw%20paradigm/make_raw_data.m) · [Spaceship reward draw](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/spaceship/exp.py#L859-L876) · [Magic carpet schedule](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/magic_carpet/exp.py#L349-L358)

For a complete-trial likelihood, use `valid == 1`. The source marks a timeout as `win = -1` in Common instructions and as `slow = 1, reward = 0` in the Zurich tasks. Feher da Silva and Hare filtered timeouts before constructing stay pairs, which allows a previous completed trial to precede one or more omissions and allows a pair to cross a break. For strictly consecutive attempts within a block, pair original rows within participant and block and require both rows to be valid. [Published logistic-regression code](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/analyses_simulations/logistic_regression_human_data.py#L114-L126)

## Rebuild and verify

The included table works offline. To regenerate it with Python, NumPy, pandas and SciPy, run this command from the project root.

```sh
python data/prepare_trials.py
```

The script downloads the pinned GitHub archive and MATLAB file. To use local copies, pass `--muddled-zip PATH --kool-mat PATH`. The optional `--output PATH` writes a separate copy. The script verifies the following SHA-256 digests, checks trial counts, chronology, transition coding and probability ranges, then writes the table with a fixed gzip timestamp.

| Input or output | SHA-256 |
| --- | --- |
| 45 selected Zurich choice files, combined manifest | `5e24176e52a288cac2aaef9e22d94fed95f4acd44a52f006d3473ff95f7a0002` |
| Kool et al. `data.mat` | `47083627c365cc3130cc8f75dd2b989e51112a95b043eaf78013750381eac7d6` |
| Included uncompressed CSV | `0dfd741fff3ca31791ce39e4265dcef957aa08b7b916e370a3007f6f81d27651` |

The Zurich manifest hashes one line per file, sorted by path. Each UTF-8 line contains the path relative to the repository root, a tab, the lowercase hexadecimal SHA-256 of the raw file bytes, and a line feed.

## Attribution and terms

The Zurich data come from Carolina Feher da Silva and Todd A. Hare, *Humans primarily use model-based inference in the two-stage task*, Nature Human Behaviour 4, 1053–1066 (2020). The authors’ repository carries the GNU General Public License, version 3. Its license text is included here. [Repository license](https://github.com/carolfs/muddled_models/blob/4567763780a2c596fd6510af720ec468a8214a8f/LICENSE.txt) · [Local license copy](LICENSE-muddled-models.txt)

The control data come from Wouter Kool, Fiery A. Cushman and Samuel J. Gershman, *When Does Model-Based Control Pay Off?*, PLOS Computational Biology 12, e1005090 (2016). The article is distributed under Creative Commons Attribution. The linked data repository provides no separate license file at the pinned revision. [Article and data availability statement](https://doi.org/10.1371/journal.pcbi.1005090) · [Pinned repository](https://github.com/wkool/tradeoffs/tree/6f849e13a8fe32c0b62ad768d3578068b93ee69e)
