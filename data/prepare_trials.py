"""Rebuild the harmonized main-task trials from the authors' pinned data releases."""

import argparse
import gzip
import hashlib
import io
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat


LAB_COMMIT = "4567763780a2c596fd6510af720ec468a8214a8f"
KOOL_COMMIT = "6f849e13a8fe32c0b62ad768d3578068b93ee69e"
LAB_URL = f"https://codeload.github.com/carolfs/muddled_models/zip/{LAB_COMMIT}"
KOOL_URL = (
    f"https://raw.githubusercontent.com/wkool/tradeoffs/{KOOL_COMMIT}"
    "/data/daw%20paradigm/data.mat"
)
LAB_FILES_SHA256 = "5e24176e52a288cac2aaef9e22d94fed95f4acd44a52f006d3473ff95f7a0002"
KOOL_FILE_SHA256 = "47083627c365cc3130cc8f75dd2b989e51112a95b043eaf78013750381eac7d6"
PROBABILITIES = ["p00", "p01", "p10", "p11"]
OUTPUT_FIELDS = [
    "cohort", "participant", "trial", "source_trial", "block", "s1", "a1",
    "a1_left", "a1_side", "s2", "a2", "reward", "common", "valid", "omission_stage",
    *PROBABILITIES,
]


def load_control(source):
    """Read the standard Daw-paradigm arm of Kool, Cushman & Gershman (2016)."""
    content = source.read()
    if hashlib.sha256(content).hexdigest() != KOOL_FILE_SHA256:
        raise ValueError("The MATLAB file differs from the pinned control data.")
    matrix = loadmat(io.BytesIO(content), squeeze_me=True)["data"]
    # The first 19 MATLAB columns match the dictionary in the target paper's loader.
    fields = [
        "participant", "isymbol_lft", "isymbol_rgt", "rt1", "choice1",
        "fsymbol_lft", "fsymbol_rgt", "rt2", "choice2", "reward", "final_state",
        "common", "score", "practice", *PROBABILITIES, "trial",
    ]
    raw = pd.DataFrame(matrix[:, :len(fields)], columns=fields)
    raw = raw.loc[raw["practice"].eq(0)].copy()
    raw[fields[1:]] = raw[fields[1:]].apply(pd.to_numeric)
    raw["slow"] = raw["reward"].eq(-1)
    return normalize(raw, "Common instructions")


def load_lab(path):
    """Read the main choice logs, validating their contents independently of ZIP metadata."""
    lab_rows = {"Spaceship": [], "Magic carpet": []}
    manifest = hashlib.sha256()
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            name = PurePosixPath(member)
            relative = PurePosixPath(*name.parts[1:])
            if relative.parent == PurePosixPath("results/magic_carpet/choices") and name.match("*_game.csv"):
                cohort = "Magic carpet"
            elif (relative.parent == PurePosixPath("results/spaceship/choices")
                  and name.suffix == ".csv" and not name.match("*_practice.csv")):
                cohort = "Spaceship"
            else:
                continue
            content = archive.read(member)
            file_hash = hashlib.sha256(content).hexdigest()
            manifest.update(f"{relative.as_posix()}\t{file_hash}\n".encode("ascii"))
            raw = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
            raw["participant"] = name.stem.split("_")[0]
            lab_rows[cohort].append(raw)
    if manifest.hexdigest() != LAB_FILES_SHA256:
        raise ValueError("The set or contents of the main choice logs differ from the pinned release.")
    return [normalize(pd.concat(rows, ignore_index=True), cohort) for cohort, rows in lab_rows.items()]


def option(values, first):
    """Convert a source binary code to 0/1 and retain the -1 timeout sentinel as missing."""
    return values.mask(values.eq(-1)).astype("Int64") - first


def normalize(raw, cohort):
    spaceship = cohort == "Spaceship"
    control = cohort == "Common instructions"
    raw = raw.reset_index(drop=True)
    data = pd.DataFrame(index=raw.index)
    data["cohort"] = cohort
    data["participant"] = raw["participant"].astype(str)
    data["source_trial"] = raw["trial"].astype(int)
    data["trial"] = data["source_trial"] + int(not spaceship)
    data["valid"] = raw["slow"].eq(0).astype(int)
    data["omission_stage"] = np.select([raw["rt1"].lt(0), raw["rt2"].lt(0)], [1, 2], default=0)
    data["common"] = option(raw["common"], 0)
    data["s2"] = option(raw["final_state"], int(not spaceship))
    data["a2"] = option(raw["choice2"], int(not spaceship))
    data["reward"] = raw["reward"].where(data["valid"].eq(1)).astype("Int64")
    if spaceship:
        data["s1"] = 2 * raw["symbol0"] + raw["symbol1"]
        # The relative action is the planet that this ticket most often reaches.
        data["a1"] = data["s2"].where(data["common"].eq(1), 1 - data["s2"])
        data["a1_left"] = raw["symbol0"] ^ raw["symbol1"]
        data["a1_side"] = option(raw["choice1"], 0)
        source_probs = [f"rwrd_prob{i}" for i in range(4)]
    else:
        data["s1"] = 0
        data["a1"] = option(raw["choice1"], 1)
        data["a1_left"] = option(raw["isymbol_lft"], 1)
        data["a1_side"] = data["a1"].ne(data["a1_left"]).astype("Int64")
        source_probs = PROBABILITIES if control else [
            "reward.1.1", "reward.1.2", "reward.2.1", "reward.2.2"
        ]
    data[PROBABILITIES] = raw[source_probs].astype(float)

    data["block"] = 1
    if not control:
        # Each task offers a break after a preset number of attempts, except near the end.
        completed = data.groupby("participant", sort=False)["valid"].cumsum()
        if spaceship:
            break_after = data["trial"].mod(84).eq(0) & completed.lt(240)
        else:
            break_after = data["trial"].mod(67).eq(0) & completed.le(191)
        breaks_so_far = break_after.groupby(data["participant"], sort=False).cumsum()
        data["block"] += breaks_so_far - break_after.astype(int)
    return data[OUTPUT_FIELDS]


def validate(trials):
    """Check the sample, chronology, transition mapping and reward schedules."""
    summary = trials.groupby("cohort").agg(
        participants=("participant", "nunique"), attempts=("trial", "size"), complete=("valid", "sum")
    )
    expected = {
        "Common instructions": (206, 25750, 24750),
        "Spaceship": (21, 5273, 5250),
        "Magic carpet": (24, 4934, 4824),
    }
    if {cohort: tuple(row) for cohort, row in summary.iterrows()} != expected:
        raise ValueError(f"Unexpected sample counts:\n{summary}")
    successive = trials.groupby(["cohort", "participant"], sort=False).cumcount() + 1
    if not trials["trial"].eq(successive).all():
        raise ValueError("Trial order disagrees with the source counters.")
    if not trials["valid"].eq(trials["omission_stage"].eq(0)).all():
        raise ValueError("Reaction-time and outcome timeout indicators disagree.")
    complete = trials.loc[trials["valid"].eq(1)]
    if not complete[["a1", "s2", "a2", "reward", "common"]].isin([0, 1]).all().all():
        raise ValueError("A completed trial is missing a binary action, state or outcome.")
    reached = trials.loc[trials["s2"].notna()]
    if not reached["a1"].eq(reached["s2"]).eq(reached["common"]).all():
        raise ValueError("The relative first action conflicts with the recorded transition.")
    if not reached["a1"].ne(reached["a1_left"]).eq(reached["a1_side"]).all():
        raise ValueError("The first-stage action conflicts with the displayed left option.")
    if not np.logical_and(trials[PROBABILITIES] >= 0, trials[PROBABILITIES] <= 1).all().all():
        raise ValueError("A reward schedule is missing a valid probability.")
    return summary


def open_source(local_path, url):
    if local_path is not None:
        return local_path.open("rb")
    with urlopen(url, timeout=120) as response:
        return io.BytesIO(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muddled-zip", type=Path, help="Use an existing pinned GitHub ZIP archive.")
    parser.add_argument("--kool-mat", type=Path, help="Use an existing pinned Daw-paradigm data.mat.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trials.csv.gz"))
    args = parser.parse_args()
    with open_source(args.kool_mat, KOOL_URL) as source:
        control = load_control(source)
    with open_source(args.muddled_zip, LAB_URL) as source:
        trials = pd.concat([control, *load_lab(source)], ignore_index=True)
    summary = validate(trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A fixed gzip header and newline convention make same-runtime rebuilds byte-identical.
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stage:
        stage_path = Path(stage.name)
    try:
        with stage_path.open("wb") as file:
            with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
                    trials.to_csv(output, index=False, lineterminator="\n")
        stage_path.replace(args.output)
    finally:
        stage_path.unlink(missing_ok=True)
    print(summary.to_string())
    print(f"Saved {len(trials):,} rows to {args.output}")


if __name__ == "__main__":
    main()
