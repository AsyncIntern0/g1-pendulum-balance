import os
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "eda_results"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def describe_array(arr):
    """Return basic statistics for an array."""
    arr = np.asarray(arr, dtype=np.float64)

    if arr.size == 0:
        return {
            "shape": str(arr.shape),
            "size": 0,
            "nan_count": 0,
            "inf_count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
        }

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "shape": str(arr.shape),
            "size": arr.size,
            "nan_count": np.isnan(arr).sum(),
            "inf_count": np.isinf(arr).sum(),
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
        }

    return {
        "shape": str(arr.shape),
        "size": arr.size,
        "nan_count": np.isnan(arr).sum(),
        "inf_count": np.isinf(arr).sum(),
        "min": finite.min(),
        "max": finite.max(),
        "mean": finite.mean(),
        "std": finite.std(),
        "median": np.median(finite),
    }


def magnitude(arr):
    """
    Compute vector magnitude along the last dimension.

    Example:
        gyro shape = (N, 3)
        output     = (N,)
    """
    arr = np.asarray(arr)

    if arr.ndim >= 2 and arr.shape[-1] > 1:
        return np.linalg.norm(arr, axis=-1)

    return np.abs(arr)


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(dataset_dir):
    labels_path = os.path.join(dataset_dir, "labels.csv")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Could not find labels.csv in: {dataset_dir}"
        )

    labels = pd.read_csv(labels_path)

    print("\n============================================================")
    print("LABEL DATASET OVERVIEW")
    print("============================================================")

    print(f"Number of label rows: {len(labels)}")
    print(f"Number of columns:    {len(labels.columns)}")

    print("\nColumns:")
    for col in labels.columns:
        print(f"  - {col}")

    return labels


# ============================================================
# BASIC LABEL EDA
# ============================================================

def analyze_labels(labels, output_dir):

    print("\n============================================================")
    print("LABEL DISTRIBUTION")
    print("============================================================")

    if "fell" in labels.columns:
        print("\nFall distribution:")
        print(labels["fell"].value_counts())

    if "stable" in labels.columns:
        print("\nStable distribution:")
        print(labels["stable"].value_counts())

    if "scenario_fn" in labels.columns:
        scenario_summary = (
            labels.groupby("scenario_fn")
            .agg(
                trials=("trial_id", "count"),
                falls=("fell", "sum"),
                stable=("stable", "sum"),
            )
        )

        scenario_summary["fall_rate"] = (
            scenario_summary["falls"] / scenario_summary["trials"]
        )

        scenario_summary.to_csv(
            os.path.join(output_dir, "scenario_summary.csv")
        )

        print("\nScenario summary:")
        print(scenario_summary)

    if "magnitude_level" in labels.columns:
        print("\nMagnitude levels:")
        print(labels["magnitude_level"].value_counts().sort_index())

    if "timing_level" in labels.columns:
        print("\nTiming levels:")
        print(labels["timing_level"].value_counts().sort_index())

    # Save a copy
    labels.describe(include="all").transpose().to_csv(
        os.path.join(output_dir, "labels_descriptive_statistics.csv")
    )


# ============================================================
# SCAN NPZ FILES
# ============================================================

def scan_npz_files(dataset_dir, labels, output_dir):

    npz_files = sorted(
        glob.glob(os.path.join(dataset_dir, "*.npz"))
    )

    print("\n============================================================")
    print("NPZ FILE SCAN")
    print("============================================================")

    print(f"NPZ files found: {len(npz_files)}")

    if len(npz_files) == 0:
        raise RuntimeError("No .npz files found.")

    label_map = {
        str(row["trial_id"]): row
        for _, row in labels.iterrows()
    }

    required_arrays = [
        "time",
        "gyro",
        "acc",
        "qpos",
        "qvel",
    ]

    records = []
    problems = []

    for index, path in enumerate(npz_files):

        filename = os.path.basename(path)
        trial_id = filename.replace(".npz", "")

        record = {
            "trial_id": trial_id,
            "file": filename,
        }

        try:
            data = np.load(path)

            # ------------------------------------------------
            # Check required arrays
            # ------------------------------------------------

            missing = [
                key for key in required_arrays
                if key not in data
            ]

            if missing:
                problems.append({
                    "trial_id": trial_id,
                    "problem": f"Missing arrays: {missing}"
                })
                continue

            # ------------------------------------------------
            # Basic shapes
            # ------------------------------------------------

            time_arr = data["time"]
            gyro = data["gyro"]
            acc = data["acc"]
            qpos = data["qpos"]
            qvel = data["qvel"]

            record["time_steps"] = len(time_arr)
            record["time_start"] = (
                float(time_arr[0]) if len(time_arr) else np.nan
            )
            record["time_end"] = (
                float(time_arr[-1]) if len(time_arr) else np.nan
            )

            record["gyro_shape"] = str(gyro.shape)
            record["acc_shape"] = str(acc.shape)
            record["qpos_shape"] = str(qpos.shape)
            record["qvel_shape"] = str(qvel.shape)

            # ------------------------------------------------
            # NaN / Inf
            # ------------------------------------------------

            for name, arr in [
                ("time", time_arr),
                ("gyro", gyro),
                ("acc", acc),
                ("qpos", qpos),
                ("qvel", qvel),
            ]:
                record[f"{name}_nan"] = int(np.isnan(arr).sum())
                record[f"{name}_inf"] = int(np.isinf(arr).sum())

            # ------------------------------------------------
            # Sensor magnitudes
            # ------------------------------------------------

            gyro_mag = magnitude(gyro)
            acc_mag = magnitude(acc)

            record["gyro_peak"] = safe_float(
                np.max(gyro_mag)
            )

            record["gyro_mean"] = safe_float(
                np.mean(gyro_mag)
            )

            record["acc_peak"] = safe_float(
                np.max(acc_mag)
            )

            record["acc_mean"] = safe_float(
                np.mean(acc_mag)
            )

            # ------------------------------------------------
            # Attach ground-truth label
            # ------------------------------------------------

            if trial_id in label_map:

                row = label_map[trial_id]

                record["fell"] = bool(row["fell"])
                record["stable"] = bool(row["stable"])

                record["scenario_id"] = row["scenario_id"]
                record["scenario_fn"] = row["scenario_fn"]

                record["magnitude_level"] = row["magnitude_level"]
                record["magnitude_value"] = row["magnitude_value"]

                record["timing_level"] = row["timing_level"]
                record["timing_phase_s"] = row["timing_phase_s"]

                record["direction_deg"] = row["direction_deg"]

                record["fall_direction"] = row["fall_direction"]

                record["time_to_ground_contact"] = (
                    row["time_to_ground_contact"]
                )

                record["peak_pelvis_ang_vel"] = (
                    row["peak_pelvis_ang_vel"]
                )

                record["peak_pelvis_lin_acc"] = (
                    row["peak_pelvis_lin_acc"]
                )

            else:
                problems.append({
                    "trial_id": trial_id,
                    "problem": "No matching labels.csv row"
                })

            records.append(record)

        except Exception as e:

            problems.append({
                "trial_id": trial_id,
                "problem": str(e)
            })

        # Progress
        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(npz_files)} files..."
            )

    summary = pd.DataFrame(records)

    summary.to_csv(
        os.path.join(output_dir, "npz_summary.csv"),
        index=False
    )

    if problems:

        problems_df = pd.DataFrame(problems)

        problems_df.to_csv(
            os.path.join(output_dir, "npz_problems.csv"),
            index=False
        )

        print(
            f"\nWARNING: {len(problems)} problems found."
        )

    else:
        print("\nNo NPZ file problems found.")

    return summary


# ============================================================
# SHAPE ANALYSIS
# ============================================================

def analyze_shapes(summary):

    print("\n============================================================")
    print("ARRAY SHAPES")
    print("============================================================")

    for column in [
        "gyro_shape",
        "acc_shape",
        "qpos_shape",
        "qvel_shape",
    ]:

        if column in summary.columns:

            print(f"\n{column}:")

            print(
                summary[column]
                .value_counts()
                .head(20)
            )


# ============================================================
# EPISODE LENGTH ANALYSIS
# ============================================================

def analyze_episode_lengths(summary, output_dir):

    print("\n============================================================")
    print("EPISODE LENGTH")
    print("============================================================")

    if "time_end" not in summary:
        return

    print(summary["time_end"].describe())

    # Histogram
    plt.figure(figsize=(10, 6))

    plt.hist(
        summary["time_end"].dropna(),
        bins=30
    )

    plt.xlabel("Episode length (s)")
    plt.ylabel("Number of trials")
    plt.title("Episode Length Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "episode_length_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# FALL VS STABLE SENSOR COMPARISON
# ============================================================

def compare_fall_stable(summary, output_dir):

    print("\n============================================================")
    print("FALL VS STABLE SENSOR COMPARISON")
    print("============================================================")

    if "fell" not in summary.columns:
        return

    metrics = [
        "gyro_peak",
        "gyro_mean",
        "acc_peak",
        "acc_mean",
    ]

    available = [
        x for x in metrics
        if x in summary.columns
    ]

    if not available:
        return

    comparison = (
        summary.groupby("fell")[available]
        .agg(["mean", "std", "median", "min", "max"])
    )

    print(comparison)

    comparison.to_csv(
        os.path.join(
            output_dir,
            "fall_vs_stable_sensor_statistics.csv"
        )
    )

    # --------------------------------------------------------
    # Box plots
    # --------------------------------------------------------

    for metric in available:

        fall_values = summary.loc[
            summary["fell"] == True,
            metric
        ].dropna()

        stable_values = summary.loc[
            summary["fell"] == False,
            metric
        ].dropna()

        plt.figure(figsize=(8, 6))

        plt.boxplot(
            [stable_values, fall_values],
            label=["Stable", "Fall"]
        )

        plt.ylabel(metric)
        plt.title(f"{metric}: Stable vs Fall")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{metric}_stable_vs_fall.png"
            ),
            dpi=150
        )

        plt.close()


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

def scenario_analysis(summary, output_dir):

    print("\n============================================================")
    print("SCENARIO ANALYSIS")
    print("============================================================")

    if "scenario_fn" not in summary.columns:
        return

    scenario = (
        summary.groupby("scenario_fn")
        .agg(
            trials=("trial_id", "count"),
            falls=("fell", "sum"),
            mean_gyro_peak=("gyro_peak", "mean"),
            max_gyro_peak=("gyro_peak", "max"),
            mean_acc_peak=("acc_peak", "mean"),
            max_acc_peak=("acc_peak", "max"),
            mean_episode_length=("time_end", "mean"),
        )
    )

    scenario["fall_rate"] = (
        scenario["falls"] / scenario["trials"]
    )

    scenario = scenario.sort_values(
        "fall_rate",
        ascending=False
    )

    print(scenario)

    scenario.to_csv(
        os.path.join(
            output_dir,
            "scenario_sensor_analysis.csv"
        )
    )

    # Fall rate chart

    plt.figure(figsize=(12, 6))

    plt.bar(
        scenario.index,
        scenario["fall_rate"] * 100
    )

    plt.ylabel("Fall rate (%)")
    plt.xlabel("Scenario")
    plt.title("Fall Rate by Scenario")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "fall_rate_by_scenario.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# MAGNITUDE ANALYSIS
# ============================================================

def magnitude_analysis(summary, output_dir):

    print("\n============================================================")
    print("MAGNITUDE ANALYSIS")
    print("============================================================")

    if "magnitude_level" not in summary.columns:
        return

    valid = summary[
        summary["magnitude_level"] >= 0
    ]

    if len(valid) == 0:
        print(
            "Magnitude levels are not populated in the labels."
        )
        return

    mag = (
        valid.groupby("magnitude_level")
        .agg(
            trials=("trial_id", "count"),
            falls=("fell", "sum"),
            mean_gyro_peak=("gyro_peak", "mean"),
            mean_acc_peak=("acc_peak", "mean"),
        )
    )

    mag["fall_rate"] = (
        mag["falls"] / mag["trials"]
    )

    print(mag)

    mag.to_csv(
        os.path.join(
            output_dir,
            "magnitude_analysis.csv"
        )
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        mag.index,
        mag["fall_rate"] * 100,
        marker="o"
    )

    plt.xlabel("Magnitude level")
    plt.ylabel("Fall rate (%)")
    plt.title("Fall Rate vs Magnitude Level")

    plt.xticks(
        mag.index
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "fall_rate_vs_magnitude.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# TIMING ANALYSIS
# ============================================================

def timing_analysis(summary, output_dir):

    print("\n============================================================")
    print("TIMING ANALYSIS")
    print("============================================================")

    if "timing_level" not in summary.columns:
        return

    valid = summary[
        summary["timing_level"] >= 0
    ]

    if len(valid) == 0:
        print(
            "Timing levels are not populated in the labels."
        )
        return

    timing = (
        valid.groupby("timing_level")
        .agg(
            trials=("trial_id", "count"),
            falls=("fell", "sum"),
            mean_gyro_peak=("gyro_peak", "mean"),
            mean_acc_peak=("acc_peak", "mean"),
        )
    )

    timing["fall_rate"] = (
        timing["falls"] / timing["trials"]
    )

    print(timing)

    timing.to_csv(
        os.path.join(
            output_dir,
            "timing_analysis.csv"
        )
    )


# ============================================================
# SAMPLE SENSOR PLOTS
# ============================================================

def plot_sample_trials(
    dataset_dir,
    summary,
    output_dir,
    number_of_trials=6
):

    print("\n============================================================")
    print("SAMPLE SENSOR PLOTS")
    print("============================================================")

    if len(summary) == 0:
        return

    # Prefer a mixture of fall and stable trials
    selected = []

    falls = summary[
        summary["fell"] == True
    ]

    stable = summary[
        summary["fell"] == False
    ]

    selected.extend(
        falls.head(number_of_trials // 2)["trial_id"].tolist()
    )

    selected.extend(
        stable.head(number_of_trials // 2)["trial_id"].tolist()
    )

    for trial_id in selected:

        path = os.path.join(
            dataset_dir,
            trial_id + ".npz"
        )

        if not os.path.exists(path):
            continue

        data = np.load(path)

        time_arr = data["time"]
        gyro = data["gyro"]
        acc = data["acc"]

        # ----------------------------------------------------
        # Gyroscope
        # ----------------------------------------------------

        plt.figure(figsize=(10, 6))

        if gyro.ndim == 2 and gyro.shape[1] >= 3:

            plt.plot(
                time_arr,
                gyro[:, 0],
                label="gyro_x"
            )

            plt.plot(
                time_arr,
                gyro[:, 1],
                label="gyro_y"
            )

            plt.plot(
                time_arr,
                gyro[:, 2],
                label="gyro_z"
            )

        else:

            plt.plot(
                time_arr,
                gyro
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Angular velocity")
        plt.title(f"Gyroscope - {trial_id}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{trial_id}_gyro.png"
            ),
            dpi=150
        )

        plt.close()

        # ----------------------------------------------------
        # Accelerometer
        # ----------------------------------------------------

        plt.figure(figsize=(10, 6))

        if acc.ndim == 2 and acc.shape[1] >= 3:

            plt.plot(
                time_arr,
                acc[:, 0],
                label="acc_x"
            )

            plt.plot(
                time_arr,
                acc[:, 1],
                label="acc_y"
            )

            plt.plot(
                time_arr,
                acc[:, 2],
                label="acc_z"
            )

        else:

            plt.plot(
                time_arr,
                acc
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration")
        plt.title(f"Accelerometer - {trial_id}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{trial_id}_acc.png"
            ),
            dpi=150
        )

        plt.close()


# ============================================================
# GLOBAL SENSOR STATISTICS
# ============================================================

def global_sensor_statistics(dataset_dir, summary, output_dir):

    print("\n============================================================")
    print("GLOBAL SENSOR STATISTICS")
    print("============================================================")

    all_gyro = []
    all_acc = []

    npz_files = glob.glob(
        os.path.join(dataset_dir, "*.npz")
    )

    for path in npz_files:

        try:

            data = np.load(path)

            gyro = data["gyro"]
            acc = data["acc"]

            if gyro.size:
                all_gyro.append(
                    gyro.reshape(-1)
                )

            if acc.size:
                all_acc.append(
                    acc.reshape(-1)
                )

        except Exception:
            continue

    if all_gyro:

        gyro_values = np.concatenate(all_gyro)

        gyro_stats = describe_array(
            gyro_values
        )

        print("\nGyroscope:")
        print(gyro_stats)

        pd.DataFrame(
            [gyro_stats]
        ).to_csv(
            os.path.join(
                output_dir,
                "global_gyro_statistics.csv"
            ),
            index=False
        )

    if all_acc:

        acc_values = np.concatenate(all_acc)

        acc_stats = describe_array(
            acc_values
        )

        print("\nAccelerometer:")
        print(acc_stats)

        pd.DataFrame(
            [acc_stats]
        ).to_csv(
            os.path.join(
                output_dir,
                "global_acc_statistics.csv"
            ),
            index=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="EDA for MuJoCo G1 fall-detection NPZ dataset"
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_DIR,
        help="Path to dataset folder"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where EDA results are saved"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print("============================================================")
    print("        MuJoCo FALL DATASET - EDA")
    print("============================================================")

    print(f"Dataset: {args.dataset}")
    print(f"Output : {args.output}")

    # --------------------------------------------------------
    # 1. Load labels
    # --------------------------------------------------------

    labels = load_labels(
        args.dataset
    )

    # --------------------------------------------------------
    # 2. Label EDA
    # --------------------------------------------------------

    analyze_labels(
        labels,
        args.output
    )

    # --------------------------------------------------------
    # 3. Scan every NPZ
    # --------------------------------------------------------

    summary = scan_npz_files(
        args.dataset,
        labels,
        args.output
    )

    # --------------------------------------------------------
    # 4. Shape analysis
    # --------------------------------------------------------

    analyze_shapes(
        summary
    )

    # --------------------------------------------------------
    # 5. Episode lengths
    # --------------------------------------------------------

    analyze_episode_lengths(
        summary,
        args.output
    )

    # --------------------------------------------------------
    # 6. Fall vs stable
    # --------------------------------------------------------

    compare_fall_stable(
        summary,
        args.output
    )

    # --------------------------------------------------------
    # 7. Scenario analysis
    # --------------------------------------------------------

    scenario_analysis(
        summary,
        args.output
    )

    # --------------------------------------------------------
    # 8. Magnitude analysis
    # --------------------------------------------------------

    magnitude_analysis(
        summary,
        args.output
    )

    # --------------------------------------------------------
    # 9. Timing analysis
    # --------------------------------------------------------

    timing_analysis(
        summary,
        args.output
    )

    # --------------------------------------------------------
    # 10. Global sensor statistics
    # --------------------------------------------------------

    global_sensor_statistics(
        args.dataset,
        summary,
        args.output
    )

    # --------------------------------------------------------
    # 11. Sample plots
    # --------------------------------------------------------

    plot_sample_trials(
        args.dataset,
        summary,
        args.output
    )

    print("\n============================================================")
    print("EDA COMPLETE")
    print("============================================================")

    print(
        f"\nResults saved to:\n{args.output}"
    )


if __name__ == "__main__":
    main()