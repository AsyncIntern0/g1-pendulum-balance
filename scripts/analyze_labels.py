"""
analyze_labels.py — quick diagnostic on dataset/labels.csv

Run:
    python analyze_labels.py dataset/labels.csv
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\dataset\labels.csv"
df = pd.read_csv(path)

n_ambiguous = len(df) - df.fell.sum() - df.stable.sum()
print(f"Total trials: {len(df)} | fell: {df.fell.sum()} | stable: {df.stable.sum()} | "
      f"ambiguous (timed out): {n_ambiguous}")
print(f"Overall fall rate: {df.fell.mean()*100:.1f}%")
if n_ambiguous > 0:
    print(f"[WARN] {n_ambiguous} trials never confirmed fell OR stable within "
          f"the time cap — likely partial degradation (e.g. a leg sagging) "
          f"that neither criterion catches. Worth inspecting these directly.\n")
else:
    print()

print("=== Fall rate per scenario ===")
by_scenario = df.groupby(["scenario_id", "scenario_fn"]).fell.agg(["mean", "count"])
by_scenario["fall_pct"] = (by_scenario["mean"] * 100).round(1)
print(by_scenario[["fall_pct", "count"]].to_string())

print("\n=== Fall rate per scenario x magnitude level ===")
# extract magnitude_level back out from magnitude_value rank within each scenario
df["mag_rank"] = df.groupby("scenario_id")["magnitude_value"].rank(method="dense").astype(int) - 1
pivot = df.pivot_table(index="scenario_id", columns="mag_rank", values="fell", aggfunc="mean") * 100
pivot = pivot.round(0)
print(pivot.to_string())

print("\n=== Scenarios needing wider magnitude range (fall rate <20% or >80% at extremes) ===")
for sid, row in pivot.iterrows():
    lo, hi = row.iloc[0], row.iloc[-1]
    if lo > 20 or hi < 80:
        print(f"  scenario {sid}: fall rate at min mag = {lo}%, at max mag = {hi}%  -> "
              f"{'raise max magnitude' if hi < 80 else 'lower min magnitude' if lo > 20 else 'ok'}")
