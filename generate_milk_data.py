"""
Simulated RAW-MILK RECEPTION lab export (one row per tanker sample).

WHY SIMULATED
-------------
Real reception-lab data belongs to dairy companies and is not published. This script builds
a dataset that mimics what a plant lab/LIMS exports. All numbers (typical composition,
effect sizes, event dates, adulteration rates) are ILLUSTRATIVE ASSUMPTIONS, not measurements
from a real plant. Replace data/raw_milk_reception.csv with your own anonymised export
(keeping the column names) and the notebook runs unchanged.

PROCESS STORY BUILT IN (so the analysis has something real to find)
-------------------------------------------------------------------
  * summer: warmer arrival temperature, more bacteria, more somatic cells, lower fat (heat stress)
  * Center-03 and Center-09: weak cooling -> warmer milk, high bacterial count
  * Center-05: high somatic cell count (udder-health problem)
  * Center-04: cooling tank FAILS mid-way (from ~day 560): warm milk and high bacterial counts
  * Center-07: starts ADDING WATER (~day 420): lower fat/protein/density, freezing point towards 0
  * Center-11: starts using a NEUTRALIZER (~day 520): lower acidity, higher pH, masked high bacteria
  * a small background rate of both frauds everywhere
  * confirmatory lab results exist only for SOME samples (flagged ones + a random 6%)
    -> any evaluation on the labelled subset is subject to selection bias

DELIBERATE RAW-DATA PROBLEMS
----------------------------
  footer row, duplicated samples, mixed date formats, decimal commas, centre-name variants,
  test-result spelling variants, temperatures typed in Fahrenheit, freezing point typed without
  the minus sign, impossible values (pH 67, fat 355), ~1% missing values.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
N = 2600
start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
n_days = (end - start).days + 1
day_idx = np.sort(rng.integers(0, n_days, N))
date = start + pd.to_timedelta(day_idx, unit="D")
month = date.month
summer = np.isin(month, [6, 7, 8, 9]).astype(float)

centers = np.array([f"Center-{i:02d}" for i in range(1, 13)])
center = rng.choice(centers, N)
cid = np.array([int(c[-2:]) for c in center])

fault4 = (cid == 4) & (day_idx >= 560)
tbc_factor = np.where(np.isin(cid, [3, 9]), 2.6, 1.0) * np.where(fault4, 2.2, 1.0)
scc_factor = np.where(cid == 5, 1.9, 1.0)
cool_offset = np.where(np.isin(cid, [3, 9]), 2.6, 0.0) + np.where(fault4, 3.0, 0.0)

# ---- true (unadulterated) milk -------------------------------------------------
temp = rng.normal(4.6, 1.1, N) + 2.3 * summer + cool_offset
fat = rng.normal(3.55, 0.22, N) - 0.15 * summer
protein = rng.normal(3.15, 0.12, N) + 0.35 * (fat - 3.55) / 0.22 * 0.12 - 0.05 * summer
snf = 8.65 + 0.8 * (protein - 3.15) + rng.normal(0, 0.08, N)
clr = 4 * (snf - 0.2 * fat - 0.14)                      # corrected lactometer reading
density = 1 + clr / 1000
fp = -0.520 + rng.normal(0, 0.004, N)
tbc = rng.lognormal(np.log(120), 0.75, N) * tbc_factor * (1 + 0.7 * summer) * (1 + 0.08 * np.maximum(temp - 5, 0))
scc = rng.lognormal(np.log(230), 0.45, N) * scc_factor * (1 + 0.25 * summer)
acidity = rng.normal(15.5, 0.8, N) + 0.5 * summer + 0.8 * np.clip(np.log10(tbc / 100), -1, 2)
ph = 6.72 - 0.015 * (acidity - 15.5) + rng.normal(0, 0.03, N)
volume = rng.choice([3000, 5000, 8000, 10000, 12000], N, p=[0.15, 0.25, 0.3, 0.2, 0.1])

antibiotic_pos = rng.random(N) < np.where(cid == 6, 0.015, 0.007)

# ---- adulteration events ---------------------------------------------------------
adulterant = np.array(["Genuine"] * N, dtype=object)
p_water = np.where((cid == 7) & (day_idx >= 420), 0.30, 0.008)
p_neut = np.where((cid == 11) & (day_idx >= 520), 0.22, 0.005)
is_water = rng.random(N) < p_water
is_neut = (~is_water) & (rng.random(N) < p_neut)
adulterant[is_water] = "Water"
adulterant[is_neut] = "Neutralizer"

w = np.where(is_water, rng.uniform(4, 18, N), 0.0)       # % water added
dil = 1 - w / 100
fat, protein, snf = fat * dil, protein * dil, snf * dil
clr = 4 * (snf - 0.2 * fat - 0.14)
density = 1 + clr / 1000
fp = fp * dil
acidity = acidity * dil
tbc, scc = tbc * dil, scc * dil

neut_drop = np.where(is_neut, rng.uniform(2.5, 4.0, N), 0.0)
acidity = acidity - neut_drop
ph = ph + np.where(is_neut, neut_drop * 0.085, 0.0)
tbc = tbc * np.where(is_neut, 2.0, 1.0)

alcohol_pos = rng.random(N) < 1 / (1 + np.exp(-(acidity - 19.0) * 1.6))
alcohol_pos = alcohol_pos | (rng.random(N) < 0.004)

df = pd.DataFrame({
    "Sample_ID": [f"MK{50000 + i}" for i in range(N)],
    "Date": date, "Center": center, "Volume_L": volume,
    "Temp_Arrival_C": temp.round(1), "Alcohol_Test": np.where(alcohol_pos, "Positive", "Negative"),
    "Antibiotic_Test": np.where(antibiotic_pos, "Positive", "Negative"),
    "Fat_pct": fat.round(2), "Protein_pct": protein.round(2), "SNF_pct": snf.round(2),
    "Density_g_ml": density.round(4), "Freezing_Point_C": fp.round(3),
    "Acidity_Dornic": acidity.round(1), "pH": ph.round(2),
    "SCC_k": scc.round(0), "TBC_k": tbc.round(0),
})

# ---- confirmatory results exist only for some samples (selection bias!) ----------
suspicious = (fp > -0.507) | (ph > 6.88) | (df["Alcohol_Test"] == "Positive")
tested = (suspicious & (rng.random(N) < 0.9)) | (rng.random(N) < 0.06)
df["Confirmed_Adulterant"] = np.where(tested, adulterant, None)

# ================= inject raw-data problems =======================================
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
idx = rng.choice(N, 40, replace=False)
df.loc[idx, "Date"] = pd.to_datetime(df.loc[idx, "Date"]).dt.strftime("%d/%m/%Y")

for col in ["Fat_pct", "Freezing_Point_C", "Protein_pct", "pH"]:
    df[col] = df[col].astype(object)
    idx = rng.choice(N, 25, replace=False)
    df.loc[idx, col] = [f"{v}".replace(".", ",") for v in df.loc[idx, col]]

idx = rng.choice(N, 30, replace=False)                      # Fahrenheit entries
df["Temp_Arrival_C"] = df["Temp_Arrival_C"].astype(object)
df.loc[idx, "Temp_Arrival_C"] = (df.loc[idx, "Temp_Arrival_C"].astype(float) * 9 / 5 + 32).round(1)

idx = rng.choice(N, 20, replace=False)                      # freezing point without minus sign
df["Freezing_Point_C"] = df["Freezing_Point_C"].astype(object)
for i in idx:
    v = str(df.loc[i, "Freezing_Point_C"]).replace(",", ".")
    df.loc[i, "Freezing_Point_C"] = v.replace("-", "")

bad = rng.choice(N, 15, replace=False)                      # SNF keyed wrongly
df["SNF_pct"] = df["SNF_pct"].astype(float)
df.loc[bad, "SNF_pct"] = (df.loc[bad, "SNF_pct"] + rng.choice([-1, 1], 15) * rng.uniform(0.8, 2.0, 15)).round(2)
bad = rng.choice(N, 4, replace=False); df.loc[bad, "pH"] = "67"
bad = rng.choice(N, 5, replace=False); df.loc[bad, "Fat_pct"] = "355"

cvar = {c: [c.replace("Center-0", "Center-").replace("Center-1", "Center-1"), c.upper() + " ", c.replace("Center-", "Centre ")] for c in centers}
for i in rng.choice(N, 150, replace=False):
    df.loc[i, "Center"] = rng.choice(cvar[df.loc[i, "Center"]])
for col, pos in [("Alcohol_Test", "Positive"), ("Antibiotic_Test", "Positive")]:
    for i in rng.choice(N, 120, replace=False):
        v = df.loc[i, col]
        df.loc[i, col] = rng.choice(["pos", "POS "]) if v == "Positive" else rng.choice(["neg", "NEG ", "Negative "])

for col in ["Fat_pct", "Protein_pct", "SNF_pct", "Density_g_ml", "Freezing_Point_C", "Acidity_Dornic", "pH", "SCC_k", "TBC_k", "Temp_Arrival_C"]:
    m = rng.random(N) < 0.008
    df.loc[m, col] = np.nan

dups = df.sample(30, random_state=3)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=5).reset_index(drop=True)
footer = {c: np.nan for c in df.columns}
footer["Sample_ID"] = "Exported from LIMS (simulated) - replace with your own export"
df = pd.concat([df, pd.DataFrame([footer])], ignore_index=True)

df.to_csv("data/raw_milk_reception.csv", index=False)
print("rows written:", len(df))
print("true adulteration counts:", pd.Series(adulterant).value_counts().to_dict())
print("labelled (confirmed) samples:", int(tested.sum()), "| of which adulterated:", int((tested & (adulterant != 'Genuine')).sum()))
print("water-added % among water samples: mean", round(float(w[is_water].mean()), 1))
print("summer share of TBC>500:", round(float((tbc[summer == 1] > 500).mean()), 3), "| non-summer:", round(float((tbc[summer == 0] > 500).mean()), 3))
print("TBC>500 share by center:", pd.Series(tbc > 500).groupby(center).mean().round(2).to_dict())
