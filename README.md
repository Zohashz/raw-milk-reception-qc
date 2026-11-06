# 🥛 Raw Milk Reception QC — Adulteration Screening, Centre Scorecards & Gate-Risk Model (Python)

A dairy-QC analysis workflow for the **milk reception lab**: from a raw LIMS-style export to adulteration screening (added water, neutralizer), collection-centre scorecards, a control chart, and a model that ranks incoming tankers by risk **before** lab results are available.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

👉 **Open [`raw_milk_reception_qc.ipynb`](raw_milk_reception_qc.ipynb)** — outputs and charts are saved in the notebook, so GitHub renders the full analysis.

## 💡 In plain words

Every tanker of raw milk is tested before it is unloaded. Some milk is simply poor (warm, bacteria-rich), and some is **adulterated** — for example diluted with water or "treated" with a neutralizer to hide souring. This project shows how a QC team can:

1. **clean** messy lab exports (unit mix-ups, lost minus signs, duplicates),
2. **screen** for added water and neutralizer using the physics/chemistry of milk plus an anomaly-detection model,
3. **monitor** collection centres (heat-maps, control chart, tiered scorecard),
4. **prioritise** incoming tankers using only information available at the gate.

## 📊 Key results (simulated data)

| Finding | Result |
|---|---|
| Cleaning | 2,631 raw rows → 2,600 samples; 30 duplicates, 30 temperatures converted °F→°C, 20 freezing points with restored minus sign |
| Physical consistency (SNF vs density and fat) | 15 samples flagged for re-test |
| Added-water rule (freezing point) on confirmed samples | precision 1.00, recall 0.97 |
| Neutralizer rule (pH vs acidity) on confirmed samples | precision 1.00, recall 1.00 |
| Isolation Forest (no labels needed) | precision 0.96, recall 0.95 |
| Centre scorecard | 6 centres need corrective action, 2 on watch, 4 approved |
| Gate-risk model (time-ordered test) | PR-AUC 0.53 vs 0.25 for a useless model; priority attention to the top 20% of tankers anticipates 44% of lab rejections |

*Results describe a simulated process. Screening scores are optimistic because the simulated fraud is strong (4–18% water) and the confirmed-sample subset is biased — both caveats are explained in the notebook.*

## ⚠️ About the data

Real reception-lab data is confidential, so the notebook runs on a **simulated export** (`generate_milk_data.py`). Compositions, limits, event dates and adulteration rates are **illustrative assumptions, not measurements from a plant**; rejection rates are deliberately high so every analysis has enough events. The file contains typical raw-data problems on purpose.

**To use real data:** save an anonymised export as `data/raw_milk_reception.csv` with the same column names, review the `SPEC` block in the setup cell, and *Restart & Run All*. Never publish confidential plant data.

## 📖 Glossary
- **Freezing point** – milk freezes at about −0.52 °C; added water pushes it towards 0 °C.
- **SNF** – solids-not-fat; **CLR** – corrected lactometer reading (from density).
- **Dornic degree (°D)** – unit of milk acidity (1 °D ≈ 0.1 g lactic acid per litre).
- **TBC / SCC** – total bacterial count / somatic cell count (hygiene and udder health).
- **Neutralizer** – alkaline substance added to hide acidity from souring.
- **Isolation Forest** – anomaly-detection model that needs no labels.
- **p-chart** – control chart for a proportion (here the monthly rejection rate).
- **Selection bias** – confirmatory tests were done mostly on suspicious samples, which distorts accuracy figures.

## Repository layout
```
├── raw_milk_reception_qc.ipynb   # the analysis (outputs included)
├── generate_milk_data.py         # builds the simulated export
├── requirements.txt
├── .gitignore
└── data/raw_milk_reception.csv   # raw input (simulated; replace with your own)
```

## Run it (VS Code)
```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Open the notebook, pick the `.venv` kernel and click **Run All**.

## Author
**Zoha Shirazi** — Food Tech | QC (dairy production and food coloring manufacturing)
