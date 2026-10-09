# Airline Flight Delay Prediction & Explainable AI (XAI)

**Course:** Machine Learning & Explainable AI (XAI)
**Team #4 — Coach You Team:** Mohammad AlabedAllatif, Omar Alhaymoni, Islam Almshaqpa, Mohammad Jarrah, Raheeq Abughoush, Aya Al Sharifi

**Current stage:** Milestone 2 — Data Understanding & Cleaning

---

## 1. Project Summary

We predict whether a flight will be delayed and, more importantly, **explain why**, using SHAP/LIME so dispatch teams can act on the root cause instead of only seeing a "black box" prediction.

### Why it matters
- About 40% of delay minutes come from late-arriving aircraft (the "domino effect"), so early root-cause detection can stop delays from spreading.
- Delays cost airlines an estimated $60 billion a year, and weather and air-traffic congestion drive a large share of them.
- Most predictive models say *that* a delay will happen but not *why*, which leaves operations teams unable to fix it.

*(Figures are taken from our project proposal.)*

### Planned workflow
1. **Data phase:** parse, clean and structure the historical flight data.
2. **Model development:** train and tune classifiers (e.g. XGBoost, Random Forest), optimising precision and recall.
3. **XAI integration:** add SHAP/LIME to turn predictions into root-cause diagnostics.
4. **Deployment:** interactive dashboard / API for dispatch teams.

---

## 2. Dataset

**Source:** Kaggle — [Airline Delay Causes](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses) (`DelayedFlights.csv`, US flights, 2008, delayed flights only)

| Property | Value |
|---|---|
| Records | ~1.93 million (1,936,758) |
| Columns | 30, including an `Unnamed: 0` index column |
| Size | ~226 MB CSV |

Main feature groups: date/time fields, scheduled vs. actual times, carrier and flight ID, route (`Origin`, `Dest`, `Distance`), elapsed/air time, taxi times, cancellation/diversion flags, and delay causes (`CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`).

> **The dataset is not stored in this repository** because of its size. The notebook downloads it automatically with `kagglehub`.

> The file contains **delayed flights only**, so delay rates computed from it are not national on-time rates.

---

## 3. Milestone 2 — Data Understanding & Cleaning

Notebook: [`milestone_2.ipynb`](milestone_2.ipynb)

The notebook covers:
- Dataset overview and data dictionary
- Missing-value analysis (what is missing, how much, and why)
- Duplicate analysis (exact duplicates and same-flight-key duplicates)
- Summary statistics before and after cleaning
- Documented cleaning decisions (each with a `# Decision / # Reason` comment)
- Outlier investigation
- A before/after comparison table and a list of remaining issues

### Key cleaning decisions
| Topic | Decision | Reason |
|---|---|---|
| Index column | Drop `Unnamed: 0` | Leftover export index, not data; it would hide duplicates |
| Duplicates | Drop exact duplicate rows | The same flight would be counted twice |
| Data types | Flags to `int8`, codes to `category`, nullable `Int64` for times and delays | Less memory, keeps NaN without turning integers into floats |
| Date | Add `FlightDate` | Easier time-based analysis in later milestones |
| Delay-cause blanks | Fill with 0 for flights that operated and were not delayed 15+ min | Causes are only recorded for delays of 15+ minutes, so blank means no reportable delay |
| Cancelled / diverted flights | Keep rows, leave times as NaN | The flight never completed; imputing would invent data |
| Impossible values | Drop rows such as `Distance <= 0` or `AirTime > ActualElapsedTime` | These cannot be real |
| Outliers | Keep, add `ExtremeDelay` flag (> 180 min late) | Long delays are real events; deleting them would hide the truth |

Exact before/after counts are in the notebook output.

---

## 4. How to Run

```bash
pip install pandas numpy kagglehub jupyter
jupyter notebook milestone_2.ipynb
```

Kaggle may ask for credentials the first time `kagglehub` downloads the dataset. Run all cells in order. The notebook also writes a cleaned file (`flights_2008_clean.csv`) locally; it is not committed.

---

## 5. Repository Structure

```
.
├── README.md
└── milestone_2.ipynb
```

---

## 6. Project Roadmap

| Milestone | Status |
|---|---|
| 1 — Project proposal | Done |
| 2 — Data understanding & cleaning | In progress / this repo |
| 3+ — Visualization, feature preparation, modeling, XAI, dashboard | Planned |

### Future work (from proposal)
- Real-time weather and live flight-data API integration
- Interactive XAI dashboard (Streamlit / Gradio) with SHAP explanations
- Network-wide domino-effect modeling with Graph Neural Networks (airports as nodes, routes as edges)
