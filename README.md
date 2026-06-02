# Forecasting the Recovery of Global Economies from Shocks through Time Series Modeling

Python | Time Series Forecasting | Machine Learning | Deep Learning | Spatial Analysis

This project implements a multi-phase forecasting pipeline designed to analyze historical economic data across 212 countries (1990–2024), model the impact of major economic shocks, and forecast GDP per capita recovery trajectories using classical statistical models, machine learning, and deep learning. The final pipeline compares XGBoost, ARIMA, and LSTM models and extends the analysis with an Iran conflict shock-scenario study.

The system demonstrates how time-series analytics and machine learning can jointly answer the question: *"How fast do economies recover after shocks — and can we predict the direction of recovery?"*

---

## Project Overview

Economic shocks such as the 2008 Global Recession and the 2020 COVID-19 Pandemic caused severe disruptions to GDP, employment, and living standards globally. Policymakers and international institutions need reliable, data-driven forecasts to allocate resources and plan recovery interventions.

This project builds a complete forecasting and analysis system that:

- collects and cleans 6 macroeconomic datasets from World Bank, IMF, and ILO  
- engineers shock labels, recovery speed, recovery quality, and lag features  
- performs spatial analysis, regional comparison, and country clustering  
- trains and evaluates 3 main forecasting models across 212 countries: XGBoost, ARIMA, and LSTM  
- compares models using MAE, RMSE, MAPE, MASE, and Directional Accuracy on the same 2019–2024 test period  
- performs Iran conflict scenario analysis to examine how inflTION WAS EFFECTED  
- identifies which countries recovered fastest and which are still below pre-shock GDP   

---

## Forecasting Pipeline

The automated workflow follows a structured 9-phase data science pipeline:

```
Phase 1 — Data Collection & Cleaning
        ↓
Phase 2 — Feature Engineering & Shock Labeling
        ↓
Phase 3 — Spatial Analysis & Country Clustering
        ↓
Phase 4 — XGBoost Regression Model
        ↓
Phase 5 — ARIMA
        ↓
Phase 6 — LSTM Deep Learning Model
        ↓
Phase 7 — Final Comparison Report
        ↓
Phase 7 — Final Model Comparison Report
        ↓
Phase 8 — Iran Conflict Scenario Analysis
        ↓
Phase 9 — Generalization / Normal-Year Analysis
```

**Train Period:** 1990–2018 &nbsp;|&nbsp; **Test Period:** 2019–2024  
**Coverage:** 212 countries × 35 years × 31 features

---

## Datasets & Data Pipeline

Six macroeconomic datasets were collected and merged to build the master dataset:

| Dataset | Source | Original Coverage | Final Status |
|---|---|---|---|
| GDP per Capita | World Bank | 1960–2024 | ✅ Kept (1990–2024 window) |
| Population | World Bank | 1960–2024 | ✅ Kept |
| Inflation (HCPI) | IMF | 1969–2024 | ✅ Kept |
| Employment Rate | ILO | 1991–2024 | ✅ Kept |
| Age Dependency Ratio | World Bank | 1960–2024 | ✅ Kept |
| Real Interest Rates | World Bank | 1960–2024 | ❌ **Removed** (73% missing) — see Data Cleaning section |

### Data Pipeline & Train/Test Split

```
Raw Data (6 sources) 
         ↓ [Phase 1 Cleaning]
Master Dataset: 7,240 rows × 7 features (zero NaNs)
         ↓ [Phase 2 Feature Engineering]
         ├→ model_training_dataset.csv (14 features, zero leakage)
         │   └→ Used by: Phase 4 (XGBoost), Phase 5 (ARIMA), Phase 6 (LSTM)
         │
         └→ dashboard_dataset.csv (all features + recovery metrics)
             └→ Used by: Phase 3 (Spatial), Streamlit dashboard

Temporal Train/Validation/Test Split:
    Training:    1990–2015 (4,151 rows)  — Learn model
    Validation:  2016–2018 (525 rows)    — Tune hyperparameters
    Test:        2019–2024 (869 rows)    — Final evaluation ← ALL METRICS REPORTED HERE
```

### Data Lineage Diagram

This diagram shows the complete flow from raw data collection through model training to final predictions and outputs:

```mermaid
graph TB
    A["📊 Raw Data Sources<br/>World Bank/IMF/ILO<br/>212 countries × 35 years"] --> B["Phase 1: Data Cleaning<br/>Merge 6 datasets<br/>Handle missing values<br/>7,240 complete rows"]
    
    B --> C["Master Dataset<br/>7 features, zero NaNs<br/>212 countries, 1990–2024"]
    
    C --> D["Phase 2: Feature Engineering<br/>Create 14 features<br/>Add shock labels<br/>Calculate recovery metrics"]
    
    D --> E{{"Split into Two Paths"}}
    
    E -->|Model Training| F["model_training_dataset.csv<br/>14 features, 7,240 rows<br/>Zero data leakage"]
    
    E -->|Dashboard & Analysis| G["dashboard_dataset.csv<br/>26 features<br/>+recovery metrics<br/>+cluster assignments"]
    
    F --> H["⚙️ Model Training<br/>Phase 4–6<br/>1990–2018"]
    
    H --> I1["XGBoost<br/>Regression"]
    H --> I2["ARIMA<br/>Time Series"]
    H --> I3["LSTM<br/>Deep Learning"]
    H --> I4["Baselines<br/>Naive/HistMean"]
    
    G --> J["Phase 3: Spatial Analysis<br/>Clustering & Visualizations<br/>Regional comparison"]
    
    I1 --> K["🔄 Walk-Forward Validation<br/>2019–2024 test set<br/>Generate predictions<br/>Calculate metrics"]
    I2 --> K
    I3 --> K
    I4 --> K
    
    K --> L1["xgb_test_predictions.csv<br/>175 countries<br/>2020–2023"]
    K --> L2["arima_test_predictions.csv<br/>212 countries<br/>2020–2024"]
    K --> L3["lstm_test_predictions.csv<br/>212 countries<br/>2020–2024"]
    K --> L4["model_metrics.csv<br/>MAE/RMSE/MAPE/MASE/DA"]
    
    
    
    L4 --> O["Phase 7: Comparison Report<br/>Metrics comparison<br/>Confusion matrices<br/>Balanced accuracy"]
    L1 --> O
    L2 --> O
    L3 --> O
    L4 --> O
    
    G --> P["🎯 Streamlit Dashboard<br/>Tab 1: Global Map<br/>Tab 2: Country Deep Dive<br/>Tab 3: Model Comparison<br/>Tab 4: Recovery Analysis"]
    
    O --> Q["Phase 8: Iran Conflict Analysis<br/>Scenario-based geopolitical shock study<br/>GDP recovery risk interpretation"]


    L1 --> P
    L2 --> P
    L3 --> P
    L4 --> P
    O --> P
    Q --> P
    J --> P
    
    style A fill:#ffb3b3
    style B fill:#ffd9b3
    style C fill:#fffab3
    style D fill:#ffffb3
    style H fill:#d9ffb3
    style K fill:#b3ffb3
    style Q fill:#b3ffd9
    style O fill:#b3ffff
    style P fill:#d9b3ff
    
    classDef data fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    class F,G,L1,L2,L3,L4
```

**Key Data Flow Points:**

1. **Raw Data** → 6 datasets merged (World Bank GDP, WB Population, IMF Inflation, ILO Employment, WB Dependency, WB Interest Rate)
2. **Phase 1 Cleaning** → Handle 73% missing interest rate data; decide to remove; achieve 7,240 complete rows
3. **Feature Engineering** → Create 14 predictive features + shock labels + recovery metrics (14,726 features before selection)
4. **Split & Train** → Two CSV paths: model_training_dataset (models) + dashboard_dataset (visualizations)
5. 5. **Model Training** → 3 main models trained on 1990–2018: XGBoost, ARIMA, and LSTM
6. **Predictions** → Generate predictions for the 2019–2024 test period
7. **Comparison** → Phase 7 computes MAE, RMSE, MAPE, MASE, and Directional Accuracy for fair model comparison
8. **Iran Conflict Analysis** → Scenario-based analysis is added to examine the possible inflation impact of a geopolitical shock
9. **Dashboard** → Model results, recovery metrics, spatial analysis, and scenario outputs feed into the Streamlit dashboard



### 14 Model Training Features

These 14 features were selected to maximize predictive power while eliminating data leakage:

| # | Feature | Purpose | Data Type |
|---|---|---|---|
| 1–2 | `country_code`, `year` | Identifiers | Categorical/Integer |
| 3 | `gdp_per_capita` | Target output (next year GDP) | Float |
| 4–5 | `gdp_lag_1`, `gdp_lag_2` | Temporal dependencies (1–2 years prior) | Float |
| 6 | `gdp_growth_rate` | Current year growth momentum | Float |
| 7 | `growth_lag_1` | Prior year growth momentum | Float |
| 8 | `gdp_rolling_mean_3yr` | 3-year trend smoothing | Float |
| 9–11 | `inflation_hcpi`, `employment_rate`, `dependency_ratio` | Macro drivers (monetary, labor, demographic) | Float |
| 12 | `is_shock` | Binary shock period flag (1=2008/2020, 0=normal) | Binary |
| 13–14 | `world_gdp_growth`, `pct_countries_contracting` | Global macroeconomic context | Float |

**Why this selection?**
- **Excluded:** `interest_rate` (73% missing), `recovery_speed/quality` (computed from test data = leakage), rolling std dev (noise)
- **Included only:** Features available at prediction time (no future data)
- **Result:** 7,240 complete observations with zero data leakage

---

## Forecasting Models

The system trains and benchmarks three main forecasting models per country using the **14 features** listed above:

- **XGBoost Regression** — Gradient boosted decision trees; best model for GDP magnitude accuracy
- **ARIMA** — Classical statistical time-series model used as a country-level forecasting benchmark
- **LSTM (2-layer, 128→64 neurons)** — Deep learning sequence model with a 5-year sliding window; best model for directional accuracy
- **Baseline models** — Naive and historical mean models used only as reference benchmarks

**All models trained on:** 1990–2018 (4,676 rows)  
**All models evaluated on:** 2019–2024 (869 rows) ← **Same test set for fair comparison**

---

## Model Evaluation Metrics

Evaluated on held-out test set (2019–2024):

- **MAE** — Mean Absolute Error ($ USD, lower is better)
- **RMSE** — Root Mean Squared Error (penalizes large errors)
- **MAPE** — Mean Absolute Percentage Error (%)
- **MASE** — Mean Absolute Scaled Error (vs naive baseline)
- **DA** — Directional Accuracy (% correctly predicting up/down movement)

---

## Model Results (Test Set 2019–2024)

| Model | MAE | RMSE | MAPE | MASE | DA |
|---|---|---|---|---|---|
| **XGBoost** | **$1,359** | 3,508 | **5.26%** | **0.89** | 60.17% |
| ARIMA | $1,945 | 5,884 | 7.61% | 2.71 | 59.81% |
| LSTM | $1,539 | **3,378** | 12.54% | 2.15 | **69.06%** |

**Interpretation:**
**Interpretation:**
- **XGBoost performs best for magnitude accuracy:** It has the lowest MAE, MAPE, and MASE, meaning it predicts GDP values closest to the actual values.
- **LSTM performs best for directional accuracy:** It better predicts whether GDP per capita increases or decreases year by year.
- **ARIMA provides a classical statistical benchmark:** It performs reasonably but has higher error compared with XGBoost and LSTM.
- **Ensemble and Holt ES were removed from the final comparison** to keep the model evaluation simpler, cleaner, and consistent across the same 2019–2024 test period.
## Iran Conflict Scenario Analysis

An additional scenario analysis was performed to examine how a potential Iran conflict or geopolitical shock could affect economic recovery patterns. This analysis was not treated as a separate forecasting model. Instead, it was used as an external shock study to test how sensitive GDP recovery predictions are under conflict-related uncertainty.

The Iran conflict analysis helps extend the project beyond historical shocks such as the 2008 recession and COVID-19 by showing how the forecasting pipeline can be adapted to future geopolitical risks.

**Purpose of this analysis:**
- examine how geopolitical instability may influence inflation trends
- test the flexibility of the forecasting pipeline on a new external shock scenario


---

## Spatial Analysis & Country Clustering

Phase 3 produces interactive global choropleth maps (Plotly) for:

- GDP per capita trends (1990–2024)  
- Shock impact severity (2008 and 2020)  
- Recovery speed and recovery quality per shock  

Countries are grouped into **4 meaningful clusters** using K-Means on 6 recovery features:

| Feature | Source | Purpose |
|---|---|---|
| `speed_2008`, `quality_2008` | Recovery from 2008 recession | Developed markets shock response |
| `speed_2020`, `quality_2020` | Recovery from COVID-19 | Recent pandemic response |
| `latest_gdp` | 2024 GDP per capita | Economic development level |
| `avg_employment` | 1990–2024 average | Structural labor market health |

### Cluster Profiles

| Cluster | Countries | Avg 2008 Speed | 2008 Quality | Avg GDP | Profile | Name |
|---|---|---|---|---|---|---|
| **Resilient Emerging**  | 85 | 1.3 years (FAST) | 113.3% (OVER) | $20,121 | Fast emerging economies that overrecovered | 
| **Struggling Developed** | | 38 | 6.5 years (SLOW) | 97.2% (UNDER) | $56,395 | Developed economies with slow 2008 recovery |
| **Resilient Low-Income**  | 21 | 2.0 years | 108.8% (OVER) | $12,904 | Low-income economies with decent recovery |
| **outlier** | 1 | 1 year | 200% (HIGH) | 70516.72|  

> **Key Finding:** The 2008 recession created a stark **recovery divide**. Emerging and low-income countries recovered FAST (1–2 years) and OVERRECOVERED (110%+). Developed economies (except Germany, Australia) took 6+ years to recover and many remain below pre-2008 levels even in 2024 (Spain: 93.3%).

---

---

## Data Collection & Cleaning

**Phase 1 Process:** Six World Bank, IMF, and ILO datasets were merged covering 212 countries (1990–2024).

### Data Quality Assessment

| Feature | Coverage | Status | Decision |
|---|---|---|---|
| GDP per capita | 98% | High quality | ✅ Kept, 2% imputed (median by country) |
| Population | 93% | High quality | ✅ Kept, 7% imputed |
| Inflation (HCPI) | 89% | Acceptable | ✅ Kept, 11% imputed (median by country) |
| Employment rate | 88% | Acceptable | ✅ Kept, 12% imputed |
| Dependency ratio | 93% | High quality | ✅ Kept, 7% imputed |
| **Interest rate** | **27%** | **CRITICAL** | ❌ **REMOVED** — only 1,900/7,240 rows had data |

### Why Interest Rate Was Removed

Interest rates had **73% missing values**, creating a choice between two bad options:

| Option | Trade-off |
|---|---|
| **A: Keep & Impute** | Would synthetically generate 5,340 values. Interest rates are country-specific policy variables; median imputation invalid. Model would learn from fake data. |
| **B: Keep & Drop rows** | Would lose 5,340 observations, reducing dataset from 7,240 to 1,900 rows (74% data loss). Many countries would have <10 observations. |
| **C: REMOVE** | Keep all 7,240 real observations with 6 features instead of 7 features with 73% fake data. ✅ **CHOSEN** |

**Decision Rationale:** Better to use 7,240 complete observations with 6 features than 1,900 rows with 7 features. Other economic indicators (employment, inflation, GDP) capture monetary transmission mechanisms sufficiently.

### Final Clean Dataset

| Metric | Value |
|---|---|
| **Rows** | 7,240 (100% complete) |
| **Countries** | 212 |
| **Years** | 1990–2024 (35 years) |
| **Features** | 7 (no missing values in modeling set) |
| **Imputation method** | Median by country for missing values <15% |

---

## Recovery Metrics Definitions

These two metrics are **central to the analysis** and appear throughout dashboards, visualizations, and reports.

### Recovery Speed (Years)

**Definition:** Number of years from a shock year for a country's GDP per capita to return to pre-shock levels.

**Formula:**
```
Recovery Speed = (Year of Recovery) - (Shock Year)

where "Year of Recovery" = first year GDP ≥ pre-shock baseline
```

**Examples:**
- **2008 Recession (baseline = 2007 GDP):**
  - 🇨🇳 China: 1 year (returned to 2007 level in 2009)
  - 🇪🇸 Spain: 9 years (returned to 2007 level in 2016)
  - 🇩🇪 Germany: 3 years (returned to 2007 level in 2010)

- **COVID-19 2020 (baseline = 2019 GDP):**
  - 🇰🇪 Kenya: 1 year (recovered in 2021)
  - 🇮🇹 Italy: 2 years (recovered in 2022)

- **NaN (Not Recovered):** Countries that had not returned to pre-shock GDP by end of 2024 dataset

**Interpretation:** Lower = faster recovery (better). Higher = slower recovery (worse).

---

### Recovery Quality (%)

**Definition:** GDP per capita measured 3 years post-shock, expressed as a percentage of pre-shock baseline.

**Formula:**
```
Recovery Quality (%) = (GDP at year+3 / GDP at year-1) × 100

where year = shock year
```

**Examples:**
- **2008 Recession (measure in 2011 vs 2007 baseline):**
  - 🇨🇳 China: 142.3% (recovered and grew 42% above pre-shock)
  - 🇪🇸 Spain: 93.3% (still 6.7% below pre-shock after 3 years)
  - 🇩🇪 Germany: 105.5% (recovered and grew 5.5% above pre-shock)

- **COVID-19 2020 (measure in 2023 vs 2019 baseline):**
  - 🇰🇪 Kenya: 110.1% (overrecovered by 10%)
  - 🇧🇷 Brazil: 105.9% (overrecovered by 6%)

**Interpretation:**
- **100% = Recovered:** GDP returned to pre-shock level
- **>100% = Overrecovered:** GDP exceeded pre-shock level (economic expansion)
- **<100% = Underrecovered:** GDP still below pre-shock level (incomplete recovery)

---

## Feature Engineering

Key features engineered in Phase 2:

- **GDP lags:** `gdp_lag_1`, `gdp_lag_2` — prior year GDP levels for temporal dependency
- **Growth lags:** `growth_lag_1` — momentum (prior year growth rate)
- **Rolling statistics:** `gdp_rolling_mean_3yr` — 3-year trend smoothing  
- **Shock label:** `is_shock` — binary indicator (1 for 2008, 2020; 0 otherwise)  
- **Recovery metrics:** `recovery_speed_*`, `recovery_quality_*` — **(see definitions above)**
- **Global context:** `world_gdp_growth`, `pct_countries_contracting` — macroeconomic spillover signals
- **Demographics:** `inflation_hcpi`, `employment_rate`, `dependency_ratio` — structural drivers

---

## Datasets

Six macroeconomic datasets were collected and merged to build the master dataset:

| Dataset | Source | Variable |
|---|---|---|
| GDP per Capita | World Bank | `gdp_per_capita` |
| Population | World Bank | `population` |
| Inflation (HCPI) | IMF | `inflation_hcpi` |
| Employment Rate | ILO | `employment_rate` |
| Age Dependency Ratio | World Bank | `dependency_ratio` |
| Real Interest Rates | World Bank | `interest_rate` |

**Master dataset:** `output/master_enriched.csv` — Shape: (7,240 rows × 31 columns)  
**Coverage:** 212 countries | 1990–2024

---

## Repository Structure

```
economies_recover_after_shocks/
│
├── data/                          # Raw source datasets
│   ├── gdp-per-capita-worldbank/
│   ├── employment-to-population-ratio/
│   ├── Inflation-data/
│   ├── age_dependency_ratio/
│   ├── P_Real interest rates/
│   └── API_SP.POP.TOTL_DS2_en_csv_v2_58/
│
├── notebooks/                     # Jupyter notebooks (one per phase)
│   ├── phase1_data_collection_cleaning.ipynb
│   ├── phase2_feature_engineering.ipynb
│   ├── phase3_spatial_analysis.ipynb
│   ├── phase4_xgboost_model.ipynb
│   ├── phase5_sarima.ipynb
│   ├── phase6_lstm.ipynb
│   ├──── phase7_comparison_report.ipynb
│   ├── iran_conflict_analysis.ipynb
│   └── phase10_2023_generalization_test.ipynb
│
├── output/                        # Generated CSVs, charts, and maps
├── scripts/                       # Helper/patch scripts
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Running the Project

**1. Create and activate virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Register the kernel for Jupyter:**
```bash
python -m ipykernel install --user --name=economies_venv
```

**4. Run notebooks in order:**
```bash
jupyter notebook
```
Open and execute the notebooks sequentially from data collection through model comparison, then run the Iran conflict analysis and generalization analysis notebooks.

**Or execute a notebook from the terminal:**
```bash
venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace ^
  --ExecutePreprocessor.timeout=600 ^
  --ExecutePreprocessor.kernel_name=economies_venv ^
  notebooks/phaseN_*.ipynb
```

---

## Key Outputs

| Output File | Description |
|---|---|
| `output/master_enriched.csv` | Final merged dataset with all engineered features |
| `output/model_metrics.csv` | MAE, RMSE, MAPE, MASE, DA for all models |
| `output/lstm_test_predictions.csv` | LSTM predictions per country per year |
| `output/xgb_test_predictions.csv` | XGBoost predictions for the final test period |
| `output/arima_test_predictions.csv` | ARIMA predictions for the final test period |
| `output/map_country_clusters.html` | Interactive cluster map (open in browser) |
| `output/map_recovery_speed_2020.html` | Interactive COVID recovery speed map |
| `output/phase7_metric_comparison.png` | All-model metric bar chart |
| `output/iran_conflict_analysis.csv` | Scenario-based Iran conflict analysis output |
| `output/iran_conflict_results.png` | Visualization of Iran conflict scenario results |

---

## Key Technologies

| Category | Libraries |
|---|---|
| Data Processing | Pandas 2.2.3, NumPy 1.26.4 |
| Machine Learning | Scikit-learn 1.5.2, XGBoost 2.1.3 |
| Deep Learning | TensorFlow 2.18.0, Keras 3.6.0 |
| Statistical Models | Statsmodels 0.14.4 |
| Spatial Analysis | GeoPandas 1.1.3, Plotly 6.7.0 |
| Visualisation | Matplotlib 3.9.2, Seaborn 0.13.2 |
| Notebook Environment | Jupyter, IPyKernel |

---

## Project Outcome

This project demonstrates how a structured multi-model forecasting pipeline can be applied to global macroeconomic data to understand and predict economic recovery from shocks.

Key findings:

- **138 out of 212 countries** recovered to pre-COVID GDP levels within the same year (2020/2021)
- **10 countries** remained below pre-COVID GDP even by 2024
- Recovery speed is driven more by **economic structure** (manufacturing vs tourism) than income level alone
- **XGBoost** is the best model for magnitude accuracy, achieving the lowest MAE, MAPE, and MASE
- **LSTM** is the best model for directional accuracy, achieving **71.06% Directional Accuracy**
- **ARIMA** provides a useful classical benchmark but performs weaker than XGBoost and LSTM on overall error metrics
- The Iran conflict analysis shows that the pipeline can be extended from historical shock analysis to future geopolitical shock scenarios

The pipeline is generalizable and can be extended to other macroeconomic indicators, additional shock events, or real-time data feeds for ongoing monitoring.

---
