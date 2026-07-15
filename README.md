# OECD Well-Being: Regional Gaps in Life Satisfaction

A data science pipeline that models regional life satisfaction across OECD countries using subnational (TL2) data, uncovering how income, employment, health, education, and social support drive well-being and how unevenly that well-being is distributed *within* countries.

Even in wealthy nations, prosperity isn't spread evenly across regions. This project builds a full, reproducible ML pipeline from raw OECD data to trained models to inequality analysis to quantify those hidden regional gaps.

## Key Results

- **Random Forest achieved a test R² of ~0.70** predicting regional life satisfaction from 13 socioeconomic and environmental indicators.
- **Disposable income and social support** emerged as the strongest predictors of life satisfaction.
- A **Gini coefficient analysis** of within-country regional life satisfaction shows internal well-being inequality is substantially higher in countries such as **Mexico, Colombia, and Turkey** compared to Northern European nations.
- National averages significantly obscure intra-national disparities a country can rank well overall while masking large regional gaps.

## Data

- **Source:** [OECD Regional Well-Being Database](https://stats.oecd.org/FileView2.aspx?IDFile=4eb67b25-b4f2-4e2a-b37b-28a9a1a1f4d3) `Indicator_Last` sheet, TL2 subnational level.
- **Indicators:** disposable income, employment/unemployment rate, homicide rate, life expectancy, education, housing (rooms per capita), mortality rate, voter turnout, broadband access, air quality (PM2.5), internet speed, and perceived social support.
- **Target variable:** Life Satisfaction (0–10 self-assessed score).
- Raw data ships with the repo at `data/raw/OECD-Regional-Well-Being-Data-File.xlsx`.

## Pipeline

The pipeline runs end-to-end from a single entry point (`main.py`) and is organized into clear phases:

1. **Setup** clean/create working directories, verify environment and imports.
2. **Data retrieval** (`src/data/fetch_data.py`) load the raw OECD Excel sheet.
3. **Preprocessing** (`src/data/pre_process_data.py`) drop metadata rows/columns, rename to readable labels, coerce to numeric, handle OECD placeholders, check missingness.
4. **Exploratory Data Analysis** (`src/data/exp_data_analysis.py`) distribution plots, missingness map, correlation matrix.
5. **Feature engineering** (`src/processing/feature_engineering.py`) 80/20 train-test split (pre-imputation to avoid leakage), missingness indicator flags, median imputation, min-max scaling.
6. **Feature selection** (`src/processing/feature_selection.py`) Random Forest importance, mutual information, Recursive Feature Elimination (RFE); keeps features at or above median RF importance.
7. **Model training** (`src/models/models_training.py`) trains Random Forest, Ridge Regression, and Gradient Boosting; saves models and train-set metrics.
8. **Model testing** (`src/models/models_testing.py`) Ridge hyperparameter tuning via `GridSearchCV`; evaluates all models on the held-out test set.
9. **Model analysis** (`src/models/models_analysis.py`) train vs. test R² comparison, actual-vs-predicted plots for the best model.
10. **Final results** (`src/final_results.py`) Gini coefficient of regional life satisfaction per country, regional boxplots for top countries by region count.

Every step logs progress to a timestamped file under `logs/`.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · openpyxl

## Project Structure

```python
OECD-WellBeing/
│
├── README.md
├── requirements.txt
├── main.py                                 #---------------- Starting Point ----------------#
├── .gitignore
│
├── configs/                                #---------------- Phase 0 ----------------#
│   ├── config_dirs.py
│   ├── config_paths_and_params.py
│   └── config_setup.py
│
├── data/                         
│   ├── processed/                        
│   └── raw/                                #---------------- Raw Data File/s ----------------#
│
├── logs/
│
├── src/
│   ├── data/                               #---------------- Phase 1 ----------------#
│   │   ├── fetch_data.py
│   │   ├── pre_process_data.py 
│   │   └── exp_data_analysis.py              
│   │
│   ├── processing/                         #---------------- Phase 2 ----------------#
│   │   ├── feature_engineering.py   
│   │   ├── feature_selection.py 
│   │   └── process.py       
│   │
│   ├── models/                             #---------------- Phase 3 ----------------#
│   │   ├── models_training.py    
│   │   ├── models_testing.py  
│   │   └── models_analysis.py                   
│   │
│   └── final_results.py                    #---------------- Phase 4 ----------------#
│
├── results/                                #---------------- Results ----------------#
│   ├── models/
│   ├── outputs/
│   ├── plots/
│   └── selected_features.csv
│
└── Reports/
    └── Project Report                       #---------------- Report ----------------#
```

## Getting Started

```bash
git clone https://github.com/Fayyaz6137/OECD-WellBeing.git
cd OECD-WellBeing
pip install -r requirements.txt
python main.py
```

Running `main.py` will regenerate `data/processed/`, `results/` (models, metrics, plots), and `logs/` from scratch.

## Outputs

After a full run, `results/` contains:

- `models/` pickled trained models (Random Forest, Ridge, Gradient Boosting)
- `outputs/` `train_metrics.csv`, `test_metrics.csv`, `gini_by_country.csv`
- `plots/` distributions, missingness, correlation matrix, feature importance, mutual information, train-vs-test R², actual-vs-predicted, Gini by country, regional boxplots

## Report

The full write-up research questions, methodology, results, and conclusions is available at [`reports/DS-Lab-Report-946912-947854-948445.pdf`](reports/DS-Lab-Report-946912-947854-948445.pdf).

This project was completed as a Data Science Lab project at Università degli Studi di Milano-Bicocca, supervised by Prof. Marco Fattore and Silvio Gerli.

## License

No license file is currently included. All rights reserved by the repository owner unless a license is added.
