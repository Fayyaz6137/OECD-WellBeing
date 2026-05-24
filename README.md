
# 🗂 Project Structure
```python
OECD-WellBeing/
│
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
│
├── configs/
│   ├── config_dirs.py
│   ├── config_paths_and_params.py
│   └── config_setup.py
│
├── data/                         
│   ├── processed/                        
│   └── raw/     
│
├── logs/
│
├── src/
│   ├── data/
│   │   ├── fetch_data.py
│   │   ├── pre_process_data.py 
│   │   └── exp_data_analysis.py              
│   │
│   ├── processing/
│   │   ├── feature_engineering.py   
│   │   ├── feature_selection.py 
│   │   └── process.py       
│   │
│   ├── models/
│   │   ├── models_training.py    
│   │   ├── models_testing.py  
│   │   └── models_analysis.py                   
│   │
│   └── final_results.py
│
├── results/
│   ├── models/
│   ├── outputs/
│   ├── plots/
│   └── selected_features.csv
│
└── Reports/
    └── Project Report
```
