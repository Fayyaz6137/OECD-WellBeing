
# 🗂 Project Structure
**Note: logs**, **results** and **data/processed** will be generated when the program is run.
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
│   └── raw/     
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
├── results/
│   ├── models/
│   ├── outputs/
│   ├── plots/
│   └── selected_features.csv
│
└── Reports/
    └── Project Report
```
