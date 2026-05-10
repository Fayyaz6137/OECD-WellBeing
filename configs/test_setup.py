def test_setup():
    print(f'\n{"─" * 55} SETUP TESTING STARTING {"─" * 55}\n')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.feature_selection import mutual_info_regression, RFE
    from sklearn.linear_model import LinearRegression

    print("✓ All imports successful!")
    print(f"pandas {pd.__version__}, sklearn available")
    print(f'\n{"─" * 55} SETUP TESTING END {"─" * 55}\n')
