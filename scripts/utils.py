import pandas as pd

import importlib

from scripts import preprocessing, feature_selection, model_selection, model, evaluation

def load_data(data_dir: str, sub_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(f'{data_dir}\{sub_dir}\{filename}.csv')

def export_data(data: pd.DataFrame, data_dir: str, sub_dir: str, filename: str):
    data.to_csv(f'{data_dir}\{sub_dir}\{filename}.csv', index=False)

def reload_scripts():
    importlib.reload(preprocessing)
    importlib.reload(feature_selection)
    importlib.reload(model_selection)
    importlib.reload(model)
    importlib.reload(evaluation)