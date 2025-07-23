import os
import pandas as pd

from xai_chem_review.utils import PATH


def load_dataset_aqsoldb(path: str = os.path.join(PATH, 'datasets', 'aqsoldb.csv')) -> pd.DataFrame:
    
    df: pd.DataFrame = pd.read_csv(path)
    return df