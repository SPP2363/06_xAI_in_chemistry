import os
import pandas as pd

from xai_chem_review.utils import PATH


def load_dataset_aqsoldb(path: str = os.path.join(PATH, 'datasets', 'aqsoldb.csv')) -> pd.DataFrame:
    
    df: pd.DataFrame = pd.read_csv(path)
    return df


def load_dataset_mutagenicity(path: str = os.path.join(PATH, 'datasets', 'mutagenicity.csv')) -> pd.DataFrame:
    
    df: pd.DataFrame = pd.read_csv(path)
    return df


def load_dataset_ames_mutagenicity(path: str = os.path.join(PATH, 'datasets', 'ames_mutagenicity_data.csv')) -> pd.DataFrame:

    df: pd.DataFrame = pd.read_csv(path)
    return df


def load_dataset_peptides(path: str = os.path.join(PATH, 'datasets', 'peptides.csv')) -> pd.DataFrame:

    df: pd.DataFrame = pd.read_csv(path)
    return df


def load_dataset_pahs(path: str = os.path.join(PATH, 'datasets', 'pahs.csv')) -> pd.DataFrame:

    df: pd.DataFrame = pd.read_csv(path)
    return df