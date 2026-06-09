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


def load_dataset_aqsoldb_heaviest(path: str = os.path.join(PATH, 'datasets', 'aqsoldb_heaviest.csv')) -> pd.DataFrame:
    """The 50 heaviest molecules in AqSolDB by molecular weight (Tutorial 4, Tier 1 / near-OOD).

    These rows are drawn from AqSolDB itself, so they carry a ground-truth `solubility`,
    but they sit in the extreme size tail of the distribution. Columns: `ID`, `smiles`,
    `solubility`, `InChIKey`, `mw`.
    """
    df: pd.DataFrame = pd.read_csv(path)
    return df


def load_dataset_astatine_substituted(path: str = os.path.join(PATH, 'datasets', 'astatine_substituted.csv')) -> pd.DataFrame:
    """Recognizable halogenated AqSolDB molecules with every F/Cl/Br/I replaced by astatine
    (Tutorial 4, Tier 2 / mid-OOD).

    Astatine (Z = 85) is a group-17 halogen, so the substitution is valence-preserving, yet
    no astatine compound appears anywhere in AqSolDB and Z = 85 sits beyond the heaviest
    element seen in training (Bi, Z = 83). Each molecule is therefore an otherwise-familiar
    organic structure that has been pushed out of distribution along a single, purely
    elemental axis. The substituted molecules have no experimental solubility; `parent_smiles`
    and `parent_solubility` record the original AqSolDB molecule for reference only. Columns:
    `name`, `parent_smiles`, `smiles`, `n_substituted`, `parent_solubility`.
    """
    df: pd.DataFrame = pd.read_csv(path)
    return df