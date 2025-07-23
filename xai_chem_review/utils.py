import os
import pathlib


PATH: str = pathlib.Path(__file__).parent.absolute()
VERSION_PATH: str = os.path.join(PATH, 'VERSION')


def get_version() -> str:
    """
    Returns the version of the package as specified in the VERSION file.
    """
    with open(VERSION_PATH, 'r') as file:
        version_string: str = file.read().strip()
        version_string = version_string.replace('\n', '')
        
        return version_string
    
        