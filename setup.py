'''
This script aims to build my entire project as a package.
'''

from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
        This function returns the list of packages in requirements.txt.

        Parameters
        ----------
        @param file_path: The path of requirements.txt file [type: string].

        Return
        ------
        @return requirements: List of strings containing the packages required for the project.
    '''
    requirements = list()
    with open(file_path, encoding='utf-16') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
    name = 'Credit Risk Case',
    version = '0.0.1',
    author = '',
    author_email = '',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)
