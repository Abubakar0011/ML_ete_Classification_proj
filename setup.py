from setuptools import find_packages, setup
from typing import List

requirements = []


def get_requirements(file_path) -> List[str]:
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    return requirements


setup(
    name='ML classification practice Project',
    version='0.0.1',
    author='Abubakar Saddiq',
    author_email='abubakarsaddiq001199@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')
)
    
