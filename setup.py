# Script to define contrastive_learning as a package
from distutils.core import setup
from setuptools import find_packages, setup

setup(
    name="see_to_touch",
    packages=find_packages(), # find_packages are not installing any extra packages for now
)