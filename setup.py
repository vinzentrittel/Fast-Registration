"""
Configuration file for the fast_registration package.
"""
from setuptools import setup, find_packages

setup(
    name="fast_registration",
    version="0.1",
    description="Module to provida a quick an possibly dirty registration for STL surface meshes.",
    author="Vinzent Rittel",
    author_email="mail@vinzentrittel.de",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
)
