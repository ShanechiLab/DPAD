# Release tutorial:
# https://packaging.python.org/tutorials/packaging-projects/
# cd source
# python setup.py sdist bdist_wheel
# python -m twine upload --repository testpypi dist/*
# pip install -i https://test.pypi.org/simple/ DPAD-omidsani --upgrade
# python -m twine upload --repository pypi dist/*

import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DPAD",
    version="1.0.0",
    author="Omid Sani",
    author_email="omidsani@gmail.com",
    description="Python implementation for DPAD (dissociative prioritized analysis of dynamics)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShanechiLab/DPAD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
