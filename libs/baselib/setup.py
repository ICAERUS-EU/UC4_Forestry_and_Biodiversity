from setuptools import setup, find_packages
setup(
    name="baselib",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "colour-science",
    ]
)
