from setuptools import setup, find_packages
setup(
    name="spectral",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "rasterio",
        "pillow",
        "shapely",
        "pyproj",
        "matplotlib",
        "colour-science"
    ]
)
