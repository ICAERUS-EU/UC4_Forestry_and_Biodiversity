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
        "colour-science",
        "baselib @ git+https://oauth:ByQhV9x8qW_vxb5VBfLU@gitlab.art21.lt/ai-projektai/baselib.git#egg=baselib"
    ]
)
