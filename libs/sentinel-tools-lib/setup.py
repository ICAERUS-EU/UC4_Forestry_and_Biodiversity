from setuptools import setup, find_packages
setup(
    name="sentinellib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "geojson",
        "boto3",
        "geopandas",
        "rasterio",
        "fiona",
        "shapely",
        "aenum",
        "utm",
        "pyproj",
        "earthengine-api",
        "geemap",
        "numba",
        "pystac_client"
    ]
)
# additionally install baselib from libs folder
