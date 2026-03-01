import os


# Sentinel 2
S2_PROCESSING_LEVEL = "S2MSI2A"
S2_ALL_BANDS = ['R10m/B08', 'R10m/B04', 'R10m/B02', 'R20m/B05', 'R10m/B03', 'R20m/B06', 'R20m/B07', 'R20m/B8A', 'R20m/B11', 'R20m/B12', 'R20m/B01', 'R60m/B09', 'R20m/SCL']
S2_ALL_BANDS_SORTED = ['R60m/B01', 'R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B07', 'R10m/B08', 'R20m/B8A', 'R60m/B09', 'R20m/B11', 'R20m/B12', 'R20m/SCL']
S2_RGB_BANDS = ["R10m/B04", "R10m/B03", "R10m/B02"]
S2_INDEX_BANDS = {'SCL': ['R20m/SCL'], 'MSAVI': ['R10m/B04', 'R20m/B8A'], 'LAI': ['R10m/B04', 'R10m/B08'], 'LAI3': ['R10m/B03', 'R10m/B02'],
                'NDVI': ['R10m/B04', 'R10m/B08'], 'EVI': ['R10m/B02', 'R10m/B04', 'R10m/B08'], 'MCARI':['R10m/B03', 'R10m/B04', 'R20m/B05'],
                'SOC': ['R20m/B05', 'R20m/B06', 'R20m/B11', 'R20m/B12', 'R10m/B02', 'R10m/B03', 'R10m/B04', 'R10m/B08'],
                'LAI2': ['R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B11', 'R20m/B12','R20m/B07', 'R20m/B8A']}
S2_CLOUD_BANDS = ["R60m/SCL"]
S2_S3_URL = "sentinel-s2-l2a"
S2_OPENSEARCH_URL = "Sentinel2/search.json?"
S2_EE_SEARCH_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_STAC_SEARCH_COLLECTION = "sentinel-2-l2a"
S2_BAND_KEYS = {"B01": "R60m/B01",
                "B02": "R10m/B02",
                "B03": "R10m/B03",
                "B04": "R10m/B04",
                "B05": "R20m/B05",
                "B06": "R20m/B06",
                "B07": "R20m/B07",
                "B08": "R10m/B08",
                "B8A": "R20m/B8A",
                "B09": "R60m/B09",
                "B11": "R20m/B11",
                "B12": "R20m/B12",
                "SCL": "R20m/SCL",
                "zenith": "zenith",
                "azimuth": "azimuth"}
S2_GENERATED_BANDS = ["zenith", "azimuth"]

S2_BANDS_TO_STAC_BANDS = {  'R10m/B08':"B08_10m",
                            'R10m/B04':"B04_10m",
                            'R10m/B02':"B02_10m",
                            'R20m/B05':"B05_20m",
                            'R10m/B03':"B03_10m",
                            'R20m/B06':"B06_20m",
                            'R20m/B07':"B07_20m",
                            'R20m/B8A':"B8A_20m",
                            'R20m/B11':"B11_20m",
                            'R20m/B12':"B12_20m",
                            'R20m/B01':"B01_20m",
                            'R60m/B09':"B09_60m",
                            'R20m/SCL':"SCL_20m"}

S2_ALL_BANDS_SORTED_EE = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
S2_INDEX_BANDS_EE = {'SCL': ['SCL'], 'MSAVI': ['B4', 'B8A'], 'LAI': ['B4', 'B8'], 'LAI3': ['B3', 'B2'],
                'NDVI': ['B4', 'B8'], 'EVI': ['B2', 'B4', 'B8'], 'MCARI':['B3', 'B4', 'B5'],
                'SOC': ['B5', 'B6', 'B11', 'B12', 'B2', 'B3', 'B4', 'B8'],
                'LAI2': ['B2', 'B3', 'B4', 'B5', 'B6', 'B11', 'B12','B7', 'B8A']}

# Sentinel 1
S1_BANDS = ["VV", "VH"]
S1_INSTRUMENTS = ["IW", "EW"]
S1_S3_URL = "sentinel-s1-l1c"
S1_OPENSEARCH_URL = "Sentinel1/search.json?"
S1_EE_SEARCH_COLLECTION = "COPERNICUS/S1_GRD"
S1_STAC_SEARCH_COLLECTION = "sentinel-1-grd"
S1_BANDS_TO_STAC_BANDS = {"VV": "vv",
                          "VH": "vh"}


OPENSEARCH_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%Z"
OPENSEARCH_BASE_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/"  # full url https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json?

# eodata
AWS_EODATA_PROFILE = "sentinel_tools"
EODATA_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1"
EODATA_BUCKET = "eodata"
