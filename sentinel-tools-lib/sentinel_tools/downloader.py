from datetime import datetime
from loguru import logger
from sentinelhub.api.opensearch import get_area_info
from sentinelhub.aws.data import AwsTile, AwsData
from sentinelhub.aws import AwsDownloadClient, AwsTileRequest
from sentinelhub.data_collections import DataCollection
from sentinelhub.exceptions import SHUserWarning, SHDeprecationWarning
from sentinelhub import BBox
from pathlib import Path
from typing import Optional, List
from pprint import pprint
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=SHDeprecationWarning)
warnings.filterwarnings("ignore", category=SHUserWarning)


def download(
        bbox: BBox,
        out_path: Path,
        ts_from: datetime,
        ts_to: datetime,
        maxcc: Optional[float] = None,
        bands: Optional[List[str]] = None
):
    if maxcc is None:
        maxcc = 1
    time_interval = (ts_from, ts_to)

    logger.info(f"Getting area info: bbox [{bbox} with {bbox.crs}] / period [{ts_from} - {ts_to}] / max cc {maxcc}")
    results = get_area_info(bbox, time_interval, maxcc=maxcc)

    results_df = pd.json_normalize(results)
    if len(results_df) == 0:
        logger.error("No results were found with requested parameters")

    results_df["properties.startDate"] = pd.to_datetime(results_df["properties.startDate"])
    results_df["year_month"] = results_df["properties.startDate"].dt.strftime('%Y-%m')

    temp_df = results_df[["year_month", "properties.cloudCover"]]
    min_cc_indices = temp_df.groupby("year_month")["properties.cloudCover"].idxmin()

    print(temp_df.groupby("year_month")["properties.cloudCover"].min())
    print(results_df.loc[min_cc_indices].to_string())

    logger.info(f"{len(results)} images acquired")

    results_df = results_df.loc[min_cc_indices].reset_index()
    for index, row in results_df.iterrows():
        idx = index + 1
        url = row["properties.s3Path"]
        tile, date, aws_index = AwsData.url_to_tile(url)

        logger.info(f"Downloading tile [{idx}/{len(results_df)}]: {tile} / {date} / {aws_index}")
        request = AwsTileRequest(
            tile=tile,
            time=date,
            aws_index=aws_index,
            bands=bands,
            data_folder=out_path,
            data_collection=DataCollection.SENTINEL2_L2A,
        )
        request.save_data()
