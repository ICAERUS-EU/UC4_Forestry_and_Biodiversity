import numpy as np

from sentinel_tools.gis_utils import BoundingBox
from sentinelhub import BBox
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.windows import Window

# https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#units
SENTINEL_RGB_MULTIPLIER = 255 / 10000

__SCL_COLORS_DICT = {
    0: (255, 255, 255, 255),  # Missing data
    1: (255, 0, 0, 255),  # Saturated or defective pixel (Red)
    2: (47, 47, 47, 255),  # Topographic casted shadows (Very dark grey),
    3: (100, 50, 0, 255),  # Cloud shadows (Dark brown)
    4: (0, 160, 0, 255),  # Vegetation (Green)
    5: (255, 230, 90, 255),  # Not-vegetated (Dark yellow)
    6: (0, 0, 255, 255),  # Water (Dark and bright) (Blue)
    7: (128, 128, 128, 255),  # Unclassified (Dark grey)
    8: (192, 192, 192, 255),  # Cloud medium probability (Grey)
    9: (255, 255, 255, 255),  # Cloud high probability (White)
    10: (100, 200, 255, 255),  # Thin cirrus (Very bright blue)
    11: (255, 150, 255, 255)  # Snow or ice (Very bright pink)
}

__UNUSABLE_DATA_LIST = [0, 1, 3, 7, 8, 9, 10]
STACKING_BORDER = 10


def calculate_output_shape(src: rasterio.io.DatasetReader, window: Window, scale_to_res: int):
    upscale_factors = tuple(map(lambda res: res / scale_to_res, src.res))
    return (
        src.count,
        round(upscale_factors[0] * window.height),
        round(upscale_factors[1] * window.width)
    )


def get_bbox(src: rasterio.io.DatasetReader, bbox: BoundingBox, scale_to_res: int):
    window = bbox.to_window(src)

    upscale_factors = tuple(dim / scale_to_res for dim in src.res)
    # upscale_factors = tuple(map(lambda res: res / scale_to_res, src.res))
    output_shape = (
        src.count,
        round(upscale_factors[0] * window.height),
        round(upscale_factors[1] * window.width)
    )

    read_data, *_ = src.read(window=window, out_shape=output_shape, resampling=rasterio.enums.Resampling.nearest)
    return read_data


def get_bbox_with_surroundings(src: rasterio.io.DatasetReader, bbox: BoundingBox, length: int, scale_to_res: int):
    org_window = bbox.to_window(src)
    length = round(scale_to_res / src.res[0] * length)

    org_wh = [org_window.width, org_window.height]
    lengths = org_wh

    mid_point = [lengths[0] / 2 + org_window.col_off, lengths[1] / 2 + org_window.row_off]
    mid_point = list(map(round, mid_point))

    max_length_index = lengths.index(max(lengths))
    other_length_index = (max_length_index + 1) % 2

    lengths[other_length_index] = round(lengths[other_length_index] * (length / lengths[max_length_index]))
    lengths[max_length_index] = length

    offsets = [round(mid_point_coord - (length / 2)) for mid_point_coord, length in zip(mid_point, lengths)]
    col_off, row_off = offsets
    width, height = lengths

    expanded_window = rasterio.windows.Window(col_off, row_off, width, height)
    out_shape = calculate_output_shape(src, expanded_window, scale_to_res)
    read_data, *_ = src.read(window=expanded_window, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest)

    return read_data  # , org_wh


def stack_images(
        images: List[Image.Image], vertical: Optional[bool] = None, horizontal: Optional[bool] = None
) -> Image.Image:
    sizes = [image.size for image in images]
    border_count = len(images) - 1

    if vertical:
        total_width = max([width for width, _ in sizes])
        total_height = sum([height for _, height in sizes]) + border_count * STACKING_BORDER

    elif horizontal:
        total_width = sum([width for width, _ in sizes]) + border_count * STACKING_BORDER
        total_height = max([height for _, height in sizes])

    else:
        raise ValueError("Either 'vertical' or 'horizontal' must be True.")

    new_im = Image.new("RGB", (total_width, total_height))
    offset = 0
    for im in images:
        if vertical:
            pasting_coords = (0, offset)
            _, img_size = im.size

        else:
            pasting_coords = (offset, 0)
            img_size, _ = im.size

        new_im.paste(im, pasting_coords)
        offset += img_size + STACKING_BORDER

    return new_im


def array_to_img(arrays_list: List[np.ndarray], scale_to: Optional[int] = None) -> Image.Image:
    rgb_image = np.dstack(arrays_list)
    rgb_image = (rgb_image * SENTINEL_RGB_MULTIPLIER).astype(np.uint8)

    img = Image.fromarray(rgb_image)
    if scale_to:
        img = scale_img(img, scale_to)

    return img


def scale_img(img: Image.Image, scale_to: int) -> Image.Image:
    dimensions = img.size
    max_dim_index = dimensions.index(max(dimensions))
    other_dim_index = (max_dim_index + 1) % 2

    new_dimensions = [0, 0]
    new_dimensions[other_dim_index] = round(dimensions[other_dim_index] * (scale_to / dimensions[max_dim_index]))
    new_dimensions[max_dim_index] = scale_to

    width, height = new_dimensions
    return img.resize((width, height), resample=Image.NEAREST)


def scl_to_img(scl_arr: np.ndarray, scale_to: Optional[int] = None) -> Image.Image:
    width_scl_arr, height_scl_arr = scl_arr.shape
    output_arr = np.zeros((width_scl_arr, height_scl_arr, 4), dtype=np.uint8)

    for x, y in np.ndindex(scl_arr.shape):
        scl_index = scl_arr[x, y]
        color = __SCL_COLORS_DICT[scl_index]
        output_arr[x, y] = np.array(color)

    img = Image.fromarray(output_arr)
    if scale_to:
        img = scale_img(img, scale_to)

    return img


def contains_unusable_pixels(scl_array: np.ndarray) -> bool:
    return np.any(
        np.isin(scl_array, __UNUSABLE_DATA_LIST)
    )

def get_unusable_pixels_ratio(scl_array: np.ndarray) -> float:
    return np.sum(np.isin(scl_array, __UNUSABLE_DATA_LIST)) / np.prod(scl_array.shape)

