from contextlib import contextmanager
from rasterio.io import MemoryFile


@contextmanager
def mem_raster(data, profile):

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


def _upscaler(data, scale_factor, axis=[1, 2]):
    assert isinstance(scale_factor, int), "Scale factor must be an integer"
    assert scale_factor > 0, "Scale factor must be greater than 0"
    assert max(axis) < len(data.shape), "Selected axis outside matrix range"
    for a in axis:
        data = data.repeat(scale_factor, axis=a)
    return data

