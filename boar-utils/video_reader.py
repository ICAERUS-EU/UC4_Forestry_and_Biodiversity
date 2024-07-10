from ._readers import AbstractReader
import os
from .utils import video_file_categorization, parse_srt, VideoCutter, get_meta_by_model
from PIL import Image
from numpy import asarray, array


class VideoReader(AbstractReader):
    """Reader to read and parse folder with video files inside.
    """

    def __init__(self, data_root, skip_every=0, camera_model="XT2", **kwargs):  # todo: WIP fix camera model hardcode
        """
        Args:
            data_root (path): path to folder with files
            skip_every (int, optional): Skip number of frames before cutting an image. Defaults to 0.
        """
        self.files = None
        self.file_categories = None
        self.skip_every = skip_every
        self.video_length = None
        self.fps = 0
        self.camera_model = camera_model
        super().__init__(data_root, **kwargs)

    def parse_folder(self):
        """Reads all files (files only) from data_root folder, non recursive.
        """
        self.files = sorted([os.path.join(self.data_root, f) for f in os.listdir(self.data_root)
                             if os.path.isfile(os.path.join(self.data_root, f))])

    def _set_meta(self):
        """Super override, reads file types from folder, generates empty meta
        """
        if self.files is None:
            self.parse_folder()
        self.file_categories = video_file_categorization(self.files)
        if self.camera_model is not None:
            self.meta = get_meta_by_model(self.camera_model.replace("\x00", ""))
        else:
            self.meta = {}

    def cut_video(self):
        """Video cutting wrapper.
        """
        vc = None
        try:  # remove empty strings if any exist
            metas = self.file_categories['metadata'].remove('')
        except ValueError:
            metas = self.file_categories['metadata']
        if metas is None:  # if all items removed, create empty list
            metas = []
        if len(metas) == 1:  # if 1 video and 1 metadata (srt) file
            vc = VideoCutter(self.file_categories['files'][0], self.file_categories['metadata'][0],
                             skip_every=self.skip_every)
            self.positional = vc.parse()
            self.video_length = vc.get_video_length()
        if len(metas) == 0 and len(self.file_categories['files']) > 0:  # >0 video ir 0 metadata
            print("Video file parse error. No metadata files found")
            self.positional = None
        # daugiau nei viena pora video/meta file'u
        if len(metas) == len(self.file_categories['files']) and len(
                self.file_categories['files']) > 1:
            self.positional = []
            for index in range(len(self.file_categories['files'])):
                vc = VideoCutter(self.file_categories['files'][index], self.file_categories['metadata'][index],
                                 skip_every=self.skip_every)
                result = vc.parse()
                if len(self.positional) != 0:
                    last_ts = self.positional[-1]['timestamp']
                    for res in result:
                        res['timestamp'] = last_ts + res['timestamp'] + 20
                self.positional.extend(result)
        if vc is not None:
            self.fps = vc.fps

    def _set_positional(self):
        """Gather video files and filter into categories and cut video.
        """
        if self.files is None:
            self.parse_folder()
        if self.file_categories is None:
            self.file_categories = video_file_categorization(self.files)

        self.cut_video()

    def _read(self, index):
        """Return iteration

        Args:
            index (int): index of data

        Returns:
            list: Image data by index
        """
        return asarray(Image.open(self.positional[index]['Path']))

