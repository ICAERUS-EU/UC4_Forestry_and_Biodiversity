import os
import re
import math
from typing import Literal, Any, Generator, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import imageio
import cv2
from ._exceptions import SrtValidationError

Bbox = Tuple[int, int, int, int]
Coordinate = Tuple[float, float]


class VideoCutter:
    """Class used to cut video into frames and read metadata from file (or OCR WIP)
    """

    def __init__(self, video_path, meta_path, skip_every=0, crop=None, crop_meta=None):
        """
        Args:
            video_path (str): full path to video file for reading
            meta_path (str): full path to video metadata file, None if it doesn't exist
            skip_every (int, optional): set the number of frames to skip before cutting a new image
             (set to 0 to use video fps aka cut every second). Defaults to 0.
            crop (list, optional): a list of indicies to cut an image for saving by [x1, x2, y1, y2]. Defaults to None.
            crop_meta (list, optional): a list of indicies to cut an image for OCR by [x1, x2, y1, y2]. Defaults to None.
        """
        self.video_path = video_path
        self.skip_every = skip_every
        self.meta_path = meta_path
        self.crop = crop
        self.crop_meta = crop_meta
        self.frame_no = 0
        self.current_coord = None
        self.out_dir = os.path.splitext(video_path)[0]
        self.times = None
        self.read_srt_check = True
        self.ocr = None

    def set_ocr_model(self, model):
        """Set the ocr model object to be used in ocr. Requires an implemented read(image) function

        Args:
            model: Ocr model class object.
        """
        self.ocr = model

    def read_srt(self):
        """Read the specified srt metadata file

        Args:
            filepath (str): full path to srt file

        Returns:
            list: list of coordinates from srt file
            list: same length list of timestamp for frame and metadata syncing.
        """
        text = []
        with open(self.meta_path, 'r') as f:
            for line in f:
                text.append(line)
        coords, times = parse_srt(text)
        return coords, times

    def gen_filename(self):
        """generate a filename of cut frame and attach the synced metadata

        Returns:
            str: generated path for current frame
            dict: metadata for current cut frame
        """
        if self.current_coord is not None:
            meta = {"GPSLatitude": float(self.current_coord[0]),
                    "GPSLongitude": float(self.current_coord[1]),
                    "RelativeAltitude": float(self.current_coord[2]),
                    "timestamp": float(self.current_time)}
        else:
            meta = {}
        return os.path.join(
            self.out_dir,
            f'{os.path.basename(self.video_path)}_{self.frame_no}.jpg'), meta

    def get_coords_by_frame(self):
        """sync the current frame with metadata and save the corresponding coordinates to current_coord
        """
        frame_time = round(self.frame_no / self.fps)
        if self.times is None:
            self.current_coord = [0] * 3
            self.current_time = 0
        else:
            indx = np.argmin(np.fabs(self.times - frame_time))
            coord = self.coords[indx]
            self.current_time = self.times[indx]
            self.current_coord = coord  # ['54.8305', '24.5078', '88.3M']

    @staticmethod
    def write_image(filepath, image):
        """save image to specified path

        Args:
            filepath (str): absolute path to save image to
            image (array): numpy or opencv image array for PIL to save
        """
        OpenCVImageAsPIL = Image.fromarray(image)
        OpenCVImageAsPIL.save(filepath, format='JPEG')

    def cropping(self, parameters):
        """Crop the current frame by set parameters

        Args:
            parameters (list): list of indicies to cut by -> [x1, x2, y1, y2]

        Returns:
            array: cut array of current frame
        """
        # input: numpy array and
        return self.curr_frame[parameters[0]:parameters[1], parameters[2]:parameters[3], ...]

    def run_ocr(self):
        """Run the coordinates using the ocr model

        Returns:
            current coords and current time set from image and frame number
        """
        assert self.ocr is not None
        if self.crop_meta is None:
            self.current_coord = self.ocr.read(self.curr_frame)
        else:
            self.current_coord = self.ocr.read(self.cropped_meta)
        self.current_time = self.frame_no / self.fps

    def read_frame(self):
        """read a frame from the video sync coordinates and save the image

        Returns:
            str: path to save image
            dict: dict of metadata
        """
        if self.frame_no >= self.frame_count - 1:
            print('Out of frames')
            return False, False
        try:
            self.curr_frame = self.vid.get_data(self.frame_no)
            self.curr_frame = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
        except IndexError:
            return False, False
        if self.crop is not None:
            self.curr_frame = self.cropping(self.crop)
        if self.crop_meta is not None:
            self.cropped_meta = self.cropping(self.crop_meta)
        if self.read_srt_check:
            self.get_coords_by_frame()
        path, meta = self.gen_filename()
        self.write_image(path, self.curr_frame)
        self.frame_no += self.skip_every
        return path, meta

    def get_video_metadata(self):
        """Collect video metadata

        Returns:
            bool: return true if succesful
        """
        self.vid = imageio.get_reader(self.video_path)
        self.frame_count = self.vid.count_frames()
        self.fps = self.vid.get_meta_data()['fps']
        # self.fps = 25  
        return True

    def get_video_length(self):
        """Return the length of video in seconds

        Returns:
            float: video length seconds
        """
        return self.frame_count / self.fps

    def parse(self):
        """Parse the video file, create images and return metadata

        Returns:
            list: metadata for each image
        """
        # init
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.get_video_metadata()

        if self.skip_every == 0:
            self.skip_every = int(self.fps)

        if '.srt' in self.meta_path or '.SRT' in self.meta_path:  # jei yra srt, tai turi buti path i srt
            self.coords, self.times = self.read_srt()
            self.read_srt_check = True
        else:
            self.coords = None
            self.times = None
            self.read_srt_check = False
        annots = []
        get_frames = True
        while get_frames:
            path, meta = self.read_frame()
            if path is not False:  # else end of file reached
                obj = {}
                obj.update({'Path': path})
                obj.update({
                    'ImageWidth': self.curr_frame.shape[1],
                    'ImageHeight': self.curr_frame.shape[0]
                })
                obj.update(meta)
                annots.append(obj)
            else:
                get_frames = False
        return annots


def add_bbox(image: np.ndarray, bbox: Bbox, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    result = image.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    return result


# video and metadata file extension filters
video_files = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
video_meta_files = ['.srt', '.SRT']
photo_files = ['.jpg', '.JPG', '.png', '.PNG']


def video_file_categorization(input_files):
    """ Parse a list of files and add them to video file or video metadata category.

    Args:
        input_files (list): of list of files (full path or filenames only)

    Returns:
        dict: files categorized into "files" and "metadata"
    """
    videos = []
    meta_files = []
    for in_file in input_files:
        pre, extension = os.path.splitext(in_file)
        if extension in video_files:
            videos.append(in_file)
            for new_ext in video_meta_files:
                check = False
                if pre + new_ext in input_files:
                    check = True
                    meta_files.append(pre + new_ext)
                    break
            if not check:
                meta_files.append('')
    meta = {"files": videos, "metadata": meta_files}
    return meta


def parse_srt(text: list, srt_type=0):
    """Parse .srt metadata files and return list of coordinates (and/or other data) and times (same length).
    Checks for the type of srt file provided. 

    Args:
        text (list os str): text of .srt file read into a list line by line
        srt_type (int, optional): set the type of srt file, to use specific parser according
            to file provide or try to auto detect. Default = 0 (auto detect)

    Returns:
        list of str: coordinate strings
        numpy array: timestamps
    """
    times = []
    coords = []
    if srt_type == 0:
        for i in range(20):
            if "GPS" in text[i]:  # srt form XT2 has GPS keyword
                srt_type = 1
                break
        if srt_type == 0:
            for i in range(20):
                # srt form Mavic has latitude keyword
                if "latitude" in text[i] and 'gb_yaw' in text[i]:
                    srt_type = 3
                    break
        if srt_type == 0:
            for i in range(20):
                if "latitude" in text[i]:  # srt form Mavic has latitude keyword
                    srt_type = 2
                    break
    if srt_type == 0:
        raise SrtValidationError()
        # return coords, np.array(times)
    if srt_type == 1:
        for i, val in enumerate(text):
            if "GPS" in val:
                # input -> GPS(54.8305,24.5078,0.0M) BAROMETER:88.3M
                coord = re.split('[( ) , : \n]', val)
                # output -> ['GPS', '57.1981', '25.4752', '0.0M', '', 'BAROMETER', '80.1M', '']

                # parse value to same output format [lat, lon, alt] float
                coord = [float(coord[1]), float(coord[2]),
                         float(coord[6].replace('M', ''))]

                # 00:00:00,000 --> 00:00:01,000  (2 lines above GPS keyword)
                ts = re.split('[, ]', text[i - 2])
                ts = sum(
                    x * int(t) for x, t in zip([3600, 60, 1], ts[0].split(":"))) + int(ts[1]) / 1000
                times.append(ts)
                coords.append(coord)
    if srt_type == 2:
        for i, val in enumerate(text):
            if "latitude" in val:
                # input -> [latitude: 54.830326] [longitude: 24.512186] [rel_alt: 79.053 abs_alt: 245.877]
                coord = list(filter(None, re.split('[( ) , \n \[ \]]', val)))
                # output -> ['latitude:', '54.830326', 'longitude:', '24.512186', 'rel_alt:', '79.053', 'abs_alt:', '245.877']

                # parse value to same output format [lat, lon, alt] float
                coord = [float(coord[1]), float(coord[3]), float(coord[5])]

                # 00:00:00,000 --> 00:00:01,000  (4 lines above GPS keyword)
                ts = re.split('[, ]', text[i - 4])
                ts = sum(
                    x * int(t) for x, t in zip([3600, 60, 1], ts[0].split(":"))) + int(ts[1]) / 1000
                times.append(ts)
                coords.append(coord)
    if srt_type == 3:
        for i, val in enumerate(text):
            if "latitude" in val:
                # input -> [latitude: 54.830326] [longitude: 24.512186] [rel_alt: 79.053 abs_alt: 245.877]
                val = val.replace(',', '')
                coord = list(filter(None, re.split('[( ) , \n \[ \]]', val)))
                # output -> ['latitude:', '54.830326', 'longitude:', '24.512186', 'rel_alt:', '79.053', 'abs_alt:', '245.877']

                # parse value to same output format [lat, lon, alt] float
                coord = [float(coord[5]), float(coord[7]), float(coord[9])]

                # 00:00:00,000 --> 00:00:01,000  (4 lines above GPS keyword)
                ts = re.split('[, ]', text[i - 3])
                ts = sum(
                    x * int(t) for x, t in zip([3600, 60, 1], ts[0].split(":"))) + int(ts[1]) / 1000
                times.append(ts)
                coords.append(coord)
    return coords, np.array(times)


def convert_to_decimal_degrees(d: int, m: int, s: int, cardinal_direction: str = 'N') -> float:
    """Converts degrees, minutes and seconds to decimal degrees

    Args:
        d (int): degrees
        m (int): minutes
        s (int): seconds
        cardinal_direction (str, optional): Defaults to 'N'.

    Returns:
        float: decimal degrees
    """
    direction = 1
    if cardinal_direction in ['S', 'W']:
        direction = -1
    return float(direction * (d + (m / 60) + (s / 3600)))


def parse_gps_info(gps_info: dict) -> dict:
    """Replaces degrees in minutes to decimal degrees in place.

    Args:
        gps_info (dict): gps_info dictionary. Must include GPSLatitude, GPSLatitudeRef, GPSLongitude and GPSLongitudeRef keys

    Returns:
        dict: dictionary with GPSLatitude and GPSLongitude values converted to decimal degrees.
    """
    dms_lat = gps_info['GPSLatitude']
    dd_lat = convert_to_decimal_degrees(*dms_lat, gps_info['GPSLatitudeRef'])
    gps_info['GPSLatitude'] = dd_lat
    dms_long = gps_info['GPSLongitude']
    dd_long = convert_to_decimal_degrees(
        *dms_long, gps_info['GPSLongitudeRef'])
    gps_info['GPSLongitude'] = dd_long
    return gps_info


def GSD_height(altitude: float, image_meta: dict) -> float:
    """Calculates ground sampling distance height from meta data.

    Args:
        altitute (float): relative altitude in meters.
        image_meta (dict): dictionary with camera metadata. Must include FocalLenght (mm), FocalPlaneYResolution(mm) and ImageHeight.

    Returns:
        float: ground sampling distance height in meters.
    """
    return _GSD(altitude, image_meta, 'Y')


def GSD_width(altitude: float, image_meta: dict) -> float:
    """Calculates ground sampling distance width from meta data.

    Args:
        altitute (float): relative altitude in meters.
        image_meta (dict): dictionary with camera metadata. Must include FocalLenght (mm), FocalPlaneXResolution(mm) and ImageWidth.

    Returns:
        float: ground sampling distance height in meters.
    """
    return _GSD(altitude, image_meta, 'X')


def _GSD(altitude: float, image_meta: dict, dimension: Literal['X', 'Y']) -> float:
    """Calculates ground sampling distance from meta data.

    Args:
        altitute (float): relative altitude in meters.
        image_meta (dict): dictionary with camera metadata and width data. Must include FocalLenght (mm),
            FocalPlaneXResolution(mm)/FocalPlaneYResolution(mm) and ImageWidth/ImageHeight.

        dimension (Literal['X','Y']): if X, calculates GSD width, if Y, calculates GSD height

    Returns:
        float: ground sampling distance in meters
    """
    flight_height = altitude
    if dimension == 'Y':
        dimension_long = 'Height'
    else:
        dimension_long = 'Width'
    focal_length = image_meta['FocalLength'] / 1000
    sensor_dimension = image_meta[f'FocalPlane{dimension}Resolution'] / 1000
    image_dimension = image_meta[f'Image{dimension_long}']
    return (flight_height * sensor_dimension) / (focal_length * image_dimension)


def findkeys(node: Any, kv: str) -> Generator:
    """Returns values by key from a given nested dict. If you want to only get one value, call it using next()
    From https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists.

    Args:
        node (Any): node where to start search
        kv (str): key values, of which dict values to yield

    Yields:
        Generator : value generator.
    """
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x


def get_exif(filename: str) -> dict:
    """Extracts exif data from a given file.

    Args:
        filename (str): filename of a image with exif data.

    Returns:
        dict: exif data dict.
    """
    img = Image.open(filename)
    exif = img._getexif()
    exif_new = dict()
    if exif is not None:

        for key, value in exif.items():
            name = TAGS[key]
            if name == 'GPSInfo':
                value = dict()
                for key_gps in exif[key].keys():
                    name_gps = GPSTAGS[key_gps]
                    value[name_gps] = exif[key][key_gps]

                value = parse_gps_info(value)

            exif_new[name] = value
    for key, val in exif_new['GPSInfo'].items():
        exif_new[key] = val
    exif_new.pop('GPSInfo', None)
    xmp = img.getxmp()
    try:
        flight_height = float(next(findkeys(xmp, 'RelativeAltitude')))
    except StopIteration:
        height_re = re.compile(
            r'RelativeAltitude>(\d*\.\d*).*RelativeAltitude>')
        flight_height = float(
            re.search(height_re, exif_new['XMLPacket'].decode('utf-8')).group(1))
    exif_new['RelativeAltitude'] = flight_height
    exif_new['ImageWidth'] = exif_new['ExifImageWidth']
    exif_new['ImageHeight'] = exif_new['ExifImageHeight']

    return exif_new


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Applies rotation transformation to a given image.

    Args:
        image (np.ndarray): input image
        angle (float): angle of rotation in radians

    Returns:
        np.ndarray: rotated image
    """
    angle = math.degrees(angle)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate(origin: Tuple[float, float], point: Tuple[float, float], angle: float) -> Tuple[float, float]:
    """Rotates a point counterclockwise by a given angle around a given origin.

    Args:
        origin (Tuple[float, float]): X,Y point of origin
        point (Tuple[float, float]): X,Y point which to rotate
        angle (float): angle in radians

    Returns:
        Tuple[float, float]: rotated point
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def get_meta_by_model(meta):
    """Get sensor metadata.
    XT2: https://www.dji.com/lt/zenmuse-xt2/specs
    calculation: https://forum.dji.com/thread-193383-1-1.html

    Args:
        meta (string): camera model XT2 only

    Returns:
        dict: camera sensor metadata
    """
    models = {"XT2": {"FocalLength": 13.0,
                      "FocalPlaneXResolution": 10.88, "FocalPlaneYResolution": 8.704}}
    return models["XT2"]


def turning_1d(data: np.array, symbol: int = 1) -> np.array:
    """
    Find turning points (where it changes direction) in 1d array.
    Args:
        data (np.array): coordinates in shape (n)
        symbol (int): symbol which will indicate where changes happened. Defaults to 1

    Returns:
        np.array: shape n array containing zeroes and +-symbol values where changes in direction occured.
    """
    data = np.array(data)
    data_all = data.copy()
    phases = np.zeros_like(data, dtype=int)
    dy = data[1:] - data[:-1]
    saddle = np.ones(len(data), dtype=bool)
    saddle[:-1] = dy == False

    data = data[np.where(~saddle)]  # eliminate equal neighbouring values

    dy = data[1:] - data[:-1]
    increasing = dy > 0
    decreasing = dy < 0
    saddle = dy == False
    change_increasing = increasing[1:] & decreasing[:-1]
    change_decreasing = decreasing[1:] & increasing[:-1]
    changes = np.zeros_like(dy)

    changes[0] = (1 * increasing[0]) + (-1 * decreasing[0])
    changes[1:][change_increasing] = 1
    changes[1:][change_decreasing] = -1
    changes[saddle] = 0
    changes[0] = 0

    changes_list = list(changes.astype(int) * symbol)
    changes_list.append(0)
    data_list = list(data)

    for idx, c in enumerate(data_all):
        if len(data_list) > 0:
            if c == data_list[0]:
                data_list.pop(0)
                phases[idx] = int(changes_list.pop(0))
    return phases


def turning_points(movement_points: np.array, return_all: bool = False) -> np.array:
    """
    Find turning points (where it changes direction) in 1d array. Changes are encoded in such way that if the same point has two directional changes - you can encode that.
    Args:
        movement_points (np.array): coordinates in shape (n,2)
        return_all (bool): set to True if you want to get individual changes as separate lists. Defaults to False.

    Returns:
        if return_all is set:
            Tuple(np.array,np.array,np.array,np.array,np.array) : returns all changes from every direction (x,x_r, y,y_r -- r means reversed) and all collective changes.
        else:
            np.array: array with non zero values where changes occured in path
    """

    x = movement_points[:, 0]
    x_reversed = x[::-1]
    y = movement_points[:, 1]
    y_reversed = y[::-1]

    x_changes = turning_1d(x, symbol=1)
    x_changes_reversed = turning_1d(x_reversed, symbol=2)[::-1]
    y_changes = turning_1d(y, symbol=4)
    y_changes_reversed = turning_1d(y_reversed, symbol=8)[::-1]
    changes_all = x_changes+x_changes_reversed+y_changes+y_changes_reversed
    if return_all:
        return (x_changes, x_changes_reversed, y_changes, y_changes_reversed, changes_all)
    else:
        return changes_all


def interpolate_points(points: np.array, pairwise_indexes: np.array, drop_neighbours: bool = False) -> np.array:
    """
    Interpolates points by pairwise indexes (start, end)
    """
    indexes = np.arange(len(points))
    points = np.array(points)
    if drop_neighbours:
        neighbours = pairwise_indexes[:, 0]+1 == pairwise_indexes[:, 1]
        neighbours[0] = False
        neighbours[-1] = False
        pairwise_indexes = pairwise_indexes[~neighbours]
    points_new = points.copy()
    for x in pairwise_indexes:
        y = np.array([points[x[0]], points[x[1]]])
        xvals = indexes[x[0]:x[1]]
        yinterp0 = np.interp(xvals, x, y[:, 0])
        yinterp1 = np.interp(xvals, x, y[:, 1])
        points_new[x[0]:x[1], 0] = yinterp0
        points_new[x[0]:x[1], 1] = yinterp1
    return points_new
