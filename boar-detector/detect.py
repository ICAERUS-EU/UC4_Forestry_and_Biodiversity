from typing import List, Type, Tuple
from pyproj import Transformer
from boars.detector.model.run_yolo import YoloModel
import math
from boars.detector.entity import Entity
from boars.io.utils import Bbox, Coordinate, add_bbox, rotate, GSD_width, GSD_height
from boars.io._readers import AbstractReader
import numpy as np
from enum import Enum


class IndexMode(Enum):
    PROBA = 1  # Highest probability mode
    CENTERED = 2  # Most centered mode
    MIXED = 3  # Something between high probability and centered.
    LAST = 4
    FIRST = 5
    CLUSTER_CENTER = 6


class Counter:
    """Tool used for counting detected entities. Checks the bboxes found by Detector and confirms whether it could be the same entity or not.
    """

    def __init__(
            self, bboxes: List[List[Bbox]],
            probabilities: List[List[float]],
            image_reader: AbstractReader, entity_class: Type[Entity],
            input_crs: int = 4326, readjust_yaw: bool = False, index_mode: IndexMode = IndexMode.CLUSTER_CENTER, ** entity_params):
        """
        Args:
            bboxes (List[List[Bbox]]): list of lists of positive class bboxes. External lists length should be the same as number of analyzed frames. 
                Internal lists may contain multiple bboxes (if there are multiple entities detected in a frame).
            probabilities (List[List[float]]): list of probabilities that correspond to bboxes.
            image_reader (AbstractReader): instance of AbstractReader. Used for reading positional data, metadata and images.
            entity_class (Type[Entity]): type of Entity that will be initialised and counted.
            input_crs (int): Coordinate Reference System of input, default: 4326
            readjust_yaw (bool): Whether to try read just yaw (if GimbalYaw is provided). If true then tries to offset GimbleYaw so that the values are absolute not relative.
            index_mode (IndexMode): How to select correct index number of entity with different entries. default:IndexMode.MIXED
            **entity_params: keyword parameters that should be passed when creating new instances of entity_class.
        """
        self.bboxes = bboxes
        self.probabilities = probabilities
        self.image_reader = image_reader
        self.Entity = entity_class
        self.entity_params = entity_params
        self.index_mode = index_mode
        self.entities = []
        self.viewing_angles = []
        self.input_crs = input_crs
        self.readjust_yaw = readjust_yaw
        self._set_viewing_angles()

    def _set_viewing_angles(self):
        """Sets self.viewing_angles. Uses _find_viewing_angles function if GimbalYaw is not provided.
        If readjust_yaw is set tries to correct GimbalYaw angle to absolute value.
        """
        if 'GimbalYaw' in self.image_reader.positional[0]:
            for pos in self.image_reader.positional:
                self.viewing_angles.append(math.radians(pos['GimbalYaw']))
            if self.readjust_yaw:
                raise NotImplementedError
        else:
            self.viewing_angles = self._find_viewing_angles()

    def _find_viewing_angles(self) -> List[float]:
        """Finds viewing angles from flying directions. Behaves unpredictably when drone is staying in place or rotating.
        """
        viewing_angles = []
        for pos1, pos2 in zip(self.image_reader.positional[:-1], self.image_reader.positional[1:]):
            x1, y1 = pos1['GPSLongitude'], pos1['GPSLatitude']
            x2, y2 = pos2['GPSLongitude'], pos2['GPSLatitude']
            angle = math.atan2(y2 - y1, x2 - x1)
            # rotating by 90 degree clockwise, so the origin would be the y axis (as in facing north)
            viewing_angles.append(angle - math.pi / 2)
        viewing_angles.append(np.mean(viewing_angles[-2:]))  # Last viewing angle is somewhat guessed
        return viewing_angles

    def _find_gsd(self, frame_number: int) -> Tuple[float, float]:
        """Find gsd for a given frame number.

        Args:
            frame_number (int): Frame numbers

        Returns:
            Tuple[float, float]: gsd height and gsd width
        """
        gsd_height = GSD_height(self.image_reader.positional[frame_number]['RelativeAltitude'], {
            **self.image_reader.meta, **self.image_reader.positional[frame_number]})
        gsd_width = GSD_width(
            self.image_reader.positional[frame_number]['RelativeAltitude'], {
                **self.image_reader.meta, **self.image_reader.positional[frame_number]})
        return gsd_height, gsd_width

    @property
    def count(self) -> int:
        return len(self.entities)

    def analyze(self,) -> None:
        """Analyzes given data. Analyzes the given data (bboxes and coordinates).
        Tries to distinguish unique entities one from another.
        After analysis is done the found unique entities are placed in self.entities list.
        """
        last_coordinate = (None, None)
        transformer = Transformer.from_crs(f"epsg:{self.input_crs}", "epsg:3346")
        for frame_number, image in enumerate(self.image_reader):

            image_center = np.array(image.shape)[:2] // 2
            image_bboxes = self.bboxes[frame_number]
            probabilities = self.probabilities[frame_number]
            curr_time = self.image_reader.positional[frame_number].get('timestamp', frame_number) # TODO: FIXME
            coordinate = self.image_reader.positional[frame_number]['GPSLatitude'], self.image_reader.positional[frame_number]['GPSLongitude']
            if (last_coordinate[0] == coordinate[0]) and (last_coordinate[1] == coordinate[1]):
                continue
            else:
                last_coordinate = coordinate
            
            coordinate = transformer.transform(*coordinate)

            for bbox, proba in zip(image_bboxes, probabilities):
                entity_coord = self._locate_entity(coordinate, bbox, image_center, frame_number)
                if self.count == 0:
                    self._add_entity(entity_coord, bbox,
                                     proba, curr_time, image,frame_number)
                else:
                    distance_entities = [entity.distance_to(
                        entity_coord) for entity in self.entities]
                    # TODO: time consideration. Maybe after some time the entities should not even be considered?
                    valid_sorted_entities = [entity for _, entity in sorted(
                        zip(distance_entities, self.entities), key=lambda x:x[0]) if entity.last_seen_second != curr_time]
                    try:
                        entity = valid_sorted_entities[0]
                        entity_exists = self._check_same(
                            entity, entity_coord, curr_time)
                        if entity_exists:
                            entity.mark_as_same(entity_coord, bbox, proba, add_bbox(
                                image, bbox), curr_time,frame=frame_number)
                        else:
                            self._add_entity(entity_coord, bbox,
                                             proba, curr_time, image,frame_number)
                    except IndexError:
                        self._add_entity(entity_coord, bbox,
                                         proba, curr_time, image,frame_number)
        self._confirm_entities()

    def _confirm_entities(self):
        def distance(a, b):
            return np.linalg.norm(a - b)
        for entity in self.entities:
            if self.index_mode == IndexMode.FIRST:
                entity.correct_index = 0
            elif self.index_mode == IndexMode.LAST:
                entity.correct_index = -1
            elif self.index_mode == IndexMode.CLUSTER_CENTER:
                distances=[]
                all_coords = entity.coordinates
                for coords in all_coords:
                    mean_dist = np.mean([np.linalg.norm(np.array(coords) - np.array(c))
                for c in all_coords])
                    distances.append(mean_dist)
                entity.correct_index = np.argmin(distances)
            else:
                a = np.ones(len(entity.probabilities))
                b = np.ones(len(entity.probabilities))
                # These values will be used to balance probabilities and distances
                if self.index_mode == IndexMode.PROBA or self.index_mode == IndexMode.MIXED:
                    a = np.array(entity.probabilities)  # If probabilities should be considered

                elif self.index_mode == IndexMode.CENTERED or self.index_mode == IndexMode.MIXED:
                    image_center = np.array(self.image_reader._read(0).shape)[
                        :2] // 2  # Assumes all images are the same size
                    distance_max = distance(image_center, np.array(self.image_reader._read(0).shape)[:2])
                    distances = np.array([
                        distance(
                            np.array((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                            image_center) for bbox in entity.bboxes])
                    log_base = math.sqrt(distance_max)
                    b = np.array([math.log(max(log_base, dis), log_base) for dis in distances])
                    # If distances should be considered. Values range from 1(center) to 2(corner)
                    # bboxes in the vicinity of the center are evaluated as equal (1)
                entity.correct_index = np.argmax(a / b)

    def _locate_entity(self, coordinate, bbox, image_center, frame_number):
        viewing_angle = self.viewing_angles[frame_number]
        gsd_height, gsd_width = self._find_gsd(frame_number)
        # Entity location in image. Center point
        entity_loc_image = (
            bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        moved_x, moved_y = entity_loc_image[0] - \
            image_center[0], image_center[1] - \
            entity_loc_image[1]  # moved_y is calculated differently so that it would match LKS94 movement

        entity_coord = coordinate[0] + moved_y * \
            gsd_height, coordinate[1] + moved_x * gsd_width  # Coordinates are saved as LatLong

        entity_coord = rotate(coordinate, entity_coord, -viewing_angle)
        return entity_coord

    def _add_entity(self, coordinate: Coordinate, bbox: Bbox, probability: float, curr_time: float, image: np.ndarray, frame_number):
        new_image = add_bbox(image, bbox)
        entity = self.Entity(coordinate, bbox, probability,
                             new_image, curr_time, frame=frame_number, **self.entity_params)
        self.entities.append(entity)

    def _check_same(self, entity: Entity, coordinate: Coordinate, curr_time: float):

        if entity.last_seen_second != curr_time:
            return entity.check_same(coordinate, curr_time)
        else:
            return False


class Detector:
    """Tool used for detecting entities in image. All neccessary data should be provided by instance of AbstractReader.
    This class only detects(runs a model), unique entities are identified in Counter class.
    You can run the detection and identification routines by calling the detect_identify method.
    """

    def __init__(
            self, image_reader: AbstractReader, weights: str, entity_class: Type[Entity] = None, model_kwargs: dict = None,
            entity_kwargs: dict = None):
        """_summary_

        Args:
            image_reader (AbstractReader): concrete instance of AbstractReader.
            weights (str): path to where YoloModel weights are stored
            entity_class (Type[Entity]): type of entity that will be used for counter. Not required if detect_identify will not be called.
            model_kwargs (dict): keyword arguments that will be passed to YoloModel.
            entity_kwargs (dict): keyword arguments that will be passed to Counter, if detect_identify is called.
        """
        self.image_reader = image_reader
        if model_kwargs is None:
            model_kwargs = dict()
        self.model = YoloModel(image_reader, weights, imgsz=max(image_reader._read(0).shape), **model_kwargs)
        self.model._load_model()
        if entity_kwargs is None:
            entity_kwargs = dict()
        self.entity_kwargs = entity_kwargs
        self.Entity = entity_class

    def _init_counter(self, bboxes: List[Bbox]):
        bboxes_ = []
        probabilities = []
        for bboxes_frame in bboxes:
            bboxes_temp = []
            probabilities_temp = []
            for bbox in bboxes_frame:
                bboxes_temp.append(bbox[:4])
                probabilities_temp.append(bbox[4])
            bboxes_.append(bboxes_temp)
            probabilities.append(probabilities_temp)
        self.counter = Counter(
            bboxes_, probabilities, self.image_reader, self.Entity, **self.entity_kwargs)

    def detect(self):
        return self.model.run()

    def detect_identify(self) -> Counter:
        assert self.Entity is not None, "When calling detect_identify entity class is required"
        bboxes = self.detect()
        self._init_counter(bboxes)
        self.counter.analyze()
        # reader.positional  # list of dicts for each frame with gps coords
        
        return self.counter
