from abc import abstractmethod
import numpy as np
from boars.io.utils import Bbox, Coordinate


class Entity:
    """Entity class represents detected boars. When entity class is created it has information about coordinates, bbox when it was detected in image,
    image index.
    When checking other frames the entities are checked whether the new detected entities are the same as already detected, thus the boars are not counted twice.
    When determining if the entity is new, the speed function is checked. Speed function is built in a way that it is faster at first and declines after a few frames.
    """

    def __init__(self, coordinate: Coordinate,
                 bbox: Bbox,
                 probability: float,
                 image: np.ndarray, curr_second: float,
                 frame: int,
                 default_speed: float,
                 ):
        """
        Args:
            coordinate (Tuple[float,float]): geographical location of first detection.
            bbox (Tuple[int,int,int,int]): bouding box location in a photo of a first detection. Described by two points: upper left and lower right corners.
            probability (float): model given probability for this entity.
            image (np.ndarray): image where entity is detected.
            curr_second (float): timestamped second where the entity is detected.
            default_speed (int): default speed value at which the entity could be moving. Speed units are m/given time unit
        """
        self.coordinates = [coordinate]
        self.bboxes = [bbox]
        self.images = [image]
        self.probabilities = [probability]
        self.timestamp_seconds = [curr_second]
        self.frames = [frame]
        self.default_speed = default_speed
        self.correct_index = 0

    @property
    def probability(self) -> float:
        return max(self.probabilities)

    @property
    def bbox(self) -> Bbox:
        return self.bboxes[self.correct_index]

    @property
    def coordinate(self) -> Coordinate:
        return self.coordinates[self.correct_index]

    @property
    def image(self) -> np.ndarray:
        return self.images[self.correct_index]

    @property
    def timestamp(self) -> float:
        return self.timestamp_seconds[self.correct_index]

    @property
    def last_position(self) -> Coordinate:
        return self.coordinates[-1]

    @property
    def last_seen_second(self) -> float:
        return self.timestamp_seconds[-1]

    @property
    def first_position(self) -> Coordinate:
        return self.coordinates[0]

    @property
    def speed(self) -> float:
        return self.default_speed
        # TODO: FIXME
        # if len(self.coordinates) == 1:
        #     return self.default_speed
        # else:
        #     speed = self.distance_to(
        #         self.coordinates[-2]) / (self.timestamp_seconds[-1] - self.timestamp_seconds[-2]) * 1.4
        #     if speed < self.default_speed / 4:
        #         # TODO: better solution for minimum speed
        #         speed = self.default_speed / 4
        #     return speed

    def distance_to(self, coordinate: Coordinate) -> float:
        """Calculates entities distance to point.

        Args:
            coordinate (Coordinate): point to which the distance should be calculated

        Returns:
            float: distance in meters
        """
        return np.linalg.norm(np.array(self.last_position) - np.array(coordinate))

    def mark_as_same(
            self, coordinate: Coordinate,
            bbox: Bbox, probability: float, image: np.ndarray,
            curr_second: float, frame: int = None) -> None:
        """Attaches given coordinates, bbox and other parameters to already detected entity.
        Given coordinate will be considered as the last seen position of given entity.

        Args:
            coordinate (Coordinate): new entity coordinate.
            bbox (Bbox): bbox of this entity in new image.
            probability (float): model given probability for this entity.
            image (np.ndarray): image where this entity is detected.
            curr_second (float): timestamped second where the entity is detected.
        """
        self.coordinates.append(coordinate)
        self.probabilities.append(probability)
        self.bboxes.append(bbox)
        self.timestamp_seconds.append(curr_second)
        self.images.append(image)
        self.frames.append(frame)

    def check_same(self, coordinate: Coordinate,
                   curr_second: float) -> bool:
        """Given the coordinates of another point checks if this entity could've moved to given position.
        Function calculates the distance from last known position to given coordinate and compares it to value of maximum possible travel distance.
        In order for this to work _distance_radius function should be implemented.

        Args:
            coordinate (Coordinate): geo point of new entity.
            curr_second (float): timestamped second where the entity is detected.

        Returns:
            bool: True if this could possibly be same entity.
        """
        distance_max = self._distance_radius(
            curr_second - self.last_seen_second)
        distance_check = self.distance_to(coordinate)
        if distance_max > distance_check:
            return True
        else:
            return False

    @abstractmethod
    def _distance_radius(self, time_diff: int) -> float:
        """Calculates the distance traveled for a given time.
        The bigger the time difference, the less speed entity will have for a time unit. f(x+1) > f(x), but f'(x+1)=<f'(x)

        Args:
            time_diff (int): time difference of which the distance should be calculated

        Returns:
            float: distance in meters.
        """
        raise NotImplementedError


class Boar(Entity):
    def __init__(self, coordinate: Coordinate,
                 bbox: Bbox,
                 probability: float,
                 image: np.ndarray, curr_second: float, frame=None, default_speed=12):
        super().__init__(coordinate, bbox, probability, image, curr_second, frame, default_speed)

    def _distance_radius(self, time_diff: int) -> float:
        return min(reciprocal_sum_distance(self.speed, time_diff), 30) # TODO: FIXME


def linear_distance(speed: float, time: float) -> float:
    """Linear travel distance speed*time

    Args:
        speed (float):
        time (float):

    Returns:
        float: traveled distance
    """
    return speed * time


def linear_distance_limit(speed: float, time: float, limit: float) -> float:
    """Linear travel distance with upper limit min(speed*time,limit)

    Args:
        speed (float):
        time (float):
        limit (float):

    Returns:
        float: traveled distance
    """
    return min(limit, linear_distance(speed, time))


def reciprocal_sum_distance(speed: float, time: float, a: int = 70) -> float:
    """Linear travel distance with time transformated using reciprocal elements sum for every second
    sum^time_x=(a/x+a)*a

    Args:
        speed (float):
        time (float):
        a (int, optional): reciprocal multiplier. The smaller the multiplier, the quicker reciprocal approaches 0. Defaults to 100.

    Returns:
        float: traveled distance
    """
    return speed * (np.reciprocal(np.arange(a, a + time).astype(float)) * a).sum()


def logarithm_lin_limited_distance(
        speed: float, time: float, a: float = 1, b: float = 1.15, h: float = 0, k: float = 0) -> float:
    """Compares logarithmic function to linear one and returns minimum, so the faster portion of logarithmic growth would be negated.
    Args:
        speed (float):
        time (float): must be higher than 1
        a (float, optional): multiplier. Defaults to 1.
        b (float, optional): logarithmic base. Defaults to 1.15.
        h (float, optional): Defaults to 0.
        k (float, optional): Defaults to 0.

    Returns:
        float: traveled distance
    """
    return min(linear_distance(speed, time), logarithm_distance(speed, time, a, b, h, k))


def logarithm_distance(speed: float, time: float, a: float = 1, b: float = 1.15, h: float = 0, k: float = 0) -> float:
    """Linear travel distance with time transformed using logarithmic formula.
    a*log_b(x-h)+k.
    Compares logarithmic function to linear one and returns minimum, so the faster portion of logarithmic growth would be negated.
    Args:
        speed (float):
        time (float): must be higher than 1
        a (float, optional): multiplier. Defaults to 1.
        b (float, optional): logarithmic base. Defaults to 1.15.
        h (float, optional): Defaults to 0.
        k (float, optional): Defaults to 0.

    Returns:
        float: traveled distance
    """
    return speed * (a * (np.log(time - h) / np.log(b)) + k)
