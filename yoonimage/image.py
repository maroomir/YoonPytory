import os.path
from typing import List

import cv2
import numpy
from numpy import ndarray


class Image:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    width: int
    height: int
    channel: int
    __buffer: numpy.ndarray  # height, width, channel(bpp)
    """

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, PLANE : {2}".format(self.width, self.height, self.channel)

    def __init__(self, width=640, height=480, channel=1, path=None):
        self.width, self.height, self.channel = 0, 0, 0
        self._path: str = ''
        self._buffer: ndarray
        if isinstance(path, str):
            self.path = path
        else:
            self.buffer = numpy.zeros((height, width, channel), dtype=numpy.uint8)

    def clone(self):
        new_image = Image(self.width, self.height, self.channel, self.path)
        new_image._buffer = numpy.copy(self._buffer)
        return new_image

    @property
    def buffer(self) -> ndarray:
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: numpy.ndarray):
        self._buffer = buffer
        # Expand dimension when contains only the height and width
        if len(self._buffer.shape) < 3:
            self._buffer = numpy.expand_dims(self._buffer, axis=-1)
        self.height, self.width, self.channel = self._buffer.shape

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, file_path: str):
        self._path = file_path
        self.buffer = cv2.cv2.imread(file_path)

    @property
    def parents(self) -> List[str]:
        return os.path.split(self._path)[0].split(os.sep)

    @property
    def name(self) -> str:
        f_name = os.path.basename(self._path)
        return os.path.splitext(f_name)[0]

    @property
    def tensor(self) -> ndarray:
        # Change the transform to (Channel, Height, Width)
        return self._buffer.transpose((2, 0, 1)).astype(numpy.float32)

    @tensor.setter
    def tensor(self, tensor: numpy.ndarray):
        # Change the transform and de-normalization
        self.buffer = tensor.transpose((1, 2, 0))

    def minmax_normalize(self) -> dict:
        min_value, max_value = numpy.min(self._buffer), numpy.max(self._buffer)
        result = self._buffer.astype(numpy.float32)
        result = (result - min_value) / (max_value - min_value)
        return {'min': min_value,
                'max': max_value,
                'buffer': result}

    def z_normalize(self) -> dict:
        mean = numpy.mean(self._buffer[self._buffer > 0])
        std = numpy.std(self._buffer[self._buffer > 0])
        result = self._buffer.astype(numpy.float32)
        result = (result - mean) / std
        return {'mean': mean,
                'std': std,
                'buffer': result}

    def normalize(self, mean=128, std=255, channel=None) -> ndarray:
        if channel is None:
            return self._normalize_all(mean, std)
        result = self._buffer
        result[:, :, channel] = (result[:, :, channel] - mean) / std
        return result

    def denormalize(self, mean=128, std=255, channel=None) -> ndarray:
        if channel is None:
            return self._denormalize_all(mean, std)
        result = self._buffer
        result[:, :, channel] = result[:, :, channel] * std + mean
        return result

    def _normalize_all(self, mean=128, std=255) -> ndarray:
        return (self._buffer - mean) / std

    def _denormalize_all(self, mean=128, std=255) -> ndarray:
        return self._buffer * std + mean

    def binarize(self, thresh=128) -> ndarray:
        return cv2.cv2.threshold(self._buffer, thresh, 255, cv2.cv2.THRESH_BINARY)[1]

    def gray_scale(self) -> ndarray:
        return cv2.cv2.cvtColor(self._buffer, cv2.cv2.COLOR_BGR2GRAY)

    def colorize(self) -> ndarray:
        return cv2.cv2.cvtColor(self._buffer, cv2.cv2.COLOR_GRAY2BGR)

    def flip_horizontal(self) -> ndarray:
        return numpy.flipud(self._buffer)

    def flip_vertical(self) -> ndarray:
        return numpy.fliplr(self._buffer)

    @classmethod
    def _parse_rect(cls, rect: (dict, tuple, list)):
        if isinstance(rect, dict):
            return rect['x'], rect['y'], rect['width'], rect['height']
        else:
            return rect[0], rect[1], rect[2], rect[3]

    @classmethod
    def _parse_line(cls, line: (dict, tuple, list)):
        if isinstance(line, dict):
            start_x, start_y = line['start']
            end_x, end_y = line['end']
        elif len(line) < 4:
            start_x, start_y = line[0]
            end_x, end_y = line[1]
        else:
            start_x, start_y, end_x, end_y = line[0], line[1], line[2], line[3]
        return start_x, start_y, end_x, end_y

    def crop(self, rect: (dict, tuple, list)) -> ndarray:
        x, y, width, height = self._parse_rect(rect)
        return self._buffer[int(y - height / 2): int(y - height / 2), int(x - width / 2): int(x + width / 2)]

    def scale(self, scale_x: (int, float), scale_y: (int, float)) -> ndarray:
        return cv2.cv2.resize(self._buffer, None, fx=scale_x, fy=scale_y)

    def resize(self, width: int, height: int, padding: int = None) -> ndarray:
        if not isinstance(padding, int):
            return cv2.cv2.resize(self._buffer, dsize=(width, height), interpolation=cv2.cv2.INTER_CUBIC)
        new_width = int(self.width * min(width / self.width, height / self.height))
        new_height = int(self.height * min(width / self.width, height / self.height))
        result = cv2.cv2.resize(self._buffer, dsize=(new_width, new_height), interpolation=cv2.cv2.INTER_CUBIC)
        new_top = (self.height - new_height) // 2
        new_left = (self.width - new_width) // 2
        canvas = numpy.full((height, width, self.channel), padding)
        canvas[new_top:new_top + new_height, new_left:new_left + new_width, :] = result
        return canvas

    def draw_line(self,
                  line: (dict, tuple, list),
                  color: numpy.ndarray,
                  pen: int = 3) -> None:
        if self.channel == 1:
            self.buffer = self.colorize()
        start_x, start_y, end_x, end_y = self._parse_line(line)
        if line is not None:
            cv2.cv2.line(self._buffer,
                         pt1=(start_x, start_y),
                         pt2=(end_x, end_y),
                         color=color,
                         thickness=pen)

    def draw_rectangle(self,
                       rect: (dict, tuple, list),
                       color: numpy.ndarray,
                       pen: int = 3):
        if self.channel == 1:
            self.buffer = self.colorize()
        x, y, width, height = self._parse_rect(rect)
        top_left, bottom_right = (x - width / 2, y - height / 2), (x + width / 2, y + height / 2)
        if rect is not None:
            cv2.cv2.rectangle(self._buffer, pt1=top_left, pt2=bottom_right, color=color, thickness=pen)

    def draw_text(self,
                  text: str,
                  pos: tuple,
                  color: numpy.ndarray,
                  scale: int = 3):
        if self.channel == 1:
            self.buffer = self.colorize()
        if pos is not None:
            cv2.cv2.putText(self._buffer, text=text, org=pos, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=scale,
                            color=color, thickness=3)

    def show(self):
        cv2.imshow(f"Image : {self._path}", self._buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def template_match(source: ndarray,
                   template: ndarray,
                   score: float = 0.7,
                   method=cv2.TM_CCOEFF_NORMED) -> dict:
    match_container = cv2.matchTemplate(image=source, templ=template, method=method)
    min_, max_, min_pos, max_pos = cv2.minMaxLoc(match_container)
    temp_height, temp_width = template.shape[0], template.shape[1]
    if method in [cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED] and max_ > score:
        return {'pos': max_pos, 'size': (temp_width, temp_height), 'score': max_}
    elif method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and min_ < score:
        return {'pos': max_pos, 'size': (temp_width, temp_height), 'score': min_}


def line_detect(source: ndarray,
                thresh1: int,
                thresh2: int,
                thresh_hough: int = 150,
                max_count: int = 30,
                trace=False) -> list:
    canny_mask = cv2.Canny(source,
                           threshold1=thresh1,
                           threshold2=thresh2,
                           apertureSize=3,
                           L2gradient=True)
    if trace:
        debug_image = Image()
        debug_image.buffer = canny_mask
        debug_image.show()
    features = cv2.HoughLines(canny_mask,
                              rho=1,
                              theta=numpy.pi / 180,
                              threshold=thresh_hough,
                              min_theta=0,
                              max_theta=numpy.pi, )
    res = []
    for i, line in enumerate(features):
        distance = line[0][0]  # Distance as the zero position
        theta = line[0][1]  # Angle of the perpendicular line
        theta = 1e-10 if theta < 1e-10 else theta
        # Intersection position with perpendicular line
        x0, y0 = distance * numpy.cos(theta), distance * numpy.sin(theta)
        height = canny_mask.shape[0]
        width = canny_mask.shape[1]
        x1 = int(x0 - width * numpy.sin(theta))
        y1 = int(y0 + height * numpy.cos(theta))
        x2 = int(x0 + width * numpy.sin(theta))
        y2 = int(y0 - height * numpy.cos(theta))
        res += [{'start': (x1, y1), 'end': (x2, y2)}]
        if i >= max_count:
            break
    return res


def blob_detect(source: ndarray,
                thresh: int,
                max_count: int = 30,
                trace=False):
    thres_image = cv2.threshold(source,
                                thresh=thresh,
                                maxval=255,
                                type=cv2.THRESH_BINARY)
    detector = cv2.SimpleBlobDetector()
    features = detector.detect(thres_image)
    if trace:
        result_image = Image()
        result_image.buffer = thres_image
        result_image.show()
    res = []
    for i, blob in enumerate(features):
        pos = (int(blob.pt[0]), int(blob.pt[1]))
        blob_height, blob_width = blob.size[0], blob.size[1]
        res += [{'pos': pos, 'size': (blob_width, blob_height)}]
        if i >= max_count:
            break
    return res


def sift(source: ndarray,
         num_feature=500,
         octaves: int = 3,
         contrast_thresh=0.4,
         edge_thresh=10,
         sigma=2.0,
         output=True,
         trace=False):
    sift_ = cv2.SIFT_create(nfeatures=num_feature, nOctaveLayers=octaves, contrastThreshold=contrast_thresh,
                            edgeThreshold=edge_thresh, sigma=sigma)
    key_points, desc = sift_.detectAndCompute(source, None)
    if trace:
        canvas = cv2.drawKeypoints(source, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SIFT", canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if output:
        res = []
        for keypoint in key_points:
            pos = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            kp_height, kp_width = keypoint.size[0], keypoint.size[1]
            res += [{'pos': pos, 'size': (kp_width, kp_height)}]
        return res
    else:
        return key_points, desc


def surf(source: ndarray,
         metric_thresh=1000,
         octaves: int = 3,
         output=True,
         trace=False):
    surf_ = cv2.xfeatures2d.SURF_create(metric_thresh, octaves)
    key_points, desc = surf_.detectAndCompute(source, None)
    if trace:
        canvas = cv2.drawKeypoints(source, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SURF", canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if output:
        res = []
        for keypoint in key_points:
            pos = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            kp_height, kp_width = keypoint.size[0], keypoint.size[1]
            res += [{'pos': pos, 'size': (kp_width, kp_height)}]
        return res
    else:
        return key_points, desc


def feature_detect(source: ndarray, match_func="sift", **kwargs):
    if match_func == "sift":
        features = 500 if kwargs["features"] is None else int(kwargs["features"])
        octaves = 3 if kwargs["octaves"] is None else int(kwargs["octaves"])
        contrast_thresh = 0.4 if kwargs["contrast_thresh"] is None else float(kwargs["contrastThresh"])
        edge_thresh = 10 if kwargs["edge_thresh"] is None else int(kwargs["edgeThresh"])
        sigma = 2.0 if kwargs["sigma"] is None else float(kwargs["sigma"])
        kp, desc = sift(source, features, octaves, contrast_thresh, edge_thresh, sigma, output=False)
    elif match_func == "surf":
        metric_thresh = 1000 if kwargs["metric_thresh"] is None else int(kwargs["metricThresh"])
        octaves = 3 if kwargs["octaves"] is None else int(kwargs["octaves"])
        kp, desc = surf(source, metric_thresh, octaves, output=False)
    else:
        raise Exception("The match function is not abnormal")
    return kp, desc


def feature_match(source: ndarray, other: ndarray, match_func="sift", **kwargs):
    features1, description1 = feature_detect(source, match_func, **kwargs)
    features2, description2 = feature_detect(other, match_func, **kwargs)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.knnMatch(description1, description2, 2)
    ratio = 0.5 if kwargs["ratio"] is None else kwargs["ratio"]
    best_matches = [first for first, second in matches if first.distance < second.distance * ratio]
    best_features1 = [features1[match.queryIdx].pt for match in best_matches]
    best_features2 = [features2[match.queryIdx].pt for match in best_matches]
    trace = False if kwargs["trace"] is None else kwargs["trace"]
    if trace:
        canvas = cv2.drawMatches(source, features1, other, features2, best_matches, None,
                                 flags=cv2.cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Feature Match", canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return best_features1, best_features2


def perspective_transform(source: ndarray, other: ndarray, points: list = None, match_func="sift", **kwargs):
    features1, description1 = feature_detect(source, match_func, **kwargs)
    features2, description2 = feature_detect(other, match_func, **kwargs)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.knnMatch(description1, description2, 2)
    ratio = 0.5 if kwargs["ratio"] is None else kwargs["ratio"]
    best_matches = [first for first, second in matches if first.distance < second.distance * ratio]
    src_features = numpy.float32([features1[match.queryIdx].pt for match in best_matches])
    dst_features = numpy.float32([features2[match.queryIdx].pt for match in best_matches])
    transfer, mask = cv2.findHomography(src_features, dst_features, cv2.RANSAC, 5.0)
    trace = False if kwargs["trace"] is None else kwargs["trace"]
    if trace:
        top_left, top_right = [0, 0], [source.shape[1] - 1, 0]
        bottom_left, bottom_right = [0, source.shape[0] - 1], [source.shape[1] - 1, source.shape[0] - 1]
        src_rect = numpy.float32([[top_left], [bottom_left], [bottom_right], [top_right]])
        dst_rect = cv2.perspectiveTransform(src_rect, transfer)
        destination = cv2.polylines(other, [numpy.int32(dst_rect)], True, 255, 3, cv2.LINE_AA)
        canvas = cv2.drawMatches(source, src_features, destination, dst_features, best_matches, None,
                                 flags=cv2.cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Perspective Transform", canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if isinstance(points, list):
        points = numpy.array(points)
        result = cv2.perspectiveTransform(points, transfer)
        return result


def find_epiline(source: Image, other: Image, match_func="sift", **kwargs):
    pass
