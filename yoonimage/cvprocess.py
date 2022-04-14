import cv2
import cv2.cv2
from cv2 import cv2
import numpy

from yoonimage import Image, Object, Dataset1D
from yoonpytory.figure import Vector2D, Rect2D, Line2D


def template_match(source: Image, template: Image,
                   score: float = 0.7, mode: int = cv2.TM_SQDIFF_NORMED):
    assert isinstance(template, Image)
    match_container = cv2.matchTemplate(image=source.get_buffer(),
                                        templ=template.get_buffer(),
                                        method=mode)
    min_, max_, min_pos, max_pos = cv2.minMaxLoc(match_container)
    if max_ > score:
        min_pos = Vector2D.from_array(min_pos)
        max_pos = Vector2D.from_array(max_pos)
        center_pos = (min_pos + max_pos) / 2
        rect = Rect2D(x=center_pos.x, y=center_pos.y, width=template.width, height=template.height)
        return Object(region=rect, image=source.crop(rect), score=max_)


def line_detect(source: Image, thresh1: int, thresh2: int,
                thresh_hough: int = 150, max_count: int = 30,
                is_debug=False):
    result_buffer = cv2.Canny(source.get_buffer(),
                              threshold1=thresh1,
                              threshold2=thresh2,
                              apertureSize=3,
                              L2gradient=True)
    features = cv2.HoughLines(result_buffer,
                              rho=1,
                              theta=numpy.pi / 180,
                              threshold=thresh_hough,
                              min_theta=0,
                              max_theta=numpy.pi, )
    result = Dataset1D()
    if is_debug:
        result_image = Image.from_buffer(result_buffer)
        result_image.show_image()
    i = 0
    for line in features:
        distance = line[0][0]  # Distance as the zero position
        theta = line[0][1]  # Angle of the perpendicular line
        theta = 1e-10 if theta < 1e-10 else theta
        x0 = distance * numpy.cos(theta)  # Intersection position with perpendicular line
        y0 = distance * numpy.sin(theta)  # Intersection position with perpendicular line
        height = result_buffer.shape[0]
        width = result_buffer.shape[1]
        vec1 = Vector2D(x=int(x0 - width * numpy.sin(theta)),
                        y=int(y0 + height * numpy.cos(theta)))
        vec2 = Vector2D(x=int(x0 + width * numpy.sin(theta)),
                        y=int(y0 - height * numpy.cos(theta)))
        result.append(Object(region=Line2D.from_vectors(vec1, vec2)))
        i += 1
        if i >= max_count:
            break

    return result


def blob_detect(source: Image,
                thresh: int, max_count: int = 30,
                is_debug=False):
    result_buffer = cv2.threshold(source.get_buffer(),
                                  thresh=thresh, maxval=255,
                                  type=cv2.THRESH_BINARY)
    detector = cv2.SimpleBlobDetector()
    features = detector.detect(result_buffer)
    result = Dataset1D()
    if is_debug:
        result_image = Image.from_buffer(result_buffer)
        result_image.show_image()
    i = 0
    for feature in features:
        pos = Vector2D(x=int(feature.pt[0]), y=int(feature.pt[1]))
        height, width = feature.size[0], feature.size[1]
        rect = Rect2D(x=pos.x, y=pos.y, width=width, height=height)
        result.append(Object(region=rect))
        i += 1
        if i >= max_count:
            break

    return result


def sift(source: Image,
         features=500, octaves: int = 3, contrastThresh=0.4, edgeThresh=10, sigma=2.0,
         is_output=True, is_debug=False):
    sift_ = cv2.SIFT_create(nfeatures=features, nOctaveLayers=octaves, contrastThreshold=contrastThresh,
                            edgeThreshold=edgeThresh, sigma=sigma)
    keypoints, desc = sift_.detectAndCompute(source.get_buffer(), None)
    if is_debug:
        result_buffer = cv2.drawKeypoints(source.get_buffer(), keypoints, None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SIFT", result_buffer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if is_output:
        result = Dataset1D()
        for keypoint in keypoints:
            pos = Vector2D(x=int(keypoint.pt[0]), y=int(keypoint.pt[1]))
            height, width = keypoint.size[0], keypoint.size[1]
            rect = Rect2D(pos.x, pos.y, width, height)
            result.append(Object(region=rect))
        return result
    else:
        return keypoints, desc


def surf(source: Image,
         metricThresh=1000, octaves: int = 3,
         is_output=True, is_debug=False):
    surf_ = cv2.xfeatures2d.SURF_create(metricThresh, octaves)
    keypoints, desc = surf_.detectAndCompute(source.get_buffer(), None)
    if is_debug:
        res_buffer = cv2.drawKeypoints(source.get_buffer(), keypoints, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SURF", res_buffer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if is_output:
        result = Dataset1D()
        for keypoint in keypoints:
            pos = Vector2D(x=int(keypoint.pt[0]), y=int(keypoint.pt[1]))
            height, width = keypoint.size[0], keypoint.size[1]
            rect = Rect2D(pos.x, pos.y, width, height)
            result.append(Object(region=rect))
        return result
    else:
        return keypoints, desc


def feature_detect(source: Image, match_func="sift", **kwargs):
    if match_func == "sift":
        features = 500 if kwargs["features"] is None else int(kwargs["features"])
        octaves = 3 if kwargs["octaves"] is None else int(kwargs["octaves"])
        contrast_thresh = 0.4 if kwargs["contrast_thresh"] is None else float(kwargs["contrastThresh"])
        edge_thresh = 10 if kwargs["edge_thresh"] is None else int(kwargs["edgeThresh"])
        sigma = 2.0 if kwargs["sigma"] is None else float(kwargs["sigma"])
        kp, desc = sift(source, features, octaves, contrast_thresh, edge_thresh, sigma, is_output=False)
    elif match_func == "surf":
        metric_thresh = 1000 if kwargs["metric_thresh"] is None else int(kwargs["metricThresh"])
        octaves = 3 if kwargs["octaves"] is None else int(kwargs["octaves"])
        kp, desc = surf(source, metric_thresh, octaves, is_output=False)
    else:
        raise Exception("The match function is not abnormal")
    return kp, desc


def feature_match(source: Image, other: Image, match_func="sift", **kwargs):
    features1, description1 = feature_detect(source, match_func, **kwargs)
    features2, description2 = feature_detect(other, match_func, **kwargs)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.knnMatch(description1, description2, 2)
    ratio = 0.5 if kwargs["ratio"] is None else kwargs["ratio"]
    best_matches = [first for first, second in matches if first.distance < second.distance * ratio]
    best_features1 = [features1[match.queryIdx].pt for match in best_matches]
    best_features2 = [features2[match.queryIdx].pt for match in best_matches]
    is_debug = False if kwargs["is_debug"] is None else kwargs["is_debug"]
    if is_debug:
        res_buffer = cv2.drawMatches(source.get_buffer(), features1, other.get_buffer(), features2, best_matches, None,
                                     flags=cv2.cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Feature Match", res_buffer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return Dataset1D.from_feature_list(best_features1), Dataset1D.from_feature_list(best_features2)


def perspective_transform(source: Image, other: Image, points: Dataset1D = None,
                          match_func="sift", **kwargs):
    features1, description1 = feature_detect(source, match_func, **kwargs)
    features2, description2 = feature_detect(other, match_func, **kwargs)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.knnMatch(description1, description2, 2)
    ratio = 0.5 if kwargs["ratio"] is None else kwargs["ratio"]
    best_matches = [first for first, second in matches if first.distance < second.distance * ratio]
    src_features = numpy.float32([features1[match.queryIdx].pt for match in best_matches])
    dst_features = numpy.float32([features2[match.queryIdx].pt for match in best_matches])
    transfer, mask = cv2.findHomography(src_features, dst_features, cv2.RANSAC, 5.0)
    is_debug = False if kwargs["is_debug"] is None else kwargs["is_debug"]
    if is_debug:
        src_rect = numpy.float32([[[0, 0]],
                                  [[0, source.height - 1]],
                                  [[source.width - 1, source.height - 1]],
                                  [[source.width - 1, 0]]])
        dst_rect = cv2.perspectiveTransform(src_rect, transfer)
        src_buffer = source.get_buffer()
        dst_buffer = cv2.polylines(other.get_buffer(), [numpy.int32(dst_rect)], True, 255, 3, cv2.LINE_AA)
        res_buffer = cv2.drawMatches(src_buffer, src_features, dst_buffer, dst_features, best_matches, None,
                                     flags=cv2.cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Perspective Transform", res_buffer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if points is not None:
        points = Vector2D.list_to_array_xy(points.pos_list())
        result = cv2.perspectiveTransform(points, transfer)
        return Dataset1D.from_feature_array(result)


def find_epiline(source: Image, other: Image, match_func="sift", **kwargs):
    pass
