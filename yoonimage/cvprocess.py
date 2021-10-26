import cv2
import cv2.cv2
from cv2 import cv2
import numpy

from yoonimage import YoonImage, YoonObject, YoonDataset
from yoonpytory.figure import YoonVector2D, YoonRect2D, YoonLine2D


def template_match(source: YoonImage, template: YoonImage,
                   score: float = 0.7, mode: int = cv2.TM_SQDIFF_NORMED):
    assert isinstance(template, YoonImage)
    match_container = cv2.matchTemplate(image=source.get_buffer(),
                                        templ=template.get_buffer(),
                                        method=mode)
    min_, max_, min_pos, max_pos = cv2.minMaxLoc(match_container)
    if max_ > score:
        min_pos = YoonVector2D.from_array(min_pos)
        max_pos = YoonVector2D.from_array(max_pos)
        center_pos = (min_pos + max_pos) / 2
        rect = YoonRect2D(x=center_pos.x, y=center_pos.y, width=template.width, height=template.height)
        return YoonObject(region=rect, image=source.crop(rect), score=max_)


def line_detect(source: YoonImage, thresh1: int, thresh2: int,
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
    result = YoonDataset()
    if is_debug:
        result_image = YoonImage.from_buffer(result_buffer)
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
        vec1 = YoonVector2D(x=int(x0 - width * numpy.sin(theta)),
                            y=int(y0 + height * numpy.cos(theta)))
        vec2 = YoonVector2D(x=int(x0 + width * numpy.sin(theta)),
                            y=int(y0 - height * numpy.cos(theta)))
        result.append(YoonObject(region=YoonLine2D.from_vectors(vec1, vec2)))
        i += 1
        if i >= max_count:
            break

    return result


def blob_detect(source: YoonImage,
                thresh: int, max_count: int = 30,
                is_debug=False):
    result_buffer = cv2.threshold(source.get_buffer(),
                                  thresh=thresh, maxval=255,
                                  type=cv2.THRESH_BINARY)
    detector = cv2.SimpleBlobDetector()
    features = detector.detect(result_buffer)
    result = YoonDataset()
    if is_debug:
        result_image = YoonImage.from_buffer(result_buffer)
        result_image.show_image()
    i = 0
    for feature in features:
        pos = YoonVector2D(x=int(feature.pt[0]), y=int(feature.pt[1]))
        height, width = feature.size[0], feature.size[1]
        rect = YoonRect2D(x=pos.x, y=pos.y, width=width, height=height)
        result.append(YoonObject(region=rect))
        i += 1
        if i >= max_count:
            break

    return result


def sift(source: YoonImage,
         octaves: int = 3, contrastThresh=0.4, edgeThresh=10, sigma=2.0,
         is_output=True, is_debug=False):
    sift_ = cv2.xfeatures2d.SIFT_create(nOctaveLayers=octaves, contrastThreshold=contrastThresh,
                                        edgeThreshold=edgeThresh, sigma=sigma)
    features, desc = sift_.detectAndCompute(source.get_buffer(), None)
    if is_debug:
        result_buffer = cv2.drawKeypoints(source.get_buffer(), features, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SIFT", result_buffer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if is_output:
        result = YoonDataset()
        for feature in features:
            pos = YoonVector2D(x=int(feature.pt[0]), y=int(feature.pt[1]))
            height, width = feature.size[0], feature.size[1]
            rect = YoonRect2D(pos.x, pos.y, width, height)
            result.append(YoonObject(region=rect))
        return result
    else:
        return features, desc


def surf(source: YoonImage,
         metricThresh=1000, octaves: int = 3,
         is_output=True, is_debug=False):
    surf_ = cv2.xfeatures2d.SURF_create(metricThresh, octaves)
    features, desc = surf_.detectAndCompute(source.get_buffer(), None)
    if is_debug:
        res_buffer = cv2.drawKeypoints(source.get_buffer(), features, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SURF", res_buffer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if is_output:
        result = YoonDataset()
        for feature in features:
            pos = YoonVector2D(x=int(feature.pt[0]), y=int(feature.pt[1]))
            height, width = feature.size[0], feature.size[1]
            rect = YoonRect2D(pos.x, pos.y, width, height)
            result.append(YoonObject(region=rect))
        return result
    else:
        return features, desc


def __feature_detect(source: YoonImage, other: YoonImage, match_func=sift, **kwargs):
    if match_func == sift:
        octaves = 3 if kwargs["octaves"] is None else kwargs["octaves"]
        contrastThresh = 0.4 if kwargs["contrastThresh"] is None else kwargs["contrastThresh"]
        edgeThresh = 10 if kwargs["edgeThresh"] is None else kwargs["edgeThresh"]
        sigma = 2.0 if kwargs["sigma"] is None else kwargs["sigma"]
        kp1, desc1 = match_func(source, octaves, contrastThresh, edgeThresh, sigma, is_output=False)
        kp2, desc2 = match_func(other, octaves, contrastThresh, edgeThresh, sigma, is_output=False)
    elif match_func == surf:
        metricThresh = 1000 if kwargs["metricThresh"] is None else kwargs["metricThresh"]
        octaves = 3 if kwargs["octaves"] is None else kwargs["octaves"]
        kp1, desc1 = match_func(source, metricThresh, octaves, is_output=False)
        kp2, desc2 = match_func(other, metricThresh, octaves, is_output=False)
    else:
        raise Exception("The match function is not abnormal")
    return kp1, desc1, kp2, desc2


def feature_match(source: YoonImage, other: YoonImage, match_func=sift, **kwargs):
    features1, desc1, features2, desc2 = __feature_detect(source, other, match_func, **kwargs)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.knnMatch(desc1, desc2, 2)
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
    best_features1 = [YoonVector2D.from_array(point) for point in best_features1]
    best_features2 = [YoonVector2D.from_array(point) for point in best_features2]
    return YoonDataset.from_list(best_features1), YoonDataset.from_list(best_features2)