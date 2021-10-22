import cv2
import cv2.cv2
import numpy

from yoonimage import YoonImage, YoonObject, YoonDataset
from yoonpytory.figure import YoonVector2D, YoonRect2D, YoonLine2D


def find_template(source: YoonImage,
                  template: YoonImage,
                  score: float = 0.7,
                  mode: int = cv2.cv2.TM_SQDIFF_NORMED):
    assert isinstance(template, YoonImage)
    match_container = cv2.cv2.matchTemplate(image=source.get_buffer(),
                                            templ=template.get_buffer(),
                                            method=mode)
    min_, max_, min_pos, max_pos = cv2.cv2.minMaxLoc(match_container)
    if max_ > score:
        min_pos = YoonVector2D.from_array(min_pos)
        max_pos = YoonVector2D.from_array(max_pos)
        center_pos = (min_pos + max_pos) / 2
        rect = YoonRect2D(x=center_pos.x, y=center_pos.y, width=template.width, height=template.height)
        return YoonObject(region=rect, image=source.crop(rect), score=max_)


def find_lines(source: YoonImage,
               thresh1: int,
               thresh2: int,
               thresh_hough: int = 150,
               max_count: int = 30):
    result_buffer = cv2.cv2.Canny(source.get_buffer(),
                                  threshold1=thresh1,
                                  threshold2=thresh2,
                                  apertureSize=3,
                                  L2gradient=True)
    line_container = cv2.cv2.HoughLines(result_buffer,
                                        rho=1,
                                        theta=numpy.pi / 180,
                                        threshold=thresh_hough,
                                        min_theta=0,
                                        max_theta=numpy.pi)
    result = YoonDataset()
    result_image = YoonImage.from_buffer(result_buffer)
    # result_image.show_image()  # Remain for logging
    iCount = 0
    for line in line_container:
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
        result.append(
            YoonObject(region=YoonLine2D.from_vectors(vec1, vec2), image=result_image)
        )
        iCount += 1
        if iCount >= max_count:
            break

    return result


def find_blobs(source: YoonImage,
               thresh: int,
               max_count: int = 30):
    result_buffer = cv2.cv2.threshold(source.get_buffer(),
                                      thresh=thresh, maxval=255,
                                      type=cv2.cv2.THRESH_BINARY)
    detector = cv2.cv2.SimpleBlobDetector()
    blob_container = detector.detect(result_buffer)
    result = YoonDataset()
    result_image = YoonImage.from_buffer(result_buffer)
    result_image.show_image()  # Remain for logging
    i = 0
    for feature in blob_container:
        pos = YoonVector2D(x=int(feature.pt[0]), y=int(feature.pt[1]))
        height, width = feature.size[0], feature.size[1]
        rect = YoonRect2D(x=pos.x, y=pos.y, width=width, height=height)
        result.append(
            YoonObject(region=rect, image=result_image.crop(rect))
        )
        i += 1
        if i >= max_count:
            break

    return result

