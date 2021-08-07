import cv2
import cv2.cv2
import numpy

from yoonimage import YoonImage, YoonObject, YoonDataset
from yoonpytory.figure import YoonVector2D, YoonRect2D, YoonLine2D


def find_template(pSourceImage : YoonImage,
                  pTemplateImage : YoonImage,
                  dScore: float = 0.7,
                  nMode: int = cv2.cv2.TM_SQDIFF_NORMED):
    assert isinstance(pTemplateImage, YoonImage)
    pMatchStorage = cv2.cv2.matchTemplate(image=pSourceImage.get_buffer(),
                                          templ=pTemplateImage.get_buffer(),
                                          method=nMode)
    _, dMaxValue, pMinPos, pMaxPos = cv2.cv2.minMaxLoc(pMatchStorage)
    if dMaxValue > dScore:
        pMinPos = YoonVector2D.from_array(pMinPos)
        pMaxPos = YoonVector2D.from_array(pMaxPos)
        pCenterPos = (pMinPos + pMaxPos) / 2
        pRect = YoonRect2D(pPos=pCenterPos, dWidth=pTemplateImage.width, dHeight=pTemplateImage.height)
        return YoonObject(pRegion=pRect, pImage=pSourceImage.crop(pRect), dScore=dMaxValue)


def find_lines(pSourceImage: YoonImage,
               nThresh1: int,
               nThresh2: int,
               nThreshHough: int = 150,
               nMaxCount: int = 30):
    pResultBuffer = cv2.cv2.Canny(pSourceImage.get_buffer(),
                                  threshold1=nThresh1,
                                  threshold2=nThresh2,
                                  apertureSize=3,
                                  L2gradient=True)
    pLineStorage = cv2.cv2.HoughLines(pResultBuffer,
                                      rho=1,
                                      theta=numpy.pi / 180,
                                      threshold=nThreshHough,
                                      min_theta=0,
                                      max_theta=numpy.pi)
    pResultDataset = YoonDataset()
    pResultImage = YoonImage(pBuffer=pResultBuffer)
    pResultImage.show_image()  # Remain for logging
    iCount = 0
    for pLine in pLineStorage:
        dDistance = pLine[0][0]  # Distance as the zero position
        dTheta = pLine[0][1]  # Angle of the perpendicular line
        dX0 = dDistance * numpy.cos(dTheta)  # Intersection position with perpendicular line
        dY0 = dDistance * numpy.sin(dTheta)  # Intersection position with perpendicular line
        nScale = pResultBuffer.shape[0] + pResultBuffer.shape[1]
        pVector1 = YoonVector2D(dX=int(dX0 - nScale * numpy.sin(dTheta)),
                                dY=int(dY0 + nScale * numpy.cos(dTheta)))
        pVector2 = YoonVector2D(dX=int(dX0 + nScale * numpy.sin(dTheta)),
                                dY=int(dY0 - nScale * numpy.cos(dTheta)))
        pResultDataset.append(
            YoonObject(pRegion=YoonLine2D(None, None, None, pVector1, pVector2),
                       pImage=pResultImage.__copy__())
        )
        iCount += 1
        if iCount >= nMaxCount:
            break

    return pResultDataset


def find_blobs(pSourceImage: YoonImage,
               nThreshold: int,
               nMaxCount: int = 30):

    pResultBuffer = cv2.cv2.threshold(pSourceImage.get_buffer(),
                                      thresh=nThreshold, maxval=255,
                                      type=cv2.cv2.THRESH_BINARY)
    pDetector = cv2.cv2.SimpleBlobDetector()
    pBlobStorage = pDetector.detect(pResultBuffer)
    pResultDataset = YoonDataset()
    pResultImage = YoonImage(pBuffer=pResultBuffer)
    pResultImage.show_image()  # Remain for logging
    iCount = 0
    for pKeypoint in pBlobStorage:
        pPosition = YoonVector2D(dx=int(pKeypoint.pt[0]), dy=int(pKeypoint.pt[1]))
        dHeight, dWidth = pKeypoint.size[0], pKeypoint.size[1]
        pRect = YoonRect2D(pPos=pPosition, dWidth=dWidth, dHeight=dHeight)
        pResultDataset.append(
            YoonObject(pRegion=pRect,
                       pImage=pResultImage.crop(pRect))
        )
        iCount += 1
        if iCount >= nMaxCount:
            break

    return pResultDataset
