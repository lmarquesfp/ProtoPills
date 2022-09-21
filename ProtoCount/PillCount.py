import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from scipy import ndimage
from tensorflow_object_counting_api.single_image_object_counting import SingleImageObjCount

def process(img):
    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    BW = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    BW2 = seg_watershed(BW, gray)
    return caculate_pill(BW2)

def seg_watershed(BW, gray):
    # Watershed Transform
    D = ndimage.distance_transform_edt(BW)
    ret, mask = cv2.threshold(D, 0.4 * D.max(), 255, 0)
    mask = np.uint8(mask)

    # Marker labeling Watershed Line ==> line
    ret, markers = cv2.connectedComponents(mask)
    labels = watershed(-D, markers, mask=gray, watershed_line=True)
    line = np.zeros(BW.shape, dtype=np.uint8)
    line[labels == 0] = 255
    line = cv2.dilate(line, np.ones((2, 2), np.uint8), iterations=1)

    # Creating BW2
    BW2 = BW.copy()
    BW2[line == 255] = 0
    return BW2


def caculate_pill(BW2):
    label_image = label(BW2)
    A = [r.area for r in regionprops(label_image)]
    A.sort()

    num = 0
    S = 0
    num_pill = 0
    warn = False
    # Find minArea
    for i in range(len(A)):
        rateArea = A[i] / A[0]
        if rateArea < 1.15:
            num = num + 1
            S = S + A[i]
    if num != 0:
        minArea = S / num

    # Calculate num_pill
    for i in range(len(A)):
        rate = A[i] / minArea
        appro_rate = round(rate, 0)
        delta_rate = abs(rate - appro_rate)

        if delta_rate < 0.3:
            num_pill = num_pill + appro_rate
        else:
            warn = True

    if num_pill == 0:
        warn = True
    return len(A)

def count_thresh(image):
    #apply grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    #apply Histogram Equalization
    equ = cv2.equalizeHist(blur)
    # stacking images side-by-side
    res = np.hstack((gray, equ))
    #apply treshold
    ret,thresh = cv2.threshold(equ,127,255,cv2.THRESH_BINARY)
    #erosion
    kernel = np.ones((5, 5), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    erosion = cv2.erode(thresh, kernel, iterations=0)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    cv2.imshow('Original image',image)
    # cv2.imshow('Gray image', gray)
    # cv2.imshow('Blur', blur)
    # cv2.imshow('Canny', canny)
    cv2.imshow('Dilated', dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def count_contours(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(image, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

    # cv2.imshow('Original image',image)
    # #cv2.imshow('Gray image', gray)
    # cv2.imshow('Blur', blur)
    # cv2.imshow('Canny', canny)
    # cv2.imshow('Dilated', dilated)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return len(cnt)
