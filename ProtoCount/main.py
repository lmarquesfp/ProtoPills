import PillCount
import cv2

if __name__ == "__main__":
    image = cv2.imread('C:/FamiliprixGit/ProtoCount/images/4.jpg')
    #SingleImageObjCount.CountObjects(image)
    result = PillCount.count_contours(image)
    print(result)
