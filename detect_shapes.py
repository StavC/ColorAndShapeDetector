import shape_detector
from color_labeler import ColorLabeler
import argparse
import imutils
import cv2

def main():


    image=cv2.imread('shapes_and_colors.jpg')
    resized=imutils.resize(image,width=300)
    ratio=image.shape[0]/float(resized.shape[0]) # ratio of resizing

    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow('img',thresh)
    cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    shape_d = shape_detector
    color_d=ColorLabeler()


    for c in cnts:
        M=cv2.moments(c)

        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)

        shape = shape_d.detect(c)
        color=color_d.label(lab,c)

        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, f'{color}, {shape}', (cX-20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(0)




if __name__ == '__main__':
    main()
