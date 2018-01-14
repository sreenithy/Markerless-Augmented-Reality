import numpy as np
import cv2
from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0
    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.
    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

img1 = cv2.imread('image.jpeg',0)          # queryImage
img2 = cv2.imread('image.jpeg',0) # trainImage

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb  = cv2.ORB()

# SIFT method 1:
(kps1, desc1) = sift.detectAndCompute(img1, None)
print("SIFT # kps: {}, descriptors: {}".format(len(kps1), desc1.shape))
(kps2, desc2) = sift.detectAndCompute(img2, None)
print("SIFT # kps: {}, descriptors: {}".format(len(kps1), desc2.shape))

# SURF method 1:
(kps1, desc1) = surf.detectAndCompute(img1, None)
print("SURF # kps: {}, descriptors: {}".format(len(kps1), desc1.shape))
(kps2, desc2) = surf.detectAndCompute(img2, None)
print("SURF # kps: {}, descriptors: {}".format(len(kps1), desc2.shape))

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# SIFT method 2:
kp1 = sift.detect(img1,None)
dec1 = sift.compute(img1,kp1)

# SURF method 2:
kp2 = surf.detect(img1,None)
dec2 = surf.compute(img1,kp2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# Match descriptors.
matches = bf.match(desc1,desc2)

# Draw first 10 matches.
img3 = drawMatches(img1,kp1,img2,kp2,matches[:10])

cv2.waitKey(0)
