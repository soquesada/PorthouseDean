import matplotlib.pyplot as plt
import keras_ocr
import math
from PIL import Image, ImageChops
import pathlib
import os
import cv2
import numpy as np

# FILE_NAME = "floorplan1.jpg"
# pipeline = keras_ocr.pipeline.Pipeline()

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return x_mid, y_mid

def inpaint_text(img_path, pipeline):
    img = keras_ocr.tools.read(img_path) # read image

    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return img

def prep_images(path, pipeline):
    # path = pathlib.Path(os.path.join(r'assets/test/', FILE_NAME))

    uncropped = inpaint_text(str(path), pipeline)
    plt.imshow(uncropped)
    
    # assign images
    # uncropped = Image.open(original_path)
    # uncropped = cv2.imread(str(original_path))

    height, width = uncropped.shape[0], uncropped.shape[1]
    before = uncropped[:, : width//2]
    after = uncropped[:, width//2 :]

    # cv2.imshow('og', original)
    # cv2.imshow('new', new)
    # cv2.waitKey(0)

    before = uncropped.crop((0, 0, width//2, height)) #.save("og", "JPEG") # (l, u, r, d)
    after = uncropped.crop((width//2, 0, width, height))#.save("new", "JPEG")
    # original.show()
    # new.show()

    return before, after

def align_images(before, after):
    height, width = before.shape[0], before.shape[1]

    # Convert to greyscale
    before_grey = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_grey = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Configure ORB feature detector Algorithm with 1000 features.
    orb_detector = cv2.ORB_create(1000)

    # Extract key points and descriptors for both images
    key_point1, descriptor1 = orb_detector.detectAndCompute(before_grey, None)
    key_point2, descriptor2 = orb_detector.detectAndCompute(after_grey, None)

    # Display keypoints for reference image in green color
    key_points_ref = cv2.drawKeypoints(after, key_point1, 0, (0,222,0), None)
    key_points_ref = cv2.resize(key_points_ref, (width//2, height//2))

    # cv2.imshow('Key Points', key_points_ref)
    # cv2.waitKey(0)

    ## Feature matching
    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptor1, descriptor2) # Match the two sets of descriptors
    
    # Sort matches on the basis of their Hamming distance
    matches = list(matches)
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Display only 100 best matches
    img_match = cv2.drawMatches(before, key_point2, after, key_point1, matches[:100], None, flags=2)
    # img_match = cv2.drawMatches(after, key_point1, before, key_point2, matches[:100], None, flags=2) # If 1st not working
    img_match = cv2.resize(img_match, (width//3, height//3))

    # cv2.imshow('Image Match', img_match)
    # cv2.waitKey(0)

    # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = key_point1[matches[i].queryIdx].pt
        p2[i, :] = key_point2[matches[i].trainIdx].pt
    
    # Find the homography matrix
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use homography matrix to transform the unaligned image wrt the reference image
    aligned_img = cv2.warpPerspective(before, homography, (width, height))
    # Resizing the image to display in our screen (optional)
    aligned_img = cv2.resize(aligned_img, (width, height))
    
    # Copy of input image
    # img_test_cp = before.copy()
    # img_test_cp = cv2.resize(img_test_cp, (width//3, height//3))
    # Save the align image output.
    # cv2.imwrite('output.jpg', aligned_img)
    
    # cv2.imshow('Input Image', img_test_cp)
    # cv2.imshow('Output Image', aligned_img)
    # cv2.waitKey(0)
    return aligned_img, after
  
def find_diff(before, after):
    before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

    before = Image.fromarray(before)
    after = Image.fromarray(after)

    # finding difference
    diff = ImageChops.difference(before, after)
    
    # showing the difference
    diff.show()

def diff_pipeline(path):
    # read images
    before, after = prep_images(path)

    # align images
    before, after = align_images(before, after)
    # cv2.imshow('Before', before)
    # cv2.imshow('After', after)
    # cv2.waitKey(0)

    find_diff(before, after)

# if __name__ == "__main__":
    # read images
    # prep_images()

    # align images
    # before, after = align_images(before, after)
    # cv2.imshow('Before', before)
    # cv2.imshow('After', after)
    # cv2.waitKey(0)

    # find_diff(before, after)