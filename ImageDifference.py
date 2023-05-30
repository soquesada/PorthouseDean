# import matplotlib.pyplot as plt
import keras_ocr
import math
from PIL import Image, ImageChops
import cv2
import numpy as np
from skimage.metrics import structural_similarity

from skimage.metrics import structural_similarity as compare_ssim
import imutils

# Calculate x and y midpoints
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return x_mid, y_mid

# Remove text from image
def inpaint_text(img_path, pipeline):
    img = keras_ocr.tools.read(img_path)

    # Generate (word, box) tuples 
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

# Prepare images by:
# 1. Removing text
# 2. Assuming that the file contains both before and after plans (left and right respectively), crop the image
def prep_images(path, pipeline):
    uncropped = inpaint_text(str(path), pipeline)
    
    # Save image
    cv2.imwrite('no_text.png', uncropped)

    height, width = uncropped.shape[0], uncropped.shape[1]
    before = uncropped[:, : width//2]
    after = uncropped[:, width//2 :]

    # cv2.imshow('og', original)
    # cv2.imshow('new', new)
    # cv2.waitKey(0)

    # Convert numpy array to PIL image
    # uncropped_image = Image.fromarray(uncropped)

    # before = uncropped_image.crop((0, 0, width//2, height)) #.save("og", "JPEG") # (l, u, r, d)
    # after = uncropped_image.crop((width//2, 0, width, height))#.save("new", "JPEG")
    # original.show()
    # new.show()

    return before, after

# Align the before and after plans
# Returns 2 cv2 images
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
    try:
        img_match = cv2.drawMatches(before, key_point2, after, key_point1, matches[:100], None, flags=2)
    except:
        img_match = cv2.drawMatches(after, key_point1, before, key_point2, matches[:100], None, flags=2) # If 1st not working
        print("Exception")
    img_match = cv2.resize(img_match, (width//3, height//3))

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

    return aligned_img, after

# Find and highlight the difference between the before and after images
def find_diff(before, after):
    # Method 2:
    before_grey = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_grey = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(before_grey, after_grey, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    # Threshold the difference image, then find contours to obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        # Compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return before, after, diff

def find_diff_m1(before, after):
    # Method 1:
    before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

    before = Image.fromarray(before)
    after = Image.fromarray(after)

    # Finding difference
    diff = ImageChops.difference(before, after)
    diff = ImageChops.invert(diff)

def find_diff_m2(before, after):
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    return before, after, filled_after

def find_diff_m3(before, after):
    # Get the sizes of the images
    height, width = before.shape[0], before.shape[1]

    # Convert to PIL Image
    before = Image.fromarray(before)
    after = Image.fromarray(after)

    # Create a new image to store the differences
    diff = Image.new("RGB", (width, height), (255, 255, 255))

    # Loop over the pixels in the images and compare them
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixels at this position in each image
            r1, g1, b1 = before.getpixel((x, y))
            r2, g2, b2 = after.getpixel((x, y))

            # If the pixels are different, set the corresponding pixel in the diff image to red
            if (r1, g1, b1) != (r2, g2, b2):
                diff.putpixel((x, y), (255, 0, 0))

    return np.asarray(before), np.asarray(after), np.asarray(diff)

def find_diff_m4(before, after):
    before = Image.fromarray(before)
    after = Image.fromarray(after)

    diff = before.copy()
    diff = cv2.absdiff(before, after, diff)
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Increasing the size of differences after that we can capture them all
    for i in range(0, 3):
        dilated = cv2.dilate(grey.copy(), None, iterations=i+1)

    (T, thresh) = cv2.threshold(dilated, 3, 255, cv2.THRESH_BINARY)

    # Now we have to find contours in the binarized image
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        # Nicely fiting a bounding box to the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(after, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return before, after, thresh

    return np.asarray(before), np.asarray(after), np.asarray(diff)

# Format the images for output on the UI
def combine_images(before, after, diff, ratio):
    images = [Image.fromarray(before), Image.fromarray(after), Image.fromarray(diff)]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    # new_im.thumbnail(ratio) # * Resizes image to label
    new_im = np.array(new_im)

    return new_im

# Overarching pipeline to find the difference between two images
def diff_pipeline(path, ratio):
    print("Started pipeline")

    # Read images
    pipeline = keras_ocr.pipeline.Pipeline()
    before, after = prep_images(path, pipeline)

    # Align images
    before, after = align_images(before, after)
    cv2.imwrite('before.png', before)
    cv2.imwrite('after.png', after)

    # Find diff
    before_box, after_box, diff = find_diff_m2(before, after)

    # Combine image
    image = combine_images(before_box, after_box, diff, ratio)

    print("End of pipeline")
    return image