import cv2
import numpy as np

from matplotlib import pyplot as plt

# Minimum percentage of pixels of same hue to consider dominant colour
# MIN_PIXEL_CNT_PCT = (1.0/20.0)

image = cv2.imread('classgt_00000.png',1)


frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# We're only interested in the hue
# h,_,_ = cv2.split(image_hsv)

peak = np.array([210,0, 200])

# peak1 = np.array([209,0, 200]) #Example
# peak2 = np.array([211,0, 200])
# Let's count the number of occurrences of each hue
# bins = np.bincount(h.flatten())
mask = cv2.inRange(frame, peak,peak)


res = cv2.bitwise_and(frame,frame, mask= mask)

cv2.imshow('masked',res)
cv2.waitKey(0)





# And then find the dominant hues
# peaks = np.where(bins > (h.size * MIN_PIXEL_CNT_PCT))[0]




# Now let's find the shape matching each dominant hue
# for i, peak in enumerate(peaks):

#     print(peak)
    # # First we create a mask selecting all the pixels of this hue
    # mask = cv2.inRange(h, peak, peak)
    # # And use it to extract the corresponding part of the original colour image
    # blob = cv2.bitwise_and(image, image, mask=mask)

    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for j, contour in enumerate(contours):
    #     bbox = cv2.boundingRect(contour)
    #     # Create a mask for this contour
    #     contour_mask = np.zeros_like(mask)
    #     cv2.drawContours(contour_mask, contours, j, 255, -1)

    #     print(f"Found hue %d in region %s." % (peak, bbox))
    #     # Extract and save the area of the contour
    #     region = blob.copy()[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    #     region_mask = contour_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    #     region_masked = cv2.bitwise_and(region, region, mask=region_mask)
    #     file_name_section = "colourblobs-%d-hue_%03d-region_%d-section.png" % (i, peak, j)
    #     cv2.imwrite(file_name_section, region_masked)
    #     print(f'* wrote {file_name_section}')

    #     # Extract the pixels belonging to this contour
    #     result = cv2.bitwise_and(blob, blob, mask=contour_mask)
    #     # And draw a bounding box
    #     top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
    #     cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
    #     file_name_bbox = "colourblobs-%d-hue_%03d-region_%d-bbox.png" % (i, peak, j)
    #     cv2.imwrite(file_name_bbox, result)
    #     print(f'* wrote {file_name_bbox}')