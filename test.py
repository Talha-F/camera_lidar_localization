import numpy as np
import os
from reconstruction import draw_epipolar_lines
import cv2


if __name__ == '__main__':
    image_1_foldername = "/media/akshay/Data/16822/data/epipolar_ordered/"
    image_1 = cv2.imread(image_1_foldername + "1.png")
    image_2 = cv2.imread(image_1_foldername + "2.png")
    img1_2_keypoints = cv2.hconcat([image_1,
                                    image_2])
    corres = np.load("/media/akshay/Data/16822/data/superpoint_old_1/1_2_matches.npz")
    point1 = corres['keypoints0']
    point2 = corres['keypoints1']
    matches = corres['matches']
    confidence = corres['match_confidence']
    print(confidence.shape)
    valid = (matches > -1) & \
            (confidence > 0.9)
    point1_valid = point1[valid]
    point2_valid = point2[matches[valid]]
    print(img1_2_keypoints.shape)
    print(point1_valid.shape)
    print(point2_valid.shape)

    for point_1, point_2 in zip(point1_valid,
                                point2_valid):
        img_lines = draw_epipolar_lines(img1_2_keypoints, 
                                        point_1,
                                        point_2)
    cv2.imshow("aabb", img_lines)
    cv2.waitKey(0)