import numpy as np
import pandas as pd
import cv2
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt


def get_hw_ratio(arr):
    hull = ConvexHull(arr)
    ch_points = arr[hull.vertices]
    # Find the longest vector and its inclination angle
    max_dist = 0
    for i in range(ch_points.shape[0]):
        for j in range(i, ch_points.shape[0]):
            diff = ch_points[i] - ch_points[j]
            if np.linalg.norm(diff) > max_dist:
                max_dist = np.linalg.norm(diff)
                indices_longest_vector = [i, j]
    # Longest length vector
    longest_vector = ch_points[indices_longest_vector[1]] - ch_points[indices_longest_vector[0]]
    # Inclination angle
    alpha = np.arctan(longest_vector[1] / longest_vector[0])
    # Rotating the array of points to horizontal position
    rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                                [np.sin(alpha),  np.cos(alpha)]])
    arr_rotated = np.matmul(rotation_matrix, arr.T).T
    # Obtaining horizontal and vertical distances between the furthest points
    indices_vertical_length = np.array([np.argmin(arr_rotated[:, 1]), np.argmax(arr_rotated[:, 1])])
    perpendicular_length_rotated = arr_rotated[indices_vertical_length]
    perpendicular_length_rotated[1, 0] = perpendicular_length_rotated[0, 0]
    # Rotating back to get a perpendicular length
    inverse_rotation_matrix = np.array([[np.cos(-alpha), -np.sin(-alpha)],
                                        [np.sin(-alpha),  np.cos(-alpha)]])
    perpendicular_length = np.matmul(inverse_rotation_matrix, perpendicular_length_rotated.T).T.astype(int)

    dic = {"longest_dist": np.array([ch_points[indices_longest_vector[0]], ch_points[indices_longest_vector[1]]]).astype(int),
           "perpendicular_dist": perpendicular_length,
           "ch_points": ch_points.astype(int)}
    dic["hw_ratio"] = np.linalg.norm(longest_vector) / np.linalg.norm(perpendicular_length[1] - perpendicular_length[0])

    return dic
# # Drawing an image
# img = cv2.polylines(img, [ch_points.astype(int)], True, (0, 0, 255), 1)
#
# # for point in arr:
# #     img = cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1)
# #
# img = cv2.line(img, ch_points[indices_longest_vector[0]].astype(int), ch_points[indices_longest_vector[1]].astype(int), (0, 255, 0), 1)
# img = cv2.line(img, perpendicular_length[0], perpendicular_length[1], (0, 255, 0), 1)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

