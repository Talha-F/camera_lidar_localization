import numpy as np
import cv2
from matplotlib import cm

def to_homogenous(points):
    return np.concatenate((points, 
                           np.ones((points.shape[0], 1))), axis=1)

def plot_pcd_on_image(image,
                      pcd_all,
                      lidar2world,
                      world2cam,
                      cam_K):
    colors = np.int16(cm.rainbow(np.linspace(0, 1, 25))*255)
    pcd = pcd_all[:, :3]
    labels = np.int16(pcd_all[:, -1])
    pcd_homogenous = to_homogenous(pcd)
    pcd_world = (lidar2world@pcd_homogenous.T).T
    pcd_camera = (world2cam@pcd_world.T).T
    pcd_camera /= pcd_camera[:, -1].reshape(-1, 1)
    pcd_camera = pcd_camera[:, :3]
    pcd_image = (cam_K@pcd_camera.T).T
    pcd_image /= pcd_image[:, -1].reshape(-1, 1)

    marked_image = np.copy(image)
    in_image_mask = (pcd_image[:, 1] >= 0) & \
                    (pcd_image[:, 0] >= 0) & \
                    (pcd_image[:, 0] < image.shape[1]) & \
                    (pcd_image[:, 1] < image.shape[0]) & \
                    (pcd_camera[:, 2] >= 0)
    refined_pcd_camera = pcd_camera[in_image_mask]
    refined_pcd_image = pcd_image[in_image_mask]
    refined_labels = labels[in_image_mask]

    for i in range(refined_pcd_camera.shape[0]):
        color = colors[refined_labels[i]]
        cv2.circle(marked_image, 
                   (int(refined_pcd_image[i, 0]), int(refined_pcd_image[i, 1])),
                   2,
                   color.tolist(), 
                   -1,
                   4,
                   0)
    return marked_image

def select_points(event, 
                  x, 
                  y, 
                  flags, 
                  params):
    image = params[0]
    # list to append point to
    line_points = params[1]
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        if len(params) == 3:
            cv2.circle(image, (x, y), 0, params[2], 10)
        else:
            cv2.circle(image, (x, y), 0, (0, 255, 0), 10)
        line_points.append([x, y])

def plot_epipolar(image_1, 
                  image_2,
                  F):
    image_annotated = np.copy(image_1)
    annotated_line = np.copy(image_2)
    cv2.imshow("line_annotated", np.concatenate((image_annotated, annotated_line), 1))
    point_list = []
    num_points = input("Enter number of annotations you need ")
    while len(point_list) < int(num_points):
        print(f"Select {len(point_list)+1} point on image")
        color = (0, 0, 255)
        cv2.setMouseCallback('line_annotated', select_points, (image_annotated, point_list, color))
        color = np.random.randint(0, 255, (3,)).tolist()

        while(1):
            cv2.imshow("line_annotated",  np.concatenate((image_annotated, annotated_line), 1))
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(point_list) == int(num_points):
                break
            if len(point_list) >= 1:
                point = point_list[-1]
                image_annotated = cv2.circle(image_annotated,
                                            (point[0], point[1]),
                                            radius=10,
                                            color=color, 
                                            thickness=-1)
                image_annotated = cv2.putText(image_annotated, 
                                            str(len(point_list)),
                                            (point[0], point[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            color,
                                            3)
                line_endpoints = calculate_line(point, image_2, F)
                annotated_line = cv2.line(annotated_line, 
                                        (line_endpoints[0][0], line_endpoints[0][1]),
                                        (line_endpoints[1][0], line_endpoints[1][1]),
                                        color=color,
                                        thickness=2)
                annotated_line = cv2.putText(annotated_line, 
                                            str(len(point_list)),
                                            ((line_endpoints[0][0]+line_endpoints[1][0])//2,
                                            (line_endpoints[0][1]+line_endpoints[1][1])//2),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            color,
                                            3)

    image_annotated = np.copy(image_1)
    annotated_line = np.copy(image_2)
    for point_no in range(len(point_list)):
        color = np.random.randint(0, 255, (3,)).tolist()
        image_annotated = cv2.circle(image_annotated,
                                    (point_list[point_no]),
                                    radius=10,
                                    color=color, 
                                    thickness=-1)
        image_annotated = cv2.putText(image_annotated, 
                                      str(point_no+1),
                                      (point_list[point_no]),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1,
                                      color,
                                      3,
                                    )
        line_endpoints = calculate_line(point_list[point_no], 
                                        image_2, F)
        annotated_line = cv2.line(annotated_line, 
                                (line_endpoints[0][0], line_endpoints[0][1]),
                                (line_endpoints[1][0], line_endpoints[1][1]),
                                color=color,
                                thickness=2)
        annotated_line = cv2.putText(annotated_line, 
                                    str(point_no+1),
                                    ((line_endpoints[0][0]+line_endpoints[1][0])//2,
                                    (line_endpoints[0][1]+line_endpoints[1][1])//2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    color,
                                    3,
                                    )
    return image_annotated, annotated_line

def plot_epipolar_correspondence(image_1, 
                                 image_2,
                                 F,
                                 corresp_finder):
    image_annotated = np.copy(image_1)
    image_annotated_2 = np.copy(image_2)
    cv2.imshow("line_annotated", np.concatenate((image_annotated, image_annotated_2), 1))
    point_list = []
    num_points = input("Enter number of annotations you need ")
    while len(point_list) < int(num_points):
        print(f"Select {len(point_list)+1} point on image")
        color = (0, 0, 255)
        cv2.setMouseCallback('line_annotated', select_points, (image_annotated, point_list, color))
        color = np.random.randint(0, 255, (3,)).tolist()

        while(1):
            cv2.imshow("line_annotated",  np.concatenate((image_annotated, image_annotated_2), 1))
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(point_list) == int(num_points):
                break
            if len(point_list) >= 1:
                point = point_list[-1]
                image_annotated = cv2.circle(image_annotated,
                                            (point[0], point[1]),
                                            radius=10,
                                            color=color, 
                                            thickness=-1)
                image_annotated = cv2.putText(image_annotated, 
                                            str(len(point_list)),
                                            (point[0], point[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            color,
                                            3)
                pts2 = corresp_finder.find_correspondence(image_1, 
                                                          image_2,
                                                          point,
                                                          F)
                image_annotated_2 = cv2.circle(image_annotated_2,
                                               (pts2[0], pts2[1]),
                                               radius=10,
                                               color=color, 
                                               thickness=-1)
                image_annotated_2 = cv2.putText(image_annotated_2, 
                                                str(len(point_list)),
                                                (pts2[0], pts2[1]),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                color,
                                                3)

    # image_annotated = np.copy(image_1)
    # annotated_line = np.copy(image_2)
    # for point_no in range(len(point_list)):
    #     color = np.random.randint(0, 255, (3,)).tolist()
    #     image_annotated = cv2.circle(image_annotated,
    #                                 (point_list[point_no]),
    #                                 radius=10,
    #                                 color=color, 
    #                                 thickness=-1)
    #     image_annotated = cv2.putText(image_annotated, 
    #                                   str(point_no+1),
    #                                   (point_list[point_no]),
    #                                   cv2.FONT_HERSHEY_SIMPLEX,
    #                                   1,
    #                                   color,
    #                                   3,
    #                                 )
    #     line_endpoints = calculate_line(point_list[point_no], 
    #                                     image_2, F)
    #     annotated_line = cv2.line(annotated_line, 
    #                             (line_endpoints[0][0], line_endpoints[0][1]),
    #                             (line_endpoints[1][0], line_endpoints[1][1]),
    #                             color=color,
    #                             thickness=2)
    #     annotated_line = cv2.putText(annotated_line, 
    #                                 str(point_no+1),
    #                                 ((line_endpoints[0][0]+line_endpoints[1][0])//2,
    #                                 (line_endpoints[0][1]+line_endpoints[1][1])//2),
    #                                 cv2.FONT_HERSHEY_SIMPLEX,
    #                                 1,
    #                                 color,
    #                                 3,
    #                                 )
    # return image_annotated, annotated_line


def calculate_line(point, 
                   image_2, 
                   F):
    point = np.asarray([point[0], point[1], 1])
    line_2 = F @ point
    line_2 /= line_2[-1] 
    line_endpoints = [[0, 0], [0, 0]]
    # first point
    line_endpoints[0][1] = int(-line_2[2] / line_2[1])
    line_endpoints[1][0] = image_2.shape[1]-1
    line_endpoints[1][1] = int(-(line_2[2]+line_endpoints[1][0]*line_2[0]) / line_2[1])
    return line_endpoints
