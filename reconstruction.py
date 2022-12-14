import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from epipolar_correspondence import EpipolarCorrespondence
from utils import plot_epipolar, plot_epipolar_correspondence
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def plot_points(img,
                point, 
                txt=None,
                radius=1,
                is_text=False):
    
    img = cv2.circle(img, 
                     (int(point[0]), int(point[1])),
                     radius=radius,
                     color=(0, 0, 255),
                     thickness=-1)
    if is_text:
        img = cv2.putText(img,
                          text=txt,
                          org=(int(point[0]), int(point[1])),
                          fontScale=1,
                          color=(0, 255, 0),
                          thickness=1,
                          fontFace=cv2.LINE_AA,
                          )
    return img

def draw_epipolar_lines(img,
                        point1,
                        point2,
                        radius=2,
                        thickness=2):
    random_color = np.random.randint(0, 255, (3, )).tolist()
    img =  cv2.circle(img, 
                      (int(point1[0]), int(point1[1])),
                      radius=radius,
                      color=random_color,
                      thickness=-1)
    img = cv2.circle(img,
                     (int(img.shape[1]//2 + point2[0]), 
                      int(point2[1])),
                     radius=radius,
                     color=random_color,
                     thickness=-1)
    img = cv2.line(img,
                   (int(point1[0]), int(point1[1])),
                   (int(point1[0]+img.shape[1]//2), int(point2[1])),
                   color=random_color,
                   thickness=thickness)
    return img

def info_extrinsic(T):
    r = R.from_matrix(T[:3, :3])
    print("Rotation: ", r.as_euler('zyx', degrees=True))
    print("Translation: ", T[:, -1])

def make_transformation(rotation, t):
    # check for quarternion
    if np.asarray(rotation).shape[0] == 4:
        rotation = R.from_quat([rotation[1], 
                                rotation[2], 
                                rotation[3], 
                                rotation[0]]).as_matrix()
    trans = np.concatenate((rotation, 
                            np.asarray(t).reshape(3, 1)), axis=1)
    trans = np.concatenate((trans, 
                            np.asarray([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return trans

def to_homogenous(points):
    return np.concatenate((points, 
                           np.ones((points.shape[0], 1))), axis=1)

def filter_cloud(points, radius):
    mask = (points[:, 0]**2 + points[:, 1]**2 <= radius**2)
    return points[mask] 

def make_triangulation_constraints(x1, x2, P1, P2):
    x1_mat = np.asarray([[0, -1, x1[1]],
                         [1, 0, -x1[0]],
                         [-x1[1], x1[0], 0]])
    x2_mat = np.asarray([[0, -1, x2[1]],
                         [1, 0, -x2[0]],
                         [-x2[1], x2[0], 0]])
    const_1 = x1_mat@P1
    const_2 = x2_mat@P2
    return np.concatenate((const_1[:2], const_2[:2]), axis=0)

def invert_extrinsics(P):
    P = np.concatenate((P, np.asarray([0, 0, 0, 1]).reshape(1, 4)), 
                        axis=0)
    return np.linalg.inv(P)

def plot_3d(x):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection = '3d')
    sample_rate = 1
    ax.scatter(x[::sample_rate, 0], 
               x[::sample_rate, 1],
               x[::sample_rate, 2])
    plt.show()

def find_four_solutions(U, VT):
    # check the eigen value of SVD(E)
    W = np.asarray([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    u3 = U[-1, :]
    R1 = U@W@VT 
    t1 = u3
    R2 = U@W@VT
    t2 = -u3
    R3 = U@W.T@VT
    t3 = u3
    R4 = U@W.T@VT
    t4 = -u3
    P1_ext = np.concatenate((R1, t1.reshape(3, 1)), axis=1)
    P2_ext = np.concatenate((R2, t2.reshape(3, 1)), axis=1)
    P3_ext = np.concatenate((R3, t3.reshape(3, 1)), axis=1)
    P4_ext = np.concatenate((R4, t4.reshape(3, 1)), axis=1)

    return P1_ext, P2_ext, P3_ext, P4_ext


def extract_features(image_1,
                     image_2,
                     is_sift=True,
                     sift=None,
                     orb=None,
                     bf=None):
    if is_sift:
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)

        pts1_tot = cv2.KeyPoint_convert(keypoints_1)
        pts2_tot = cv2.KeyPoint_convert(keypoints_2)
        pts1_req = []
        pts2_req = []
        for match in matches:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            pts1_req.append(pts1_tot[img1_idx])
            pts2_req.append(pts2_tot[img2_idx])
        print(f"Matches found {len(matches)}")
        img3 = cv2.drawMatches(image_1, keypoints_1, 
                               image_2, keypoints_2, 
                               matches[:len(matches)], 
                               image_2, flags=2)

    else:
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        pts1_tot = cv2.KeyPoint_convert(keypoints_1)
        pts2_tot = cv2.KeyPoint_convert(keypoints_2)
        pts1_req = []
        pts2_req = []
        for match in matches:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            pts1_req.append(pts1_tot[img1_idx])
            pts2_req.append(pts2_tot[img2_idx])
    img3 = cv2.drawMatches(image_1, keypoints_1, 
                           image_2, keypoints_2, 
                           matches[:], 
                           image_2, flags=2)
    cv2.imshow("features", img3)
    cv2.waitKey(0)
    return pts1_req, pts2_req

def reconstruct(image_folder_name_1,
                image_folder_name_2,
                pcd_folder_name=".",
                cam_info_path="./data/camera_info.json",
                lidar_2_world=None,
                cam1_2_world=None):
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    with open(cam_info_path, "r") as f:
        cam_info = json.load(f)
    
    P1 = np.asarray(cam_info['P1']).reshape(3, 4)
    P2 = np.asarray(cam_info['P2']).reshape(3, 4)
    P1_ext = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)
    K = np.asarray(cam_info['K']).reshape(3, 3)
    # K[]
    corresp_finder = EpipolarCorrespondence()

    for filename in os.listdir(pcd_folder_name):
        if filename.endswith(".npy"):
            img_filename = filename.split(".npy")[0] + ".png"
            pcd_filename = filename
        print(img_filename)
        pcd_np = np.load(pcd_filename)
        color_red = np.asarray([1, 0, 0]).reshape(-1, 3)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
        pcd1.colors = o3d.utility.Vector3dVector(np.tile(color_red, 
                                                        (pcd_np.shape[0], 1)))
        # o3d.visualization.draw_geometries([pcd1])

        img1 = cv2.imread(os.path.join(image_folder_name_1, img_filename))
        img2 = cv2.imread(os.path.join(image_folder_name_2, img_filename))    
        pts1_req, pts2_req =extract_features(img1, 
                                             img2, 
                                             is_sift=True,
                                             sift=sift,
                                             orb=orb,
                                             bf=bf)
        pts1_req = np.asarray(pts1_req)
        pts2_req = np.asarray(pts2_req)
        F = cv2.findFundamentalMat(pts1_req,
                                   pts2_req, 
                                   cv2.FM_RANSAC)
        print("inliers: ", np.sum(F[1])/F[1].shape[0])
        req_points = (F[1] == 1).reshape(-1)
        pts1_req_inliers = pts1_req[req_points, :].reshape(-1, 2)
        pts2_req_inliers = pts2_req[req_points, :].reshape(-1, 2)
        points_image_1 = np.copy(img1)
        points_image_2 = np.copy(img2)
        for point_no, pts1_for_plot in enumerate(pts1_req_inliers):
            points_image_1 = plot_points(points_image_1, 
                                         pts1_for_plot, 
                                         str(point_no))
            points_image_2 = plot_points(points_image_2,
                                         pts2_req_inliers[point_no],
                                         str(point_no))
        cv2.imshow("match_image_1", points_image_1)
        cv2.imshow("match_image_2", points_image_2)
        cv2.waitKey(0)
        inlier_matches = np.arange(pts1_req_inliers.shape[0])
        # matches = cv2.drawMatches(img2, pts1_req_inliers, 
        #                           img1, pts2_req_inliers, 
        #                           inlier_matches, 
        #                           img2, flags=2)
        # cv2.imshow("matches", matches)
        # pts1_req_inliers[:, [0, 1]] = pts1_req_inliers[:, [1, 0]]
        # pts2_req_inliers[:, [0, 1]] = pts2_req_inliers[:, [1, 0]]

        print(pts1_req_inliers.shape)
        pts1_req_rev = np.copy(pts1_req)
        pts1_req_rev[:, [1, 0]] = pts1_req_rev[:, [0, 1]]
        pts2_req_rev = np.copy(pts2_req)
        pts2_req_rev[:, [1, 0]] = pts2_req_rev[:, [0, 1]]
        # plot_epipolar(img1, img2, F[0])
        # plot_epipolar_correspondence(img1, img2, F[0], corresp_finder)
        # break
        # print(F[0])
        E = K.T@F[0]@K
        U, S, VT = np.linalg.svd(E)
        S_new = np.diag([(S[0]+S[1])/2, (S[0]+S[1])/2, 0])
        E = U@S_new@VT
        U, _, VT = np.linalg.svd(E)
        P2s = find_four_solutions(U, VT)
        
        # cv2.imshow("matches", img1)
        print(pts1_req.max(axis=0), img1.shape)
        colors = img1[np.int32(pts1_req[:, 1]), 
                      np.int32(pts1_req[:, 0]), 
                      :]
        colors = np.float32(colors)/255
        cv2.waitKey(100)
        for P2_ext in P2s:
            P1 = K@P1_ext
            P2 = K@P2_ext
            print(P2_ext)
            info_extrinsic(P2_ext)
            P2_ext_inv = invert_extrinsics(P2_ext)            
            # print(pts1_req_rev.max(0), img1.shape)
            # print(pts1_req_inliers.shape, pts2_req_inliers.shape)
            X = cv2.triangulatePoints(P1, 
                                      P2, 
                                      pts1_req_inliers.T, 
                                      pts2_req_inliers.T).T
            X_in_second_cam = (P2_ext@X.T).T
            # X_in_second_cam /= X_in_second_cam[:, -1].reshape(-1, 1)
            X /= X[:, -1].reshape(-1, 1)
            print(f"percent {np.sum(X[:, 2] >= 0) / X.shape[0]}")
            print(f"percent {np.sum(X_in_second_cam[:, 2] >= 0) / X_in_second_cam.shape[0]}")
            # print(X.min(0), X.max(0), X_in_second_cam.min(0), X_in_second_cam.max(0))
            # print(X)
            X_filtered = filter_cloud(X, 1000000)
            # X_filtered = (cam1_2_world@X_filtered.T).T
            # X_filtered = (np.linalg.inv(lidar_2_world)@X_filtered.T).T
            X_filtered /= X_filtered[:, -1].reshape(-1, 1)
            print(X_filtered)
            # colors_filtered = colors[X[:,0]**2 + X[:, 1]**2 + X[:, 2]**2 <= 625]
            color_blue = np.asarray([0, 0, 1]).reshape(-1, 3)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(X_filtered[:, :3])
            pcd2.colors = o3d.utility.Vector3dVector(np.tile(color_blue, 
                                                     (X_filtered.shape[0], 1)))

            # o3d.visualization.draw_geometries([pcd2, pcd1])
            # break
            # plot_3d(X)        
        print(X.shape)

# def find_corres(img1, img2):


def main():
    json_folder_name = "./data/transforms.json"
    json_folder_name_cam_info = "./data/camera_info.json"
    with open(json_folder_name, "r") as f:
        transforms = json.load(f)
    with open(json_folder_name_cam_info, "r") as f:
        cam_info = json.load(f)
    cam_R = transforms["ego_vehicle/epipolar_one_R"]
    cam_t = np.asarray(transforms["ego_vehicle/epipolar_one_t"])
    lidar_R = transforms["ego_vehicle/lidar_custom_R"]
    lidar_t = np.asarray(transforms["ego_vehicle/lidar_custom_t"])
    cam_K = np.asarray(cam_info["K"]).reshape(-1, 3)
    lidar2world = make_transformation(lidar_R,
                                      lidar_t)
    cam1_2_world = make_transformation(cam_R,
                                       cam_t)
    image_path_1 = "./data/epipolar_1/"
    image_path_2 = "./data/epipolar_2/"
    reconstruct(image_path_1, 
                image_path_2,
                lidar_2_world=lidar2world,
                cam1_2_world=cam1_2_world)

if __name__ == '__main__':
    main()