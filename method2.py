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
import time
from reconstruction import extract_features, make_transformation,\
                           to_homogenous, plot_points, draw_epipolar_lines
import torch
from torch_scatter import scatter_min
import copy
import argparse
from icp import icp as icp_new


def transform_points(points_1,
                     T):
    points_n = (T@to_homogenous(points_1).T).T
    points_n /= points_n[:, -1].reshape(-1, 1)
    return points_n[:, :3]

def transform_and_make_o3d(points_1,
                           points_2,
                           pcd1,
                           pcd2,
                           T):
    points_1_copy, points_2_copy = np.copy(points_1), np.copy(points_2)
    points_1_copy = (T@to_homogenous(points_1_copy).T).T
    points_1_copy /= points_1_copy[:, -1].reshape(-1, 1)
    pcd1.points = o3d.utility.Vector3dVector(points_1_copy[:, :3])
    pcd2.points = o3d.utility.Vector3dVector(points_2_copy)
    return pcd1, pcd2

def invert_rotation(quat=[0.0,
                          -0.08715574274765817,
                          0.0,
                          0.9961946980917455,]):
    r = R.from_quat(quat)
    rot_mat = r.as_matrix()
    return rot_mat

def ground_removal(pcd, thresh=1):
    mask = (pcd[:, 2] >= thresh)
    return mask

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def icp(pcd1Numpy,
        pcd2Numpy,
        threshold=5):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pcd1Numpy)
    target.points = o3d.utility.Vector3dVector(pcd2Numpy)
    initTransform = np.asarray([[1.0,0.0,0.0,0.0],
                                [0.0,1.0,0.0,0.0],
                                [0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,1.0]])

    # draw_registration_result(source, target, initTransform)

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, initTransform)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initTransform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(type(reg_p2p))
    print(reg_p2p.inlier_rmse)

    # print("Transformation is:")
    # print(reg_p2p.transformation)
    return reg_p2p.transformation, reg_p2p.inlier_rmse


def closed_loop_svd(pts1,
                    pts2):
    print(f"Got {pts1.shape[0]} matches")
    pts1 = pts1.reshape(-1, 3)
    pts2 = pts2.reshape(-1, 3) 
    centroid_1 = np.mean(pts1, 0)
    centroid_2 = np.mean(pts2, 0)
    print(pts1.shape, centroid_1.shape)
    pts1_norm = pts1 - centroid_1.reshape(1, -1)
    pts2_norm = pts2 - centroid_2.reshape(1, -1)
    H = (pts1_norm.T)@pts2_norm
    U, S, VT = np.linalg.svd(H)
    Rot = VT.T@U.T

    if np.linalg.det(Rot) < 0:
        VT[2, :] *= -1
        Rot = VT.T@U.T
    t = -Rot@centroid_1.reshape(3, -1) + centroid_2.reshape(3, -1)
    return Rot, t

def scatter_minimize(pcd_image_int_unique_invs, 
                     pcd_raw_filtered, 
                     pcd_cam):
    pcd_image_int_unique_invs_tensor = torch.from_numpy(pcd_image_int_unique_invs).type(torch.LongTensor)
    pcd_cam_tensor = torch.from_numpy(pcd_cam)
    pcd_image_int_unique_invs_tensor = pcd_image_int_unique_invs_tensor.reshape(1, -1)
    pcd_cam_tensor_z = pcd_cam_tensor[:, 2].reshape(1, -1)
    mins, argmin = scatter_min(pcd_cam_tensor_z,
                               pcd_image_int_unique_invs_tensor,
                               dim=1)
    # print(mins.shape, argmin.shape)
    return pcd_raw_filtered[argmin[0, :]]

def intersect2d(arr1, arr2):
    '''
    arr2 comes from pcd projection
    '''
    arr1_set = dict()
    print(arr1.shape, arr2.shape)
    for i, arr in enumerate(arr1):
        arr1_set[f"{arr1[i, 0]},{arr1[i, 1]}"] = i

    common_idx_arr1 = np.zeros((0, 1))
    common_idx_arr2 = np.zeros((0, 1))
    common_pts = np.zeros((0, 2))
    for i, arr in enumerate(arr2):
        arr_str = f"{arr2[i, 0]},{arr2[i, 1]}"
        if arr_str in arr1_set:
            common_idx_arr1 = np.concatenate((common_idx_arr1,
                                              np.asarray([arr1_set[arr_str]]).reshape(-1, 1)), 0)
            common_idx_arr2 = np.concatenate((common_idx_arr2, 
                                              np.asarray([i]).reshape(-1, 1)), 0)
            common_pts = np.concatenate((common_pts, 
                                         arr.reshape(-1, 2)), 0)
    return common_idx_arr1, common_idx_arr2, common_pts

def find_lidar_point_from_keypoint(keypoints,
                                   img_points):
    keypoint_idx, img_points_idx, common_pts = intersect2d(keypoints,
                                                           img_points)
    return keypoint_idx, img_points_idx, common_pts

def project_lidar2cam(lidar2world,
                      world2cam,
                      cam_K,
                      pcd,
                      image):
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(pcd[:, :3])
    # o3d.visualization.draw_geometries([pcd2])
    pcd_homogenous = to_homogenous(pcd)
    pcd_world = (lidar2world@pcd_homogenous.T).T
    
    ## check this
    pcd_camera = (world2cam@pcd_world.T).T
    pcd_camera /= pcd_camera[:, -1].reshape(-1, 1)
    pcd_camera = pcd_camera[:, :3]
    
    ## check this
    min_z = np.min(pcd_camera[:, 2])
    max_z = np.max(pcd_camera[:, 2])
    scale = 255 / (max_z - min_z)
    depth = pcd_camera[:, 2]
    depth_color = cv2.applyColorMap(np.uint8(scale*(depth-min_z)),
                                    cv2.COLORMAP_JET)
    pcd_image = (cam_K@pcd_camera.T).T
    pcd_image /= pcd_image[:, -1].reshape(-1, 1)
    marked_image = np.copy(image)
    # for i in range(pcd_camera.shape[0]):
    #     color = depth_color[i, 0, :].astype(np.uint8).tolist()
    #     # color_set.add(str(color))
    #     # print(pcd_image[i, 1], pcd_image[i, 0])

    #     (pcd_image[:, 1] >= 0) & \
    #                 (pcd_image[:, 0] >= 0) & \
    #                 (pcd_image[:, 0] < image.shape[1]) & \
    #                 (pcd_image[:, 1] < image.shape[0]) & \
    #                 (pcd_camera[:, 2] >= 0)

    #     if pcd_image[i, 1] < 0 or pcd_image[i, 0] < 0 or\
    #        pcd_image[i, 1] >= image.shape[0] or \
    #        pcd_image[i, 0] >= image.shape[0] or \
    #        pcd_camera[i, 2] < 0:
    #        continue
    #     cv2.circle(marked_image, 
    #                (int(pcd_image[i, 0]), int(pcd_image[i, 1])),
    #                2,
    #                color, 
    #                -1,
    #                4,
    #                0)
    # cv2.imshow("proj", marked_image)
    # cv2.waitKey(0)

    mask = (pcd_image[:, 0] >= 0) &\
           (pcd_image[:, 1] >= 0) &\
           (pcd_image[:, 0] <= image.shape[1]) &\
           (pcd_image[:, 1] <= image.shape[0]) &\
           (pcd_camera[:, 2] >= 0)
    pcd_image = pcd_image[mask]
    pcd_raw_filtered = pcd[mask]
    # marked_image = np.copy(image)
    return pcd_image, pcd_raw_filtered, pcd_camera[mask]

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
        pts1_req, pts2_req = extract_features(img1, 
                                              img2, 
                                              is_sift=False,
                                              sift=sift,
                                              orb=orb,
                                              bf=bf)
        pts1_req = np.asarray(pts1_req)
        pts2_req = np.asarray(pts2_req)
        # F = cv2.findFundamentalMat(pts1_req,
        #                            pts2_req, 
        #                            cv2.FM_RANSAC)
        # print("inliers: ", np.sum(F[1])/F[1].shape[0])
        # req_points = (F[1] == 1).reshape(-1)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_superpoint', action='store_true')
    parser.add_argument('--image_folder', default="epipolar_ordered")
    parser.add_argument('--lidar_folder', default="pcd_ordered")
    parser.add_argument('--superpoint_folder', default="superpoint_old_1")
    parser.add_argument('--sample_no', default=1)
    parser.add_argument('--trajectory_folder_ours', default="./results/trajectory_5_ours")
    parser.add_argument('--trajectory_folder_theirs', default="./results/trajectory_5_theirs")
    parser.add_argument('--is_report', action='store_true')
    args = parser.parse_args()

    results_json_file = "./results/info_5_frames_closed_form.json"
    results_info = {}

    report_foldername = "./results/for_report"
    common_folder_name = "/media/akshay/Data/16822/data/"
    image1_foldername = os.path.join(common_folder_name, 
                                     args.image_folder)
    lidar_foldername = os.path.join(common_folder_name, 
                                    args.lidar_folder)
    superpoint_foldername = os.path.join(common_folder_name,
                                         args.superpoint_folder)
    lidar_filenames = os.listdir(lidar_foldername)
    lidar_filenames_ints = [int(lidar_file.split('.')[0]) for lidar_file in lidar_filenames]
    lidar_filenames_ints.sort()
    lidar_filenames_strs = [str(lidar_int) for lidar_int in lidar_filenames_ints]
    ###
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
    world2cam = np.linalg.inv(make_transformation(cam_R,
                                                  cam_t))
    ####
    # json_folder_name = "./data/tf_info.json"
    # json_folder_name_cam_info = "./data/sem_camera_info.json"
    # cam_1_name = "wide_angle"
    # sem_cam_name = "semantic_segmentation_front"
    # lidar_name = "semantic_lidar"
    # with open(json_folder_name, "r") as f:
    #     transforms = json.load(f)
    # with open(json_folder_name_cam_info, "r") as f:
    #     cam_info = json.load(f)
    # cam_R = transforms[f"ego_vehicle/{sem_cam_name}_R"]
    # cam_t = np.asarray(transforms[f"ego_vehicle/{sem_cam_name}_t"])
    # lidar_R = transforms[f"ego_vehicle/{lidar_name}_R"]
    # lidar_t = np.asarray(transforms[f"ego_vehicle/{lidar_name}_t"])
    # cam_K = np.asarray(cam_info["K1"]).reshape(-1, 3)
    # lidar2world = make_transformation(lidar_R,
    #                                   lidar_t)
    # world2cam = np.linalg.inv(make_transformation(cam_R,
    #                                               cam_t))
    sample_no = int(args.sample_no)
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    sample_counter = 0
    time_our_algo = 0.0
    time_icp = 0.0
    number_of_frames = 0
    final_losses_theirs = 0.0
    final_losses_ours = 0.0 

    for i in tqdm(range(0, len(lidar_filenames_ints)-1),
                  total=len(os.listdir(superpoint_foldername))):
        number_of_frames += 1
        pcd_np_1 = np.load(os.path.join(lidar_foldername,
                                        lidar_filenames_strs[i]+".npy"))[:, :3]
        pcd_color_1 = np.tile(np.asarray([1, 0.706, 0]).reshape(1, 3), 
                              (pcd_np_1.shape[0], 1))
        pcd_np_2 = np.load(os.path.join(lidar_foldername,
                                        lidar_filenames_strs[i+sample_no]+".npy"))[:, :3]  
        pcd_color_2 = np.tile(np.asarray([0, 0.651, 0.929] ).reshape(1, 3), 
                              (pcd_np_2.shape[0], 1))  
        pcd_np_1_org = np.copy(pcd_np_1)
        pcd_np_2_org = np.copy(pcd_np_2)

        inv_rot = invert_rotation()
        pcd_np_1_rot = (inv_rot@pcd_np_1[:, :3].T).T
        pcd_np_2_rot = (inv_rot@pcd_np_2[:, :3].T).T
        pcd_np_1_gr_ind = ground_removal(pcd_np_1_rot, -2)
        pcd_np_2_gr_ind = ground_removal(pcd_np_2_rot, -2)
        pcd_np_1_gr = pcd_np_1_rot[pcd_np_1_gr_ind]
        pcd_np_2_gr = pcd_np_2_rot[pcd_np_2_gr_ind]
        pcd_np_1 = pcd_np_1[pcd_np_1_gr_ind]
        pcd_np_2 = pcd_np_2[pcd_np_2_gr_ind]
        pcd_color_1 = pcd_color_1[pcd_np_1_gr_ind]
        pcd_color_2 = pcd_color_2[pcd_np_2_gr_ind]

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd_np_1)
        pcd1.colors = o3d.utility.Vector3dVector(pcd_color_1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd_np_2)
        pcd2.colors = o3d.utility.Vector3dVector(pcd_color_2)                      
        if not os.path.exists(os.path.join(image1_foldername, 
                                       lidar_filenames_strs[i] + ".png")) or \
           not os.path.exists(os.path.join(image1_foldername,
                                          lidar_filenames_strs[i+sample_no] + ".png")):
            print(f'skipped filename {lidar_filenames[i]}')
            continue
        image_1 = cv2.imread(os.path.join(image1_foldername, 
                                          lidar_filenames_strs[i] + ".png"))
        image_2 = cv2.imread(os.path.join(image1_foldername,
                                          lidar_filenames_strs[i+sample_no] + ".png"))
        image1_with_keypoints = np.copy(image_1)
        image2_with_keypoints = np.copy(image_2)
        if not args.is_superpoint:
            pts1_req, pts2_req = extract_features(image_1, 
                                                  image_2, 
                                                  is_sift=True,
                                                  sift=sift,
                                                  orb=orb,
                                                  bf=bf)
            if args.is_report:
                for pt1_req in pts1_req:
                    image1_with_keypoints = plot_points(image1_with_keypoints,
                                                            pt1_req,
                                                            radius=4,
                                                            thickness=1)
                for pt2_req in pts2_req:
                    image2_with_keypoints = plot_points(image2_with_keypoints,
                                                        pt2_req,
                                                        radius=4,
                                                        thickness=1)
                cv2.imshow("sift", image1_with_keypoints)
                sift_concat = cv2.hconcat([image1_with_keypoints, image2_with_keypoints])
                cv2.imwrite(os.path.join(report_foldername, f"{sample_no}_{sample_counter}_sift.png"), 
                            image1_with_keypoints)
                cv2.imwrite(os.path.join(report_foldername, 
                                         f"{sample_no}_{sample_counter}_sift_combined_with_{len(pts1_req)}_{len(pts2_req)}_kps.png"), 
                            sift_concat)

                img1_2_keypoints = cv2.hconcat([image1_with_keypoints,
                                                image2_with_keypoints])
                for point1, point2 in zip(pts1_req, pts2_req):
                    img1_2_keypoints = draw_epipolar_lines(
                                                        img1_2_keypoints,
                                                        np.int16(point1),
                                                        np.int16(point2),
                                                        thickness=1)
                cv2.imshow("sift", img1_2_keypoints)
                cv2.imwrite(os.path.join(
                            report_foldername, 
                            f"{sample_no}_{sample_counter}_matches_sift.png"), 
                            img1_2_keypoints)
                exit()
        else:
            corres_filename = f"{sample_counter}_{sample_counter+sample_no}_matches.npz"
            print(corres_filename)
            sample_counter += sample_no
            superpoint_file = os.path.join(
                                superpoint_foldername,
                                corres_filename)
            superpoint_corres = np.load(superpoint_file)
            point1 = superpoint_corres['keypoints0']
            point2 = superpoint_corres['keypoints1']
            matches = superpoint_corres['matches']
            confidence = superpoint_corres['match_confidence']
            valid = (matches > -1) & \
                    (confidence > 0.9)
            pts1_req = point1[valid]
            pts2_req = point2[matches[valid]]
            if args.is_report:
                for pt1_req in pts1_req:
                    image1_with_keypoints = plot_points(image1_with_keypoints,
                                                            pt1_req,
                                                            radius=4,
                                                            thickness=1)
                for pt2_req in pts2_req:
                    image2_with_keypoints = plot_points(image2_with_keypoints,
                                                        pt2_req,
                                                        radius=4,
                                                        thickness=1)
                cv2.imshow("super", image1_with_keypoints)
                sift_concat = cv2.hconcat([image1_with_keypoints, image2_with_keypoints])
                cv2.imwrite(os.path.join(report_foldername, f"{sample_no}_{sample_counter}_superpoint.png"), 
                            image1_with_keypoints)
                cv2.imwrite(os.path.join(
                                report_foldername, 
                                f"{sample_no}_{sample_counter}_superpoint_combined_with_{len(pts1_req)}_{len(pts2_req)}_kps.png"), 
                            sift_concat)
                img1_2_keypoints = cv2.hconcat([image1_with_keypoints,
                                                image2_with_keypoints])
                for point1, point2 in zip(pts1_req, pts2_req):
                    img1_2_keypoints = draw_epipolar_lines(
                                                        img1_2_keypoints,
                                                        np.int16(point1),
                                                        np.int16(point2),
                                                        thickness=1)
                cv2.imshow("super", img1_2_keypoints)
                cv2.imwrite(os.path.join(
                            report_foldername, 
                            f"{sample_no}_{sample_counter}_matches_super.png"), 
                            img1_2_keypoints)
                cv2.waitKey(0)
                # exit()

        pts1_req = np.asarray(pts1_req)
        pts2_req = np.asarray(pts2_req)

        pts1_req_int = np.round(pts1_req).astype(np.int16)
        pts2_req_int = np.round(pts2_req).astype(np.int16)
        pcd_image1, pcd_raw_filtered_1, pcd_cam1 = project_lidar2cam(
                                                        lidar2world,
                                                        world2cam,
                                                        cam_K,
                                                        pcd_np_1_org,
                                                        image_1)
        #
        # pcd_image2: N*1
        # pcd_raw_filtered_2: N*3 
        # pcd_cam2: N*3 in cam coordinates
        #
        pcd_image2, pcd_raw_filtered_2, pcd_cam2 = project_lidar2cam(
                                                        lidar2world,
                                                        world2cam,
                                                        cam_K,
                                                        pcd_np_2_org,
                                                        image_2)

        pcd_image1_int = np.int16(pcd_image1[:, :2])
        pcd_image1_int_unique_values, \
        pcd_image1_int_unique_idxs, \
        pcd_image1_int_unique_invs, \
        pcd_image1_int_unique_counts = np.unique(pcd_image1_int,
                                                 axis=0,
                                                 return_index=True,
                                                 return_inverse=True, 
                                                 return_counts=True)
        pcd_image2_int = np.int16(pcd_image2[:, :2])
        pcd_image2_int_unique_values, \
        pcd_image2_int_unique_idxs, \
        pcd_image2_int_unique_invs, \
        pcd_image2_int_unique_counts = np.unique(pcd_image2_int,
                                                 axis=0,
                                                 return_index=True,
                                                 return_inverse=True, 
                                                 return_counts=True)
        pcd1_raw_unique = scatter_minimize(pcd_image1_int_unique_invs, 
                                           pcd_raw_filtered_1, 
                                           pcd_cam1)
        pcd2_raw_unique = scatter_minimize(pcd_image2_int_unique_invs,
                                           pcd_raw_filtered_2,
                                           pcd_cam2)

        common_idx_keypoints_1,\
        common_pts_cam_points_idx_1,\
        common_pts_img1 = intersect2d(pts1_req_int[:, :],
                                      pcd_image1_int_unique_values)
        
        # print(np.max(common_idx_keypoints_1), np.min(common_idx_keypoints_1))

        
        common_idx_keypoints_2,\
        common_pts_cam_points_idx_2, \
        common_pts_img2 = intersect2d(pts2_req_int[:, :],
                                      pcd_image2_int_unique_values)
        common_idx_keypoints,\
        common_idx_kp1, \
        common_idx_kp2 = np.intersect1d(common_idx_keypoints_1,
                                        common_idx_keypoints_2,
                                        return_indices=True)
  
        common_pts_img2_common = common_pts_cam_points_idx_2[common_idx_kp2].astype(np.int16)
        common_pts_img1_common = common_pts_cam_points_idx_1[common_idx_kp1].astype(np.int16)
        req_pcd1_raw = pcd1_raw_unique[common_pts_img1_common[:, 0], :]
        req_pcd2_raw = pcd2_raw_unique[common_pts_img2_common[:, 0], :]
        img1_final_keypoints = np.copy(image_1)
        img2_final_keypoints = np.copy(image_2)
        temp_keypoints_1 = pts1_req_int[np.int16(common_idx_keypoints_1), :] 
        temp_keypoints_2 = pts2_req_int[np.int16(common_idx_keypoints_2), :]
        img1_2_keypoints = cv2.hconcat([img1_final_keypoints,
                                        img2_final_keypoints])
        for point_no in range(common_idx_kp1.shape[0]):
            point1 = temp_keypoints_1[common_idx_kp1[point_no]]
            point2 = temp_keypoints_2[common_idx_kp2[point_no]]
            img1_2_keypoints = draw_epipolar_lines(img1_2_keypoints,
                                                   point1[0],
                                                   point2[0],
                                                   radius=4,
                                                   thickness=1)
 
        points_image_1 = np.copy(image_1)
        # print(np.unique(pcd_image1_int, 0), 
        #       pcd_image1_int.shape)
        for point_no, pts1_for_plot in enumerate(pcd_image1_int):
            points_image_1 = plot_points(points_image_1, 
                                         pts1_for_plot, 
                                         str(point_no))
        if args.is_report:
            cv2.imwrite(os.path.join(report_foldername,
                                    f"{sample_counter}_{sample_no}_projected.png"), 
                        points_image_1)
            cv2.imwrite(os.path.join(report_foldername,
                                    f"{sample_counter}_{sample_no}_final_kps.png"), 
                        img1_2_keypoints)
            cv2.imshow("plotted", points_image_1)
            cv2.imshow("final", img1_2_keypoints)
            cv2.waitKey(0)
        # break
            exit()
        pcd2.points = o3d.utility.Vector3dVector(pcd_np_2[:, :3])
        pcd1.points = o3d.utility.Vector3dVector(pcd_np_1[:, :3])
        pcd1.colors = o3d.utility.Vector3dVector(pcd_color_1)
        pcd2.colors = o3d.utility.Vector3dVector(pcd_color_2)
        print("initial")

        o3d.visualization.draw_geometries([pcd1, pcd2])
        print("Ours")
        our_start_time = time.time()
        Rot, t = closed_loop_svd(req_pcd1_raw, req_pcd2_raw)
        T_ours_start = np.concatenate((Rot, t.reshape(3, 1)), axis=1)
        T_ours_start = np.concatenate((T_ours_start, 
                                       np.asarray([0, 0, 0, 1]).reshape(1, -1)), 0)
        T_ours = T_ours_start
        
        our_elapsed_time = time.time() - our_start_time
        time_our_algo += our_elapsed_time
        np.save(os.path.join(args.trajectory_folder_ours, 
                             f"{number_of_frames}.npy"), T_ours)

        pcd1, pcd2 = transform_and_make_o3d(pcd_np_1_gr[:, :3],
                                            pcd_np_2_gr[:, :3],
                                            pcd1,
                                            pcd2,
                                            T_ours)
        o3d.visualization.draw_geometries([pcd1, pcd2])
        print("Ours 2")
        T_ours, first_error_ours, last_error_ours = icp_new(req_pcd1_raw, 
                                                            req_pcd2_raw,
                                                            max_iterations=10,
                                                            tolerance=0.0001,
                                                            init_pose=T_ours_start
                                                            )
        pcd1, pcd2 = transform_and_make_o3d(pcd_np_1_gr[:, :3],
                                            pcd_np_2_gr[:, :3],
                                            pcd1,
                                            pcd2,
                                            T_ours)
        o3d.visualization.draw_geometries([pcd1, pcd2])                                                    
        # print(f"Ours {first_error_ours}, {last_error_ours}")
        print("Theirs")
        their_time_start = time.time()
        # T_theirs, final_loss_theirs = icp(pcd_np_1_org, pcd_np_2_org, threshold=5)
        T_theirs, _, final_loss_theirs = icp_new(pcd_np_1_org,
                                                 pcd_np_2_org)
        their_elapsed_time = time.time() - their_time_start
        pcd1, pcd2 = transform_and_make_o3d(pcd_np_1_gr[:, :3],
                                            pcd_np_2_gr[:, :3],
                                            pcd1,
                                            pcd2,
                                            T_theirs)
        o3d.visualization.draw_geometries([pcd1, pcd2])
        time_icp += their_elapsed_time
        final_losses_theirs += final_loss_theirs
        np.save(os.path.join(args.trajectory_folder_theirs,
                            f"{number_of_frames}.npy"), T_theirs)
       # print("Open 3d ours")
        # T_ours_o3d = icp(req_pcd1_raw, req_pcd2_raw, threshold=50)
        ours_trans = transform_points(pcd_np_1_org[:, :3], T_ours)

        _, final_loss_ours, _ = icp_new(ours_trans, 
                                        pcd_np_2_org, 
                                        max_iterations=0)
        print(f"Open 3d ours whole cloud final loss={final_loss_ours}\n," 
              f"Their final loss={final_loss_theirs}")
        print(f"Ours time {our_elapsed_time}, their {their_elapsed_time}")  
        final_losses_ours += final_loss_ours
        # exit()
        # pcd1, pcd2 = transform_and_make_o3d(pcd_np_1_gr[:, :3],
        #                                     pcd_np_2_gr[:, :3],
        #                                     pcd1,
        #                                     pcd2,
        #                                     T)
        # print("Theirs")
        # o3d.visualization.draw_geometries([pcd1, pcd2])
     

        # pcd1, pcd2 = transform_and_make_o3d(pcd_np_1_gr[:, :3],
        #                                     pcd_np_2_gr[:, :3],
        #                                     pcd1,
        #                                     pcd2,
        #                                     T)
        # _, first_error, _ = icp_new(ours_trans, 
        #                             pcd_np_2, 
        #                             max_iterations=0)
        # print(f"First error ours {first_error}")
        
        # exit()
        # results_info['ours_loss'] = final_losses_ours / number_of_frames
        # results_info['icp_loss'] = final_losses_theirs / number_of_frames
        # results_info['ours_time'] = time_our_algo / number_of_frames
        # results_info['theirs_time'] = time_icp / number_of_frames
        # with open(results_json_file, "w") as f:
        #     json.dump(results_info, f)

if __name__ == "__main__":
    pcd = o3d.geometry.PointCloud()
    main()