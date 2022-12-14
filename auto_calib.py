import numpy as np
import cv2
import open3d as o3d
import json
from project_pcd_to_image import make_transformation, to_homogenous
import os
from scipy.spatial.transform import Rotation 
from scipy.optimize import minimize, fmin_powell
import time
from utils import to_homogenous, plot_pcd_on_image
from tqdm import tqdm
import argparse


def make_transformation_from_euler(euler_angles,
                                   trans):
    rot = Rotation.from_euler('zyx', 
                              euler_angles.tolist(), 
                              degrees=True)
    T = np.concatenate((rot.as_matrix(),
                        trans.reshape(3, 1)), axis=1)
    T = np.concatenate((T, np.asarray([0, 0, 0, 1]).reshape(-1, 4)), 0)
    return T                               

def consistency_label_cost(label_pcd,
                           label_img,
                           epsilon=1e-3):
    C = np.zeros((label_pcd.shape[0], 1))
    req_idx = (label_pcd != label_img)
    C[req_idx] = 1
    return C

def nearest_pixel(image_semantic_mask,
                  pcd_semantic_label,
                  pcd_projection):
    val_idxs = np.argwhere(image_semantic_mask == pcd_semantic_label)
    manhattan_distance = np.abs(val_idxs[:, 0] - pcd_projection[0]) + \
                         np.abs(val_idxs[:, 1] - pcd_projection[1])
    min_idx = np.argmin(manhattan_distance, 0)
    return val_idxs[min_idx]

def batched_nearest_pixel(pcd_projection,
                          img_val_idxs):
    '''
    image_semantic_mask = h*w*1
    pcd_semantic_label = scalar int
    pcd_projection = N*2
    '''
    # print(image_semantic_mask)
    if len(pcd_projection.shape) == 2:
        pcd_projection = np.reshape(pcd_projection,  
                                    (-1, 1, pcd_projection.shape[-1]))
    img_val_idxs = img_val_idxs.reshape(1, -1, 2)
    # print("pcds:\n ", pcd_projection)
    # print("val idxs:\n ", val_idxs)
    # print(pcd_projection.shape, 
        #   val_idxs.shape)
    # print(pcd_projection.shape, img_val_idxs.shape)
    pcd_projection = np.int16(pcd_projection)
    img_val_idxs = np.int16(img_val_idxs)
    manhattan_distances = np.abs(pcd_projection - img_val_idxs)
    # print("abs values:")
    manhattan_distances = np.sum(manhattan_distances, -1)
    # print(manhattan_distances, manhattan_distances.shape)
    # min_args = np.argmin(manhattan_distances, axis=1)
    min_dists = np.min(manhattan_distances, axis=1)
    # min_idxs = val_idxs.reshape(-1, 2)[min_args, :]
    # print(min_args.shape, min_args)
    # print(val_idxs.shape, min_idxs)
    # for i in range(min_args.shape[0]):
    #     print(f"for pcd {pcd_projection.reshape(-1, 2)[i]}" +\
    #           f"dist: {min_dists[i]}, at index of {min_idxs[i]}")
    # print(min_dists, )
    return min_dists

def cost_for_class(class_no,
                   pcd,
                   pcd_labels,
                   pcd_projection,
                   pcd_preds,
                   image_valid_idx):
    '''
    class_no = scalar
    pcd = N*3,
    pcd_labels = N*1,
    pcd_projection = N*2,
    pcd_preds = N*1,
    image_valid_idxs = M*2
    '''
    # print(pcd_labels.shape, pcd_preds.shape)
    consistent_cost = consistency_label_cost(pcd_labels.reshape(-1, 1).astype(np.int16), 
                                             pcd_preds.astype(np.int16)) #N*1
    # print("hi1")
    non_consistent_indxs = (consistent_cost != 0)
    inconsistent_loss = np.zeros((pcd_labels.shape[0], 1)).astype(np.float16) 
    # print(pcd_projection.shape, non_consistent_indxs.shape, consistent_cost.shape)
    inconsistent_loss_non_consistent_idx = batched_nearest_pixel(
                                              pcd_projection[non_consistent_indxs[:, 0], :],
                                              image_valid_idx).astype(np.float16) #N*1
    # print("hi2")
    inconsistent_loss[non_consistent_indxs] = inconsistent_loss_non_consistent_idx
    # print("inco", inconsistent_loss_non_consistent_idx.shape, non_consistent_indxs.shape)
    total_loss = inconsistent_loss.reshape(-1) * \
                 (np.linalg.norm(pcd, axis=1)**2).reshape(-1) * \
                 consistent_cost.reshape(-1)
    return np.mean(total_loss)

def rotation_mat2_euler(rot):
    r = Rotation.from_matrix(rot[:3, :3])
    return r.as_euler('zyx', degrees=True)

def add_noise(rot, mean=0, std=1, multiply=2):
    rot_new = rot + np.random.randn(rot.shape[0])*multiply
    return rot_new

def optimization_function(opt_variables,
                          lidar2world_trans,
                          world2cam,
                          cam_K,
                          pcd_all,
                          semantic_image):
    start_time = time.time()
    if opt_variables.shape[0] == 3:
        lidar2world = make_transformation_from_euler(opt_variables, lidar2world_trans)
        print("fi")
    else:
        lidar2world = make_transformation_from_euler(opt_variables[:3], opt_variables[3:])
    pcd = pcd_all[:, :3]
    labels = np.int16(pcd_all[:, -1])
    # colors = np.int16(cm.rainbow(np.linspace(0, 1, 25))*255)
    pcd_homogenous = to_homogenous(pcd)
    pcd_world = (lidar2world@pcd_homogenous.T).T
    
    ## check this
    pcd_camera = (world2cam@pcd_world.T).T
    pcd_camera /= pcd_camera[:, -1].reshape(-1, 1)
    pcd_camera = pcd_camera[:, :3]
    
    pcd_image = (cam_K@pcd_camera.T).T
    pcd_image /= pcd_image[:, -1].reshape(-1, 1)
    unique_labels = np.unique(labels)
    valid_pcd_image_idx = (pcd_image[:, 1] >= 0) & \
                          (pcd_image[:, 0] < semantic_image.shape[1]) & \
                          (pcd_image[:, 0] >= 0) & \
                          (pcd_image[:, 1] < semantic_image.shape[0]) 
    valid_pcd_image = np.int16(pcd_image[valid_pcd_image_idx, :])
    pcd_image_preds = np.ones((pcd_image.shape[0], 1))*-1 #N*1
    # print(pcd_image_preds.shape, semantic_image.shape)
    pcd_image_preds[valid_pcd_image_idx, 0] = semantic_image[valid_pcd_image[:, 1],
                                                             valid_pcd_image[:, 0], 
                                                             0] 
    semantic_int_image = semantic_image[:, :, 0]
    # print(pcd_image_preds.shape, labels.shape)
    total_cost = 0
    for class_count, class_no in enumerate(unique_labels):
        curr_class_idx = (labels == class_no)  
        curr_class_pcd = pcd[curr_class_idx, :]
        curr_class_labels_gt = labels[curr_class_idx] 
        curr_class_img = pcd_image[curr_class_idx, :]
        curr_class_img_preds = pcd_image_preds[curr_class_idx]
        image_valid_idxs = np.argwhere((semantic_int_image == class_no))
        # print(image_valid_idxs.shape)
        # print(curr_class_pcd.shape, curr_class_img_preds.shape)
        curr_cost = cost_for_class(class_no,
                                   curr_class_pcd,
                                   curr_class_labels_gt,
                                   np.int16(curr_class_img[:, [1, 0]]),
                                   curr_class_img_preds,
                                   image_valid_idxs)
        total_cost += curr_cost
    mean_cost = total_cost/(class_count+1)
    print(f"curr_cost: {mean_cost}")
    print(f"time taken {time.time() - start_time}")
    return mean_cost
    # print(f"total cost {mean_cost}")

def main():
    ####
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_trans', action='store_false')
    args = parser.parse_args()

    common_folder_name = "/media/akshay/Data/16822/data/"
    filename = "521835264021"
    pcd_folder = os.path.join(common_folder_name, "pcd_new")
    image_folder = os.path.join(common_folder_name, "semantic_image_new")
    json_folder_name = "./data/tf_info.json"
    json_folder_name_cam_info = "./data/sem_camera_info.json"
    cam_1_name = "wide_angle"
    sem_cam_name = "semantic_segmentation_front"
    lidar_name = "semantic_lidar"
    with open(json_folder_name, "r") as f:
        transforms = json.load(f)
    with open(json_folder_name_cam_info, "r") as f:
        cam_info = json.load(f)
    cam_R = transforms[f"ego_vehicle/{sem_cam_name}_R"]
    cam_t = np.asarray(transforms[f"ego_vehicle/{sem_cam_name}_t"])
    lidar_R = transforms[f"ego_vehicle/{lidar_name}_R"]
    lidar_t = np.asarray(transforms[f"ego_vehicle/{lidar_name}_t"])
    cam_K = np.asarray(cam_info["K1"]).reshape(-1, 3)
    lidar2world = make_transformation(lidar_R,
                                      lidar_t)
    world2cam = np.linalg.inv(make_transformation(cam_R,
                                                  cam_t))
    
    # noisy_lidar2world_angles = lidar2world_angles
    # print(f"Unnoisy\n{lidar2world_angles}")
    # print(f"Noisy\n{noisy_lidar2world_angles}")

        ###
    semantic_refined_folder_name = os.path.join(common_folder_name, 
                                                "refined_semantic_image")
    refined_lidar_folder_name = os.path.join(common_folder_name,
                                             "refined_pcd")
    color_map_folder_name = os.path.join(common_folder_name,
                                         "refined_colormap")
    semantic_refined_ints_folder_name = os.path.join(common_folder_name,
                                                     "refined_semantic_ints_image")
    wide_angle_image_folder_name = os.path.join(common_folder_name,
                                                "wide_angle_image_new")
    before_folder_name = "./results/before/"
    after_folder_name = "./results/after/"
    info_folder_name = "./results/info/"

    for filename in tqdm(sorted(os.listdir(semantic_refined_ints_folder_name)),
                         total=len(os.listdir(semantic_refined_ints_folder_name))):
        lidar2world_angles = np.asarray(rotation_mat2_euler(lidar2world))
        noisy_lidar2world_angles = add_noise(lidar2world_angles)
        if args.is_trans:
            noisy_lidar2world_trans = add_noise(lidar_t, multiply=0.01)
            optimization_variables = np.concatenate((noisy_lidar2world_angles.reshape(-1),
                                                     noisy_lidar2world_trans.reshape(-1)))
            print(f"Noisy variables {optimization_variables}")
            print(f"Unnoisy {lidar2world_angles}, {lidar_t}")
        else:
            noisy_lidar2world_trans = lidar_t
            optimization_variables = noisy_lidar2world_angles.reshape(-1)
            print(f"Noisy variables {optimization_variables}")

        filename = filename.split('.png')[0]
        if os.path.exists(os.path.join(before_folder_name,
                                       f'{filename}.png')):
            print(f"already there {filename}")
            continue

        pcd_all = np.loadtxt(os.path.join(refined_lidar_folder_name,
                                          f'{filename}.txt'))
        image_labels = cv2.imread(os.path.join(semantic_refined_ints_folder_name,
                                              f'{filename}.png'))
        
        wide_angle_image = cv2.imread(os.path.join(wide_angle_image_folder_name,
                                                   f"{filename}.png"))
        if wide_angle_image is None:
            continue

        lidar2world_noisy = make_transformation_from_euler(noisy_lidar2world_angles, 
                                                           noisy_lidar2world_trans)
        marked_image_noisy = plot_pcd_on_image(wide_angle_image,
                                               pcd_all,
                                               lidar2world_noisy,
                                               world2cam,
                                               cam_K)
        est_p = fmin_powell(optimization_function, 
                            optimization_variables, 
                            args=(lidar_t,
                                  world2cam,
                                  cam_K,
                                  pcd_all,
                                  image_labels),
                            maxiter=10)
        print(f"Before {optimization_variables}, After {est_p}")
        lidar2world_after_opt = make_transformation_from_euler(
                                            np.asarray(est_p).reshape(-1)[:3], 
                                            np.asarray(est_p).reshape(-1)[3:])
        marked_image_after_opt = plot_pcd_on_image(wide_angle_image,
                                               pcd_all,
                                               lidar2world_after_opt,
                                               world2cam,
                                               cam_K)
        # cv2.imshow("pcd_to_image", marked_image)
        # cv2.imshow("pcd_noisy", marked_image_noisy)
        # cv2.imshow("after opt", marked_image_after_opt)
        # cv2.waitKey(0)
        results_info = {}
        results_info['initial_angles'] = optimization_variables[:3].tolist()
        results_info['initial_trans'] = optimization_variables[3:].tolist()
        results_info['final_angles'] = np.asarray(est_p).reshape(-1)[:3].tolist()
        results_info['final_trans'] = np.asarray(est_p).reshape(-1)[3:].tolist()
        with open(os.path.join(info_folder_name, 
                               f"{filename}.json"), "w") as f:
            json.dump(results_info, f)

        cv2.imwrite(os.path.join(before_folder_name,
                                 f"{filename}.png"), marked_image_noisy)
        cv2.imwrite(os.path.join(after_folder_name,
                                 f"{filename}.png"), marked_image_after_opt)
        # break
    ###

    ####
    pcd = np.random.randn(30_000, 3).astype(np.float16)
    pcd_labels = np.random.randint(0, 10, (30_000, 1)).astype(np.int8)
    img_labels = np.random.randint(0, 10, (1000, 100)).astype(np.int8)
    pcd_projection = np.random.randint(0, 10, (30000, 2)).astype(np.int8)
    print("hi")
    # cost_for_class(1, pcd, pcd_labels, pcd_projection, img_labels)

if __name__=='__main__':
    main()