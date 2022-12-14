import numpy as np
import cv2
import os
import json
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from matplotlib import cm
from tqdm import tqdm
from utils import to_homogenous


def str_to_color(color_str):
    ints = ints = color_str.split(',')
    color_np = np.asarray([int(ints[0]),   
                           int(ints[1]),
                           int(ints[2])]).reshape(-1, 3)
    return color_np

def dict_to_colors(final_dict):
    colors = np.zeros((25, 3), dtype=np.int16)
    for k in final_dict.keys():
        ints = final_dict[k].split(',')
        color_np = np.asarray([int(ints[0]),   
                               int(ints[1]),
                               int(ints[2])]).reshape(-1, 3)
        colors[int(k)] = color_np
    np.save("./data/color_map.npy", colors)
    return colors

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



def project_lidar2cam(lidar2world,
                      world2cam,
                      cam_K,
                      pcd,
                      image):
    print(pcd.shape, np.min(pcd, 0), np.max(pcd, 0))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd[:, :3])
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
    color_set = set()
    #cv2.imshow("a", image)
    for i in range(pcd_camera.shape[0]):
        color = depth_color[i, 0, :].astype(np.uint8).tolist()
        color_set.add(str(color))
        # print(pcd_image[i, 1], pcd_image[i, 0])
        if pcd_image[i, 1] < 0 or pcd_image[i, 0] < 0 or\
           pcd_image[i, 1] >= 800 or pcd_image[i, 1] >= 800 or \
           pcd_camera[i, 2] < 0:
           continue
        cv2.circle(marked_image, 
                   (int(pcd_image[i, 0]), int(pcd_image[i, 1])),
                   2,
                   color, 
                   -1,
                   4,
                   0)
    cv2.imshow("proj", marked_image)
    cv2.waitKey(10)
    
def project_lidar2cam_semantic(lidar2world,
                               world2cam,
                               cam_K,
                               pcd,
                               image,
                               labels,
                               filename,
                               semantic_refined_folder_name=None,
                               semantic_refined_ints_folder_name=None,
                               refined_lidar_folder_name=None,
                               color_map_folder_name=None):
    idx = np.argsort(labels, 0)
    pcd = pcd[idx, :]
    labels = labels[idx]
    # colors = np.int16(cm.rainbow(np.linspace(0, 1, 25))*255)
    pcd_homogenous = to_homogenous(pcd)
    pcd_world = (lidar2world@pcd_homogenous.T).T
    
    ## check this
    pcd_camera = (world2cam@pcd_world.T).T
    pcd_camera /= pcd_camera[:, -1].reshape(-1, 1)
    pcd_camera = pcd_camera[:, :3]
    
    pcd_image = (cam_K@pcd_camera.T).T
    pcd_image /= pcd_image[:, -1].reshape(-1, 1)
    marked_image = np.copy(image)
    #cv2.imshow("a", image)
    # lidar_cam_color_map = {{}}
    valid_points = 0
    curr_label = -1
    curr_dict = {}
    final_dict = {}
    in_image_mask = (pcd_image[:, 1] >= 0) & \
                    (pcd_image[:, 0] >= 0) & \
                    (pcd_image[:, 0] < image.shape[1]) & \
                    (pcd_image[:, 1] < image.shape[0]) & \
                    (pcd_camera[:, 2] >= 0)
    refined_pcd_camera = pcd_camera[in_image_mask]
    refined_pcd_image = pcd_image[in_image_mask]
    refined_labels = labels[in_image_mask]

    for i in range(refined_pcd_camera.shape[0]):
        if curr_label != -1 and int(refined_labels[i]) != curr_label:
            curr_max = -1
            max_col = ""
            for k in curr_dict:
                if curr_dict[k] > curr_max:
                    curr_max = curr_dict[k]
                    max_col = k
            final_dict[int(curr_label)] = max_col
            curr_dict = {}

        this_col = image[int(refined_pcd_image[i, 1]), 
                         int(refined_pcd_image[i, 0])]
        color_str = f"{this_col[0]},{this_col[1]},{this_col[2]}"
        if not color_str in curr_dict:
            curr_dict[color_str] = 1
        else:
            curr_dict[color_str] += 1
        curr_label = int(refined_labels[i])
        # color = colors[np.int16(refined_labels[i]), :3]
        # cv2.circle(marked_image, 
        #            (int(refined_pcd_image[i, 0]), int(refined_pcd_image[i, 1])),
        #            2,
        #            color.tolist(), 
        #            -1,
        #            4,
        #            0)
        valid_points += 1
    for k in curr_dict:
        if curr_dict[k] > curr_max:
            curr_max = curr_dict[k]
            max_col = k
    final_dict[int(curr_label)] = max_col
    curr_dict = {}

    sematic_result_image = np.zeros_like(image)
    sematic_result_image_ints = np.zeros((image.shape[0],
                                          image.shape[1], 
                                          1))
    for i in range(refined_pcd_camera.shape[0]):
        # print(labls[i])
        sematic_result_image[
                int(refined_pcd_image[i, 1]), 
                int(refined_pcd_image[i, 0])] = str_to_color(final_dict[int(refined_labels[i])])
        sematic_result_image_ints[
                int(refined_pcd_image[i, 1]), 
                int(refined_pcd_image[i, 0])] = int(refined_labels[i])
    
    # refined_semantic_pcd = np.concatenate(ref)
    # print(refined_labels.shape,
    #       refined_pcd_camera.shape,
    #       refined_pcd_image.shape)
    # print("for image")
    # print("non zero pixels ", np.sum(sematic_result_image_ints.reshape(-1) != 0))
    # unique_labels, counts = np.unique(sematic_result_image_ints.reshape(-1, 1), 0, return_counts=True)
    # print(unique_labels, counts)
    # print("for pcd")
    # unique_labels, counts = np.unique(refined_labels.reshape(-1, 1), 0, return_counts=True)
    # print(unique_labels, counts)
    pcd_refined_with_label = np.concatenate((pcd[in_image_mask],
                                             refined_labels.reshape(-1, 1)), 1)
    # cv2.imshow("semantic_res", sematic_result_image)
    # cv2.imshow("semantic_new", sematic_result_image_ints)
    color_map = dict_to_colors(final_dict) 
    cv2.imwrite(os.path.join(semantic_refined_folder_name, 
                             f"{filename}.png"), 
                sematic_result_image)
    cv2.imwrite(os.path.join(semantic_refined_ints_folder_name, 
                             f"{filename}.png"), 
                sematic_result_image_ints)
    np.savetxt(os.path.join(refined_lidar_folder_name,
                            f"{filename}.txt"),
               pcd_refined_with_label)
    np.savetxt(os.path.join(color_map_folder_name,
                            f"{filename}.txt"),
               color_map)
    # cv2.waitKey(10)

def main():
    #####
    common_folder_name = "/media/akshay/Data/16822/data/"
    filename = "521835264021"
    pcd_folder = os.path.join(common_folder_name, "pcd_new")
    image_folder = os.path.join(common_folder_name, "semantic_image_new")
    json_folder_name = "./data/tf_info.json"
    json_folder_name_cam_info = "./data/sem_camera_info.json"
    cam_1_name = "wide_angle"
    sem_cam_name = "semantic_segmentation_front"
    lidar_name = "semantic_lidar"
    semantic_refined_folder_name = os.path.join(common_folder_name, 
                                                "refined_semantic_image")
    refined_lidar_folder_name = os.path.join(common_folder_name,
                                             "refined_pcd")
    color_map_folder_name = os.path.join(common_folder_name,
                                         "refined_colormap")
    semantic_refined_ints_folder_name = os.path.join(common_folder_name,
                                         "refined_semantic_ints_image")
    os.mkdir(semantic_refined_folder_name) if not os.path.exists(semantic_refined_folder_name) else None
    os.mkdir(refined_lidar_folder_name) if not os.path.exists(refined_lidar_folder_name) else None
    os.mkdir(color_map_folder_name) if not os.path.exists(color_map_folder_name) else None
    os.mkdir(semantic_refined_ints_folder_name) if not os.path.exists(semantic_refined_ints_folder_name) else None
    ######

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
    print(lidar2world)
    # return

    for filename in tqdm(os.listdir("/media/akshay/Data/16822/data/pcd_new")):
        if filename.endswith(".npy"):
            filename = filename.split(".npy")[0]
        else:
            continue
        pcd = np.load(os.path.join(
                                pcd_folder,
                                f"{filename}.npy"))
        # print(pcd.shape)
        image = cv2.imread(os.path.join(
                                    image_folder,
                                    f"{filename}.png"))
        project_lidar2cam_semantic(lidar2world,
                                world2cam,
                                cam_K,
                                pcd[:, :3],
                                image,
                                pcd[:, -1],
                                filename,
                                semantic_refined_folder_name,
                                semantic_refined_ints_folder_name,
                                refined_lidar_folder_name,
                                color_map_folder_name)
        # break
        # cv2.waitKey(0)

if __name__ == '__main__':
    main()