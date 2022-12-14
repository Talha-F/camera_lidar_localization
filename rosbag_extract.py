import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
import os
import json
from scipy.spatial.transform import Rotation as R
import ros_numpy
from sensor_msgs import point_cloud2
from tqdm import tqdm


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float32):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
	a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['ObjTag']
    return points

def extract_transforms(bag_path="./data/2022-11-12-23-04-54.bag",
                       topics=["ego_vehicle/epipolar_one",
                               "ego_vehicle/epipolar_two",
                               "ego_vehicle/lidar_custom"],
                       json_path="./data/transforms.json"):
    results_json = {}
    added_topic = set()
    bag = rosbag.Bag(bag_path)
    topic = "/tf"
    for i, (_, msg, time) in enumerate(bag.read_messages(topic)):
        # print(msg)
        print("ss ", len(added_topic))
        if len(added_topic) == len(topics):
            break

        if msg.transforms[0].child_frame_id in added_topic or \
           not msg.transforms[0].child_frame_id in topics:
            continue
        
        added_topic.add(msg.transforms[0].child_frame_id)
        results_json[f"{msg.transforms[0].child_frame_id}_t"] = [msg.transforms[0].transform.translation.x,
                                                                 msg.transforms[0].transform.translation.y,
                                                                 msg.transforms[0].transform.translation.z]
        results_json[f"{msg.transforms[0].child_frame_id}_R"] = [msg.transforms[0].transform.rotation.w,
                                                                 msg.transforms[0].transform.rotation.x,
                                                                 msg.transforms[0].transform.rotation.y,
                                                                 msg.transforms[0].transform.rotation.z]
        print(results_json)
        # break
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=3)
    
def extract_pcds(bag_path="./data/2022-11-12-23-04-54.bag"):
    bag = rosbag.Bag(bag_path)
    topic = "/carla/ego_vehicle/lidar_custom"
    for i, (_, msg, time) in enumerate(bag.read_messages(topic)):
        pcd = np.zeros((0, 4))
        # print(msg)
        print(time)
        for point in tqdm(point_cloud2.read_points(msg)):
            # print(point)
            pcd = np.concatenate((pcd, np.asarray(point).reshape(1, -1)), 0)
        np.save(f"{str(time)}.npy", pcd)
        # np_data = ros_numpy.numpify(msg.data)
        # print(points)
        # print(np_data.shape)
        # break

def extract_pcds_vectorized(bag_path="/media/akshay/Data/auto_calib_2022-12-03-19-02-56.bag"):
    bag = rosbag.Bag(bag_path)
    topic = "/carla/ego_vehicle/semantic_lidar"
    output_folder_name = "/media/akshay/Data/16822/data/pcd_new/"
    loop = tqdm(bag.read_messages(topic), total=43103)
    for i, (_, msg, time) in enumerate(loop):
        pcd = np.zeros((0, 4))
        loop.set_postfix({"time: ": time})
        pcd = get_xyz_points(ros_numpy.point_cloud2.pointcloud2_to_array(msg))
        print(np.unique(pcd[:, -1]))
        break
        np.save(os.path.join(output_folder_name,
                             f"{time}.npy"), 
                pcd)

def extract_semantic_image(bag_path="/media/akshay/Data/auto_calib_2022-12-03-19-02-56.bag",
                           topic = "/carla/ego_vehicle/semantic_segmentation_front/image",
                           output_folder_name = "/media/akshay/Data/16822/data/semantic_image_new/"):
    bag = rosbag.Bag(bag_path)
    if not os.path.exists(output_folder_name):
        os.mkdir(output_folder_name)

    bridge = CvBridge()
    loop = tqdm(bag.read_messages(topic), total=43103)
    for i, (_, msg, time) in enumerate(loop):
        print(i)
        img = bridge.imgmsg_to_cv2(msg)
        cv2.imwrite(os.path.join(output_folder_name,
                                 f"{time}.png"),
                    img)
        cv2.imshow("semantic_res", img)
        cv2.waitKey(5)
        # print(img.shape)
        # print(np.unique(img.reshape(-1, 4), axis=0))
        
def extract_images():
    bag_path = "./data/2022-11-12-23-04-54.bag"
    output_path_1 = "./data/epipolar_1/"
    output_path_2 = "./data/epipolar_2/"
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    topic = ["/carla/ego_vehicle/epipolar_one/image",
             "/carla/ego_vehicle/epipolar_two/image"]

    for i, (_, msg, time) in enumerate(bag.read_messages(topics=topic[0])):
        # print(msg)
        img = bridge.imgmsg_to_cv2(msg)
        # print(img.shape)
        cv2.imwrite(os.path.join(output_path_1, f"{str(time)}.png"), img)
        cv2.imshow("hi", img)
        cv2.waitKey(10)
    
    for i, (_, msg, time) in enumerate(bag.read_messages(topics=topic[1])):
        # print(msg)
        img = bridge.imgmsg_to_cv2(msg)
        # print(img.shape)
        cv2.imwrite(os.path.join(output_path_2, f"{str(time)}.png"), img)
        cv2.imshow("hi", img)
        cv2.waitKey(10)
    cv2.destroyAllWindows()

def extract_camera_info(bag_path = "./data/2022-11-12-23-04-54.bag",
                        topic = ["/carla/ego_vehicle/epipolar_one/camera_info",
                                 "/carla/ego_vehicle/epipolar_two/camera_info"],
                        output_folder="./data/sem_camera_info.json"):
    bag = rosbag.Bag(bag_path)
    cam_info = {}

    for i, (_, msg, _) in enumerate(bag.read_messages(topics=topic[0])):
        cam_info["K1"] = msg.K
        break
    for i, (_, msg, _) in enumerate(bag.read_messages(topics=topic[1])):
        print(f"second cam {msg.K}")
        break
    # for extrinsic
    # epipolar 1
    if len(topic) >= 3:
        euler_angles = np.asarray([0, 0, -5])
        translation = np.asarray([2, 0.25, 2]).reshape(3, 1)
        r = R.from_euler("z", euler_angles[-1], degrees=True).as_matrix()
        rc = -r@translation
        # rc = translation
        extrinsic_1 = np.concatenate((r, rc), axis=1)

        # epipolar 2
        euler_angles = np.asarray([0, 0, -5])
        translation = np.asarray([2, -0.25, 2]).reshape(3, 1)
        r = R.from_euler("z", euler_angles[-1], degrees=True)
        r = r.as_matrix()
        rc = -r@translation
        # rc = translation
        extrinsic_2 = np.concatenate((r, rc), axis=1)
        cam_info["extrinsic_2"] = extrinsic_2.tolist()
        cam_info["extrinsic_1"] = extrinsic_1.tolist()
        cam_info["P1"] = (np.asarray(cam_info["K"]).reshape(3, -1)@extrinsic_1).tolist()
        cam_info["P2"] = (np.asarray(cam_info["K"]).reshape(3, -1)@extrinsic_2).tolist()

    print(cam_info)
    with open(output_folder, "w") as f:
        json.dump(cam_info, f, indent=4)

def main():
    extract_semantic_image(bag_path="/home/akshay/auto_calib_2022-12-03-19-02-56.bag",
                           topic="/carla/ego_vehicle/wide_angle/image",
                           output_folder_name = "/media/akshay/Data/16822/data/wide_angle_image_new/")
    # extract_camera_info(bag_path="/home/akshay/auto_calib_2022-12-03-19-02-56.bag",
    #                     topic=["/carla/ego_vehicle/wide_angle/camera_info",
    #                            "/carla/ego_vehicle/semantic_segmentation_front/camera_info"])
    # extract_transforms(bag_path="/home/akshay/auto_calib_2022-12-03-19-02-56.bag",
    #                    topics=["ego_vehicle/semantic_lidar",
    #                            "ego_vehicle/wide_angle",
    #                            "ego_vehicle/semantic_segmentation_front"],
    #                     json_path="./data/tf_info.json")
    # extract_semantic_image()
    # extract_pcds_vectorized()
    # extract_transforms()
    # extract_images()
    # extract_camera_info()
    # extract_pcds()

if __name__ == '__main__':
    # extract()
    main()