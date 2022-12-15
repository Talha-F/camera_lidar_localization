import numpy as np
import os
import cv2


if __name__ == '__main__':
    base_path = "/media/akshay/Data/16822/data"
    image_path = os.path.join(base_path, "epipolar_1")
    pcd_path = os.path.join(base_path, "pcd")
    pcd_out = os.path.join(base_path, "pcd_ordered")
    epipolar_ordered = os.path.join(base_path, "epipolar_ordered")

    count = 0
    non_count = 0
    print(f"lidars {len(os.listdir(pcd_path))}")

    for i, filename in enumerate(sorted(os.listdir(image_path))):
        if os.path.exists(os.path.join(pcd_path, 
                                       filename.split('.png')[0]+".npy")):
            # print(filename)
            pcd = np.load(os.path.join(pcd_path, filename.split('.')[0] + ".npy"))
            # print(pcd.shape)
            np.save(os.path.join(pcd_out, f"{count}.npy"), pcd)
            img = cv2.imread(os.path.join(image_path, filename))
            cv2.imwrite(os.path.join(epipolar_ordered, f"{count}.png"), img)
            cv2.imshow("fff", img)
            cv2.waitKey(2)
            count += 1
        else:
            # if os.remove(os.path.join(image_path, filename)):
            #     print(f"deleted {filename}")
            # print(filename)
            non_count += 1

    print(f"count {count}, non_count {non_count}")