import numpy as np
import json
import os
import cv2


if __name__ == '__main__':
    folder_name = "./results_aws/info/"
    image_1_folder = "./results_aws/before/"
    image_2_folder = "./results_aws/after"
    org_angles = np.asarray([0, -18, 0])
    initial_errors = 0
    final_errors = 0
    for count, filename in enumerate(os.listdir(folder_name)):
        with open(os.path.join(folder_name, filename), 'r') as f:
            json_file = json.load(f)
        initial_angles = np.asarray(json_file['initial_angles']).reshape(-1)
        final_angles = np.asarray(json_file['final_angles']).reshape(-1)
        initial_errors += np.mean(np.power(org_angles - initial_angles, 2))
        final_errors += np.mean(np.power(org_angles-final_angles, 2))
        before_image = cv2.imread(os.path.join(image_1_folder,
                                              f"{filename.split('.')[0]}.png"))
        after_image = cv2.imread(os.path.join(image_2_folder,
                                              f"{filename.split('.')[0]}.png"))
        img = cv2.hconcat((before_image, after_image))
        img = cv2.resize(img, (1800, 800))
        cv2.imshow("before_and_after", img)
        cv2.waitKey(30)
    
    print(f"intial_error {initial_errors/(count+1)}, {final_errors/(count+1)}")