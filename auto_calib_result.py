import numpy as np
import json
import os

if __name__ == '__main__':
    folder_name = "./results_aws/info/"
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
    
    print(f"intial_error {initial_errors/(count+1)}, {final_errors/(count+1)}")