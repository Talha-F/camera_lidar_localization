# script to plot the trajectory of vehicle given a set of 4x4 tranformation matrices

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Plot the trajectory of vehicle given a set of 4x4 tranformation matrices')
    parser.add_argument('--transformICP', type=str, default='.', help='directory of the transformation matrices from ICP')
    parser.add_argument('--transformCCS', type=str, default='.', help='directory of the transformation matrices from CCS')
    
    args = parser.parse_args()

    # the matrix is the transformation matrix from the current frame to the next frame

    transformsICP = []
    transformsCCS = []

    for filename in os.listdir(args.transformICP):
        if filename.endswith('.npy'):
            transformsICP.append(np.load(os.path.join(args.transformICP, filename)))

    for filename in os.listdir(args.transformCCS):
        if filename.endswith('.npy'):
            transformsCCS.append(np.load(os.path.join(args.transformCCS, filename)))

    initPointICP = np.array([0, 0, 0, 1])
    initPointCCS = np.array([0, 0, 0, 1])

    trajectoryICP = []
    trajectoryCCS = []

    trajectoryCCS.append(initPointCCS.copy()[:3])
    trajectoryICP.append(initPointICP.copy()[:3])

    for i in range(len(transformsICP)):
        initPointICP = np.dot(transformsICP[i], initPointICP)
        initPointICP = initPointICP / initPointICP[3]
        trajectoryICP.append(initPointICP.copy()[:3])

    for i in range(len(transformsCCS)):
        initPointCCS = np.dot(transformsCCS[i], initPointCCS)
        initPointCCS = initPointCCS / initPointCCS[3]
        trajectoryCCS.append(initPointCCS.copy()[:3])

    trajectoryICP = np.array(trajectoryICP)
    trajectoryCCS = np.array(trajectoryCCS)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # plot trajectory by ICP in red
    ax.plot(trajectoryICP[:, 0], trajectoryICP[:, 1], trajectoryICP[:, 2], color='r')
    # plot trajectory by CCS in blue
    ax.plot(trajectoryCCS[:, 0], trajectoryCCS[:, 1], trajectoryCCS[:, 2], color='b')
    plt.show()

if __name__ == '__main__':
    main()





