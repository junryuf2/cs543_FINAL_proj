import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np


def load_keypoints(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [np.array(frame['keypoints'][0]) for frame in data if frame['keypoints']]


def get_camera_parameters():
    focal_length_pixels = 1785
    pp = (1920 / 2, 1280 / 2)
    camera_matrix = np.array([
        [focal_length_pixels, 0, pp[0]],
        [0, focal_length_pixels, pp[1]],
        [0, 0, 1]
    ])
    R = np.eye(3)
    T = np.array([[100, 0, 0]]).T
    return camera_matrix, R, T


def triangulate_points(kp1, kp2, cam_matrix, R, T):
    if len(kp1) == 0 or len(kp2) == 0:
        return np.array([])
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)


    inv_cam_matrix = np.linalg.inv(cam_matrix)
    kp1_normalized = (inv_cam_matrix @ np.vstack((kp1.T, np.ones((1, kp1.shape[0])))))[:2]
    kp2_normalized = (inv_cam_matrix @ np.vstack((kp2.T, np.ones((1, kp2.shape[0])))))[:2]


    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, kp1_normalized, kp2_normalized)
    points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
    return points_3d.T


def filter_limits(points, lower_percentile=5, upper_percentile=95):
    """Compute axis limits that exclude extreme outliers."""
    p_low = np.percentile(points, lower_percentile)
    p_high = np.percentile(points, upper_percentile)
    return p_low, p_high


def main():
    json_file1 = 'revised_videos/keypoints/IMG_0674.json'
    json_file2 = 'revised_videos/keypoints/IMG_2358.json'
    keypoints1 = load_keypoints(json_file1)
    keypoints2 = load_keypoints(json_file2)
    camera_matrix, R, T = get_camera_parameters()


    points_series = []
    for kp1, kp2 in zip(keypoints1, keypoints2):
        points_3d = triangulate_points(kp1, kp2, camera_matrix, R, T)
        if points_3d.size > 0:
            points_series.append(points_3d)


    all_3d_points = np.dstack(points_series)  #  (N_frames, 17, 3)
    all_3d_points = np.transpose(all_3d_points, (2, 1, 0))  # (3, 17, N_frames)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # percentile filter to axis limits
    x_min, x_max = filter_limits(all_3d_points[0])
    y_min, y_max = filter_limits(all_3d_points[1])
    z_min, z_max = filter_limits(all_3d_points[2])


    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])


    # remove axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


    scatters = [ax.scatter(all_3d_points[0, 0, i], all_3d_points[0, 1, i], all_3d_points[0, 2, i]) for i in range(17)]
   
 
    def update(num, data, scatters, ax):
        for i, sc in enumerate(scatters):
            sc._offsets3d = (data[num:num+1, 0, i], data[num:num+1, 1, i], data[num:num+1, 2, i])


        # Dynamically adjust axes limits
        current_data = data[num, :, :]
        x_min, x_max = filter_limits(current_data[0])
        y_min, y_max = filter_limits(current_data[1])
        z_min, z_max = filter_limits(current_data[2])
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])


        return scatters


    # Modify the FuncAnimation call to include ax as a part of fargs
    ani = FuncAnimation(fig, update, frames=len(points_series), fargs=(all_3d_points, scatters, ax), repeat=True)




    plt.legend(loc='upper left')
    ani.save('3D_points_motion.mp4', writer='ffmpeg', fps=20)


if __name__ == '__main__':
    main()
