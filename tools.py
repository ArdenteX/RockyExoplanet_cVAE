import numpy as np
from scipy.spatial import distance

def get_pred_mean(z, sample_num):
    sub_arrays = np.array_split(z, sample_num)
    mean_z = []

    for i in range(len(sub_arrays)):
        tmp_s = sub_arrays[i]
        mean_z.append(np.mean(tmp_s, axis=0))

    mean_z = np.vstack(mean_z)

    return mean_z

def get_cluster_center(z_2d, cluster_labels, dense_mask, mean_z, planet_x):

    cluster_center_z = []
    cluster_center_x = []

    for l in np.unique(dense_mask):
        z_2d_sel_1 = z_2d[cluster_labels == l]
        curr_dis = distance.cdist(z_2d_sel_1, z_2d_sel_1, 'euclidean')
        dist_means = np.mean(curr_dis, axis=1)
        cluster_center_z.append(mean_z[cluster_labels == l, :][np.argmax(dist_means), :])
        cluster_center_x.append(planet_x[cluster_labels == l, :][np.argmax(dist_means), :])

    cluster_center_z = np.vstack(cluster_center_z)
    cluster_center_x = np.vstack(cluster_center_x)

    return cluster_center_z, cluster_center_x

def switch_rgb(hex_color, alpha=None):
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    if alpha is not None:
        return tuple(int(hex_color[i: i+2], 16) / 256 for i in (0, 2, 4)) + (alpha,)

    else:
        return tuple(int(hex_color[i: i + 2], 16) / 256 for i in (0, 2, 4))