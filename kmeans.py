import os.path
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Optional, List
import seaborn as sns


def get_elbow(pred_standard, save_path=None, planet_name=None, model='kmeans'):
    n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # overall_mean = np.mean(cvae_pred_standard_reshape[:, 0], axis=0)

    kmeans_results = {}

    for n in n_clusters:
        if model == 'kmeans':
            kmeans = KMeans(random_state=0, n_clusters=n)
            labels = kmeans.fit_predict(pred_standard)  # (n_samples * 8, 1)

        elif model == 'gmm':
            gmm = GaussianMixture(n_components=n, random_state=0)
            labels = gmm.fit_predict(pred_standard)

        cur_max_var = []
        for l in np.unique(labels):
            cluster_point = pred_standard[np.where(labels == l)]  # Select the current cluster data

            if cluster_point.shape[0] > 1:
                pca = PCA(n_components=1)  # Calculate maximum variance
                pca.fit(cluster_point)
                max_var = pca.explained_variance_[0]

                cur_max_var.append(max_var)

            else:
                cur_max_var.append(0)
        kmeans_results[n] = np.mean(cur_max_var)

    # Normalization
    m_scaler = MinMaxScaler()
    kmeans_results_scaled = m_scaler.fit_transform(np.array(list(kmeans_results.values())).reshape(-1, 1))

    # Find the elbow
    elbow = 0
    max_dis = -1
    for i in range(2, len(kmeans_results_scaled) - 1):
        dis = abs(kmeans_results_scaled[i] - kmeans_results_scaled[i + 1])
        if dis > max_dis:
            max_dis = dis
            elbow = i + 1

    fig = plt.figure(figsize=(8, 6))
    plt.plot(kmeans_results.keys(), kmeans_results_scaled, marker='s', linestyle='-', color='b', markersize=8,
             markeredgewidth=2, markeredgecolor='black')
    plt.scatter(elbow, kmeans_results_scaled[elbow - 1], facecolors='none', edgecolors='red', s=600, linewidths=2)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Normalization Maximum Variance")
    plt.xticks(range(1, 11))
    plt.show()

    if save_path is not None:
        fig.savefig("{}{}_elbow.pdf".format(save_path, planet_name), bbox_inches='tight')

    return elbow

def get_mean_std(op, l, pred):
    # Calculate mean and std each column of each cluster
    m_s_dict = {
        i: {
            c: {
                'mean': 0.0,
                'std': 0.0,
            } for c in np.unique(l)
        } for i in op
    }

    for i in range(len(op)):
        for c in np.unique(l):
            # Select the current original label data which belong current cluster label e.g. first original label data which belong cluster 1
            tmp = pred[:, i][np.where(l == c)]
            mean = tmp.mean()
            std = tmp.std()
            m_s_dict[op[i]][c]['mean'] = mean
            m_s_dict[op[i]][c]['std'] = std

            # print("Original Label: {}, Cluster Label: {}, Mean: {}, Std: {}".format(op[i - 1], c, round(mean, 2), round(std, 4)))

    return m_s_dict


def filter_by_std(pred, op):
    columns_intervals = [[pred[:, i].mean() - pred[:, i].std(), pred[:, i].mean() + pred[:, i].std()] for i in
                         range(len(op))]

    filter_data_idx = []
    for j in range(len(pred)):
        accept = False
        for i in range(len(op)):
            if columns_intervals[i][0] <= pred[j, i] <= columns_intervals[i][1]:
                accept = True

            else:
                accept = False
                break

        if accept:
            filter_data_idx.append(j)


    return filter_data_idx


def get_error_bar(op, l, ms_dict, ncols=2, nrows=4, save_path: Optional[str]=None, planet_name: Optional[str]=None, y_max: List[int]=None, y_locator: List[int]=None):
    limit_on = False

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    color_list = ['red', 'blue', 'green', 'yellow', 'pink', 'purple']
    for i, ax in zip(range(1, len(op) + 1), axs.flatten()):

        unique_label, counts = np.unique(l, return_counts=True)

        counts = counts / np.sum(counts) * 100
        hue_levels = sorted(unique_label)
        palette = sns.color_palette(n_colors=len(hue_levels))

        cat_palette_dict = dict(zip(unique_label, palette))

        # color_list = [cat_palette_dict[l] for l in unique_label]
        handles = []
        for idx, c in enumerate(unique_label):
            label_text = f"Cluster {c}: {counts[idx]:.1f}%"
            patch = mpatches.Patch(color=color_list[idx], label=label_text)
            handles.append(patch)

            ax.errorbar(x=c, y=ms_dict[op[i - 1]][c]['mean'], yerr=ms_dict[op[i - 1]][c]['std'], fmt='o', capsize=3,
                        color=color_list[idx])

        ax.set_ylabel("{}".format(op[i - 1]))
        ax.set_xticks(range(1, len(np.unique(l)) + 1))
        if i >= len(op) - 1:
            ax.set_xlabel("Cluster")

        if y_max and y_locator:
            limit_on = True
            ax.set_ylim(0, y_max[i - 1])

            ax.yaxis.set_major_locator(plt.MultipleLocator(y_locator[i - 1]))
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.legend(handles=handles, title='Proportion', loc='upper right', framealpha=0.5)

    plt.show()

    if save_path and planet_name:
        fig.savefig(os.path.join(save_path, "{}_Error_Bar_{}.pdf".format(planet_name, 'Limit_on' if limit_on == True else 'Limit_off')), bbox_inches='tight')

    if save_path and not planet_name:
        warnings.warn("Received image save path but get an unknown planet name, use the default name: 'Unknown Planet'", UserWarning)
        fig.savefig(os.path.join(save_path, "{}_Error_Bar_{}.pdf".format('Unknown Planet', 'Limit_on' if limit_on == True else 'Limit_off')), bbox_inches='tight')
