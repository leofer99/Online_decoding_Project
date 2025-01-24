import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.decomposition import PCA, KernelPCA
from pathlib import Path
import pickle
import re
import logging
import os
from os.path import dirname
from scipy.ndimage import gaussian_filter
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from bcidecode.preprocessing.data import cursor_preprocessing, load_data
import matplotlib.pyplot as plt


def none(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    #without alterations:
    dim_1=0
    dim_2=1
    dim_3=2

    # Plot each trajectory
    for trajectory_data, color, label in zip(
        [average_left_trial_latents, average_straight_trial_latents, average_right_trial_latents],
        ['b', 'g', 'r'],
        ['Left', 'Straight', 'Right']
    ):
        ax.scatter3D(trajectory_data[5:, dim_1], trajectory_data[5:, dim_2], trajectory_data[5:, dim_3], marker='o', c=color, label=label)
        ax.scatter3D(trajectory_data[5, dim_1], trajectory_data[5, dim_2], trajectory_data[5, dim_3], marker='X', c=color, s=100)
        ax.plot3D(trajectory_data[5:, dim_1], trajectory_data[5:, dim_2], trajectory_data[5:, dim_3], color)

    ax.set_title('Visualization using the 3 latent modes with the most co-variance')
    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')
    ax.set_zlabel('LD 3')

    # plt.legend()
    # plt.show()
    return ax

def processData(neural_file, latents, dest_dir, DPAD=False):
    trials, taskparameters = load_data(neural_file)
    task = taskparameters["task"]
    monkey = re.search(r'_(\w+)_\d{8}_\d{4}_\w_', neural_file).group(1) #monkey identifier
    logging.info(f"Number of trials {len(trials)}")  
    if task == "reach":
        trials = trials[-120:] #selects the last 120 trials
    else:
        trials = trials[-90:]
    targets = [np.array([trial.targetPosition[0], trial.targetPosition[1], trial.targetPosition[2]]) for trial in trials]

    if DPAD:
        with open(os.path.join(dest_dir, f"all_observations.pkl"), 'rb') as f:
            observations = pickle.load(f)
        with open(os.path.join(dest_dir, f"all_states.pkl"), 'rb') as f:
            states = pickle.load(f)

        #reshape DPAD so it becomes 3D again
        n_trials=len(observations) #
        n_bins= np.min([obs.shape[0] for obs in observations])
        latents= latents.reshape(n_trials, n_bins, 6)
        latents=latents[:90, :,:] #one of the files has 97 trials
    else:
        latents=latents[:90]

    # Pad shorter trials with NaNs to make them equal in length
    max_time_bins = max(len(trial) for trial in latents)  # max number of bins for a trial
    logging.info(f"Distance: {len(latents)}")
    logging.info(f"Distance: {len(latents[0].shape)}")

    # Create an array of 10 NaN values
    nan_array = [np.full((latents[0].shape[1],), np.nan)]
    # Pad each trial with NaN arrays to have a consistent length of max_num_arrays arrays  
    latents = [np.append(trial, nan_array * (max_time_bins - len(trial)), axis=0) if len(trial) < max_time_bins else trial for trial in latents] 

    ## Plot data LATENT VARIABLES
    straight_trials = []
    slight_right_trials = []
    slight_left_trials = []
    right_trials = []
    left_trials = []
    straight_up_trials = []
    slight_right_up_trials = []
    slight_left_up_trials = []
    right_up_trials = []
    left_up_trials = []
    for trial in range(len(latents)):
        if task == "fixedCamera":
            if np.array_equal(targets[trial], np.array([0., 0.75, 9.2])):
                straight_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([7., 0.75, 6.])):
                right_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([-7., 0.75, 6.])):
                left_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([-3.5, 0.75, 8.5])):
                slight_left_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([3.5, 0.75, 8.5])):
                slight_right_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([0., 2.35, 9.2])):
                straight_up_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([7., 2.35, 6.])):
                right_up_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([-7., 2.35, 6.])):
                left_up_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([-3.5, 2.35, 8.5])):
                slight_left_up_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([3.5, 2.35, 8.5])):
                slight_right_up_trials.append(latents[trial])
        else:
            if np.array_equal(targets[trial], np.array([0., 1., 9.2])):
                straight_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([6., 1., 7.])):
                right_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([-6., 1., 7.])):
                left_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([-3., 1., 8.7])):
                slight_left_trials.append(latents[trial])
            elif np.array_equal(targets[trial], np.array([3., 1., 8.7])):
                slight_right_trials.append(latents[trial])

    return left_trials, right_trials, straight_trials

def PCA_(neural_file, observations_path):
        
    with open(observations_path, 'rb') as f:
        observations = pickle.load(f)

    #Preparation for PCA:
    min_trials=len(observations)
    min_time = np.min([array.shape[0] for array in observations])
    min_components = np.min([array.shape[1] for array in observations])

    observations1 = [array[:min_time, :min_components] for array in observations[:min_trials]]
    observations1_ = [ gaussian_filter(obs, sigma=0.05) for obs in observations ]

    observations1_2d=np.concatenate(observations1_, axis=0)
    # observations1_2d = gaussian_filter(observations1_2d, sigma=20)

    PCA_model = PCA(n_components=50, svd_solver='full').fit(observations1_2d)
    PCA_latent_trials= [PCA_model.transform(obs) for obs in observations1]

    # cumvar_expl= np.concatenate([[0], np.cumsum(PCA_model.explained_variance_ratio_)])
    # print(f"{np.where(cumvar_expl > 0.9)[0][0]} PCs needed to explain 90% of variance.")
    # plt.plot(cumvar_expl)
    # plt.grid()
    # plt.xlabel("Number of PCs")
    # _ = plt.ylabel("Cumulative variance explained")

    return PCA_latent_trials

def kPCA_(neural_file, observations_path):
        
    with open(observations_path, 'rb') as f:
        observations = pickle.load(f)

    #Preparation for PCA:
    min_trials=len(observations)
    min_time = np.min([array.shape[0] for array in observations])
    min_components = np.min([array.shape[1] for array in observations])

    observations1 = [array[:min_time, :min_components] for array in observations[:min_trials]]
    observations1_ = [ gaussian_filter(obs, sigma=0.05) for obs in observations ]

    observations1_2d=np.concatenate(observations1_, axis=0)
    # observations1_2d = gaussian_filter(observations1_2d, sigma=20)

    kernel_pca = KernelPCA(
        n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
    )

    kernel_pca.fit(observations1_2d)
    kPCA_latent_trials= [ kernel_pca.transform(obs)  for obs in observations1]

    # cumvar_expl= np.concatenate([[0], np.cumsum(PCA_model.explained_variance_ratio_)])
    # print(f"{np.where(cumvar_expl > 0.9)[0][0]} PCs needed to explain 90% of variance.")
    # plt.plot(cumvar_expl)
    # plt.grid()
    # plt.xlabel("Number of PCs")
    # _ = plt.ylabel("Cumulative variance explained")

    return kPCA_latent_trials

neural_file_list = [
    r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl",
    r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl",
    r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl",
    r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl"
]
neural_file= neural_file_list[1]
#2 is weird (doesnt match the result of the training script), 3 is not working!!

dest_dir = dirname(neural_file)
observations_path= rf"{dest_dir}\all_observations.pkl"

extracted_part = neural_file.split("navtrainingsphere_")[-1].split(".pkl")[0]
date = extracted_part.split('_')[1]


#PCA Dimension Reduction:
PCA_latents= PCA_(neural_file, observations_path)
left_trials, right_trials, straight_trials = processData(neural_file, PCA_latents, dest_dir)

Path(dest_dir).mkdir(parents=True, exist_ok=True)
with open( os.path.join(dest_dir, rf"PCA_latents.pkl"), "wb") as f:
    pickle.dump(PCA_latents, f)

average_left_trial_latents = np.mean(left_trials, axis=0)
average_straight_trial_latents = np.mean(straight_trials, axis=0)
average_right_trial_latents = np.mean(right_trials, axis=0)

#3D Visualization:
fig1 = plt.figure(figsize=(6, 6), facecolor="white")
fig1.suptitle(f"Extraction of latent states with PCA - File: {date}", size=13)
axnone=none(fig1, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)

#kPCA Dimension Reduction:
kPCA_latents= kPCA_(neural_file, observations_path)
left_trials, right_trials, straight_trials = processData(neural_file, kPCA_latents, dest_dir)

Path(dest_dir).mkdir(parents=True, exist_ok=True)
with open( os.path.join(dest_dir, rf"kPCA_latents.pkl"), "wb") as f:
    pickle.dump(kPCA_latents, f)

average_left_trial_latents = np.mean(left_trials, axis=0)
average_straight_trial_latents = np.mean(straight_trials, axis=0)
average_right_trial_latents = np.mean(right_trials, axis=0)

#3D Visualization:
fig1 = plt.figure(figsize=(6, 6), facecolor="white")
fig1.suptitle(f"Extraction of latent states with kPCA - File: {date}", size=13)
axnone=none(fig1, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)


# #PSID Dimension Reduction:
latents_path= rf"{dest_dir}\latents.pkl"

if os.path.exists(latents_path):
    print("File exists.")
else:
    print("File does not exist.")

with open(latents_path, 'rb') as f:
    PSID_latents = pickle.load(f)

left_trials, right_trials, straight_trials = processData(neural_file, PSID_latents, dest_dir)
average_left_trial_latents = np.mean(left_trials, axis=0)
average_straight_trial_latents = np.mean(straight_trials, axis=0)
average_right_trial_latents = np.mean(right_trials, axis=0)

#3D Visualization:
fig2 = plt.figure(figsize=(6, 6), facecolor="white")
fig2.suptitle(f"Extraction of latent states with PSID - File: {date}", size=13)
axnone=none(fig2, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)


# #DPAD Dimension Reduction:
latents_path= rf"{dest_dir}\DPAD_latents.pkl"

if os.path.exists(latents_path):
    print("File exists.")
else:
    print("File does not exist.")

with open(latents_path, 'rb') as f:
    DPAD_latents = pickle.load(f)

left_trials, right_trials, straight_trials = processData(neural_file, DPAD_latents, dest_dir, DPAD=True)
average_left_trial_latents = np.mean(left_trials, axis=0)
average_straight_trial_latents = np.mean(straight_trials, axis=0)
average_right_trial_latents = np.mean(right_trials, axis=0)

#3D Visualization:
fig3 = plt.figure(figsize=(6, 6), facecolor="white")
fig3.suptitle(f"Extraction of latent states with DPAD - File: {date}", size=13)
axnone=none(fig3, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)

plt.show()

