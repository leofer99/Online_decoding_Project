import numpy as np
import yaml
import xml.etree.ElementTree as ET
from tnsbmi.onlinedecoding import localconfig, tasks, util
from tnsbmi.tnsbmi import bintrials, dataconversion
from tnsbmi.tnsbmi import bintrials, nevdata
from tnsbmi.tnsbmi import modeling
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
from bcidecode.preprocessing.ratesTransformer import (EpochTransformer,
                                                      RatesTransformer)
from bcidecode.online.models import Model
from bcidecode.tools import ccaTools as cca
from sklearn.model_selection import train_test_split

##############################################
##############################################

def process_latents(latents_0_path):

    #load latent variables of model0:
    # common_path=os.path.dirname(neural_file)
    # latents_0_path= os.path.join(common_path, "latents.pkl")
    with open(latents_0_path, 'rb') as f:
        latents = pickle.load(f)

    #equalize size
    min_timebins = min( trial.shape[0] for trial in latents)
    latents_cut= [ trial[:min_timebins, :] for trial in latents ]
    latents_cut = np.array(latents_cut)    

    return latents_cut

def process_latents_Kalman(latents_path):

    #Used to process the latents that were obtained "online", aka using the Kalman filter

    #load latent variables of model0:
    # common_path=os.path.dirname(neural_file)
    # latents_0_path= os.path.join(common_path, "latents.pkl")
    with open(latents_path, 'rb') as f:
        latents = pickle.load(f)

    #equalize size
    min_timebins= min( trial.shape[0] for trial in latents)

    #cuts first 4timebins, because they are empty!
    latents_cut= [ np.array(trial[4:min_timebins]) for trial in latents ] 

    #shape is (90,50,1,6), convert to (90,50,6)
    latents_cut= np.squeeze(np.array(latents_cut))


    return latents_cut


def CCA_coefficients(latents_0, latents_k, targets_0, targets_k):
    
    #2.1 Ensure Day-0 and Day-K have the same number of trials:
    # min_trials= min(latents_0.shape[0], latents_k.shape[0])
    n_trials_0 = latents_0.shape[0]
    n_trials_k = latents_k.shape[0]

    #2.2 Ensure Day-0 and Day-K have the same number of timebins:
    min_timebins= min(latents_0.shape[1], latents_k.shape[1])

    latents_0_cut= latents_0[:, :min_timebins, :]
    latents_k_cut= latents_k[:, :min_timebins, :]

    # 2.3 (Optional) Obtains the average for each direction and uses that to obtain the CCA coefficients:
    avg_straight_0, avg_right_0, avg_left_0= separate_by_target_preprocessing(n_trials_0, targets_0, latents_0_cut)
    avg_straight_k, avg_right_k, avg_left_k= separate_by_target_preprocessing(n_trials_k, targets_k, latents_k_cut)

    latents_0_cut = np.vstack([np.tile(avg_straight_0, (30, 1, 1)),
                            np.tile(avg_right_0, (30, 1, 1)),
                            np.tile(avg_left_0, (30, 1, 1))])
    
    latents_k_cut = np.vstack([np.tile(avg_straight_k, (30, 1, 1)),
                            np.tile(avg_right_k, (30, 1, 1)),
                            np.tile(avg_left_k, (30, 1, 1))])

    latents_0_2d= latents_0_cut.reshape(-1, latents_0_cut.shape[-1])
    latents_k_2d= latents_k_cut.reshape(-1, latents_k_cut.shape[-1])


    #3. Perform CCA:
    A, B, r, U, V = cca.canoncorr(latents_0_2d, latents_k_2d, fullReturn=True)
    CCA=[A, B, min_timebins]

    return CCA

def apply_CCA(CCA, latents_0, latents_k):
    U, _, Vh = linalg.svd(CCA[0], full_matrices=False, compute_uv=True, overwrite_a=False)
    U2, _,Vh2 = linalg.svd(CCA[1], full_matrices=False, compute_uv=True, overwrite_a=False)

    latents_sim0_canonical=[]
    latents_simk_canonical=[]

    for i in range(len(latents_k)):
        # x_0_canonical = latents_0[i] @ U @ Vh
        # x_k_canonical = latents_k[i] @ U2 @ Vh2
        # # x_0_canonical = padded_latents_0_cut[i] @ A
        # # x_k_canonical = padded_latents_k_cut[i] @ B
        # latents_0_canonical.append(x_0_canonical)
        # latents_k_canonical.append(x_k_canonical)

        x_sim0_canonical= latents_k[i] @ U2 @ Vh2 @ np.linalg.pinv(U @ Vh)
        latents_sim0_canonical.append(x_sim0_canonical)

    for i in range(len(latents_0)):
        x_simk_canonical= latents_0[i] @ U @ Vh @ np.linalg.pinv(U2 @ Vh2)
        # ex= padded_latents_k[i] @ A @ np.linalg.pinv(B)

        latents_simk_canonical.append(x_simk_canonical)

    # latents_0_canonical=np.array(latents_0_canonical)
    # latents_k_canonical=np.array(latents_k_canonical)
    latents_sim0_canonical=np.array(latents_sim0_canonical)
    latents_simk_canonical=np.array(latents_simk_canonical)

    return latents_sim0_canonical, latents_simk_canonical

def separate_by_target_preprocessing(n_trials, targets, latents):

    straight_trials = []
    right_trials = []
    left_trials = []

    if n_trials>90:
        latents=latents[:90]
        n_trials=90

    for trial in range(n_trials):
        if np.array_equal(targets[trial], np.array([0., 0.75, 9.2])):
            straight_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([7., 0.75, 6.])):
            right_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([-7., 0.75, 6.])):
            left_trials.append(latents[trial])
    else:
        if np.array_equal(targets[trial], np.array([0., 1., 9.2])):
            straight_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([6., 1., 7.])):
            right_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([-6., 1., 7.])):
            left_trials.append(latents[trial])

    average_left_trial_latents = np.mean(left_trials, axis=0)
    average_straight_trial_latents = np.mean(straight_trials, axis=0)
    average_right_trial_latents = np.mean(right_trials, axis=0)

    average_straight_trial_latents = np.expand_dims(average_straight_trial_latents, axis=0)
    average_right_trial_latents = np.expand_dims(average_right_trial_latents, axis=0)
    average_left_trial_latents = np.expand_dims(average_left_trial_latents, axis=0)

    return average_straight_trial_latents, average_right_trial_latents, average_left_trial_latents

def separate_by_target(n_trials, targets, latents):

    straight_trials = []
    right_trials = []
    left_trials = []

    if n_trials>90:
        latents=latents[:90]
        n_trials=90

    for trial in range(n_trials):
        if np.array_equal(targets[trial], np.array([0., 0.75, 9.2])):
            straight_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([7., 0.75, 6.])):
            right_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([-7., 0.75, 6.])):
            left_trials.append(latents[trial])
    else:
        if np.array_equal(targets[trial], np.array([0., 1., 9.2])):
            straight_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([6., 1., 7.])):
            right_trials.append(latents[trial])
        elif np.array_equal(targets[trial], np.array([-6., 1., 7.])):
            left_trials.append(latents[trial])

    average_left_trial_latents = np.mean(left_trials, axis=0)
    zdata_left = []
    ydata_left = []
    xdata_left = []
    for trial in range(len(average_left_trial_latents)):
        xdata_left.append(average_left_trial_latents[trial][0])
        ydata_left.append(average_left_trial_latents[trial][1])
        zdata_left.append(average_left_trial_latents[trial][2])

    average_straight_trial_latents = np.mean(straight_trials, axis=0)
    zdata_straight = []
    ydata_straight = []
    xdata_straight = []
    for trial in range(len(average_straight_trial_latents)):
        xdata_straight.append(average_straight_trial_latents[trial][0])
        ydata_straight.append(average_straight_trial_latents[trial][1])
        zdata_straight.append(average_straight_trial_latents[trial][2])

    average_right_trial_latents = np.mean(right_trials, axis=0)
    zdata_right = []
    ydata_right = []
    xdata_right = []
    for trial in range(len(average_right_trial_latents)):
        xdata_right.append(average_right_trial_latents[trial][0])
        ydata_right.append(average_right_trial_latents[trial][1])
        zdata_right.append(average_right_trial_latents[trial][2])

    data_straight=np.array([xdata_straight, ydata_straight, zdata_straight])
    data_right=np.array([xdata_right, ydata_right, zdata_right])
    data_left=np.array([xdata_left, ydata_left, zdata_left])

    return data_straight, data_right, data_left

def plot_trajectories(ax, data_straight, data_right, data_left, colors, labels):

    #data_straight[0] is X, [1] is Y, [2] is Z
    ax.scatter3D(data_straight[0, 5:], data_straight[1, 5:], data_straight[2, 5:], marker='o', c=colors[0])  # c=zdata_straight[6:], cmap='Greens');
    ax.scatter3D(data_straight[0, 5], data_straight[1, 5], data_straight[2, 5], marker='X', c=colors[0], s=100)
    ax.plot3D(data_straight[0, 5:], data_straight[1, 5:], data_straight[2, 5:], colors[0], label="Straight "+labels)

    ax.scatter3D(data_left[0, 5:], data_left[1, 5:], data_left[2, 5:], marker='o', c=colors[1])  # , c=zdata_left, cmap='Blues')
    ax.scatter3D(data_left[0, 5], data_left[1, 5], data_left[2, 5], marker='X', c=colors[1], s=100)
    ax.plot3D(data_left[0, 5:], data_left[1, 5:], data_left[2, 5:], c=colors[1], label="Left "+labels)

    ax.scatter3D(data_right[0, 5:], data_right[1, 5:], data_right[2, 5:], marker='o', c=colors[2])  # c=zdata_right, cmap='Reds')
    ax.scatter3D(data_right[0, 5], data_right[1, 5], data_right[2, 5], marker='X', c=colors[2], s=100)
    ax.plot3D(data_right[0, 5:], data_right[1, 5:], data_right[2, 5:], c=colors[2], label="Right "+labels)

    plt.legend()
    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')
    ax.set_zlabel('LD 3')
    plt.show(block=False)

    return ax

def full_process(latents_0_path, latents_k_path, targets_day0_path, targets_dayk_path, neural_file_0, neural_file_k):

    # Identify the targets:
    with open(targets_day0_path, 'rb') as f:
        targets_0 = pickle.load(f)
    with open(targets_dayk_path, 'rb') as f:
        targets_k = pickle.load(f)

    #0. Equalize number of timebins across the trials:
    latents_0 = process_latents(latents_0_path)
    latents_k = process_latents(latents_k_path)


    # 1. Split the Data into Train and Test:
    test_size=0.9
    random_state=38
    targets_0_train, targets_0_test, latents0_train, latents0_test = train_test_split(targets_0, latents_0, test_size=test_size, random_state=random_state)
    targets_k_train, targets_k_test, latentsk_train, latentsk_test = train_test_split(targets_k, latents_k, test_size=test_size, random_state=random_state)


    # 2. CCA Coefficients + Apply CCA:
    #TRAIN: (latents0 = latents0_train when averaging)
    CCA_ = CCA_coefficients(latents_0, latentsk_train, targets_0, targets_k_train)
    #TEST:
    latents_sim0_canonical, latents_simk_canonical= apply_CCA(CCA_, latents_0, latentsk_test)


    #3. Plot Results:
    n_trials_0= len(latents_0)
    n_trials_k= len(latentsk_test)
    n_trials_sim= len(latentsk_test)

    data_straight_day0, data_right_day0, data_left_day0= separate_by_target(n_trials_0, targets_0, latents_0)
    data_straight_dayK, data_right_dayK, data_left_dayK= separate_by_target(n_trials_k, targets_k_test, latentsk_test)
    data_straight_sim_day0, data_right_sim_day0, data_left_sim_day0= separate_by_target(n_trials_sim, targets_k_test, latents_sim0_canonical)
    # data_straight_sim_dayk, data_right_sim_dayk, data_left_sim_dayk= separate_by_target(n_trials_0, targets_0, latents_simk_canonical)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    colors_day0=['green', 'blue', 'red']
    colors_sim_day0=['lightgreen', 'lightsteelblue', 'pink']
    colors_dayk=['darkgreen', 'navy', 'maroon']
    labels_day0= 'Day0'
    labels_sim_day0= 'Sim Day0'
    labels_dayk= 'DayK'

    plot_trajectories(ax, data_straight_dayK, data_right_dayK, data_left_dayK, colors_dayk, labels_dayk)
    plot_trajectories(ax, data_straight_sim_day0, data_right_sim_day0, data_left_sim_day0, colors_sim_day0, labels_sim_day0)
    plot_trajectories(ax, data_straight_day0, data_right_day0, data_left_day0, colors_day0, labels_day0)
    
    date_day0 = neural_file_0.split("navtrainingsphere_")[-1].split(".pkl")[0].split('_')[1]
    date_dayk = neural_file_k.split("navtrainingsphere_")[-1].split(".pkl")[0].split('_')[1]

    ax.set_title(f'CCA (Data Split, Test size={test_size}): Comparing Day-K ('+date_dayk+'), Day-0 ('+date_day0+') and Aligned Day-K')

    directions_sim_day0 = np.array([data_straight_sim_day0, data_right_sim_day0, data_left_sim_day0])

    return fig, directions_sim_day0


#0. Get the latent variables:
neural_file_list=[
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl",
# r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl",
# r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl",
# r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl",
# r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241119_1007_A\navtrainingsphere_Maui_20241119_1007_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\navtrainingsphere_Maui_20241121_0953_A_trials.pkl",
# r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\navtrainingsphere_Vino_20240827_0926_A_trials.pkl"

]
neural_file_0 = neural_file_list[0] 
neural_file_k = neural_file_list[1] 

latents_path_list=[
#Day0:
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\latents.pkl",

#DayK:
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\latents.pkl",

#DayK transformed to Day0, with offline data, in online simulation
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\test offline cca\latents_sim0_canonical.pkl"
]

latents_0_path = latents_path_list[0]
latents_k_path = latents_path_list[1]
# latents_1_online_aligned_path = latents_path_list[2]
latents_k_offline_aligned_path = latents_path_list[2]

targets_day0_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\targets_training.pkl"
# targets_day1_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\targets_training.pkl"
# targets_day2_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\targets_training.pkl"
# targets_day3_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\targets_training.pkl"
# targets_day4_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\targets_training.pkl"
targets_dayk_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\targets_training.pkl"
# targets_DIFMONKEY_day6_path=R"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\targets_training.pkl"


# PART 1 - Plot the Day0, DayK and the simulated Day0 (that is transformed from DayK)
#Each full_process() will show a figure:
fig_01, directions_sim_day0_01=full_process(latents_0_path, latents_k_path, targets_day0_path, targets_dayk_path, neural_file_0, neural_file_k)


# PART 2 - Plot the Day0, and all simulated Day0 together in 1 figure 
# (just to visualize that they are all similar between each other)

fig = plt.figure()
ax = plt.axes(projection='3d')

colors_0=['green', 'blue', 'red'] #Day0
colors_k=['darkgreen', 'navy', 'maroon'] #DayK
colors_off=['olive', 'teal', 'peru'] #offline results gotten on online simulation
colors_cor=['lightgreen', 'lightsteelblue', 'pink'] #DayK aligned, becoming Day0 "correct"

colors_04=['mediumseagreen', 'teal', 'indianred']
colors_05=['mediumseagreen', 'teal', 'indianred']

labels_0='Day0 Latents'
labels_k='DayK Latents'
labels_off='Simulation: Aligned offline'
labels_cor='"Correct" alignment of Day0(normal) and DayK(normal)'



latents_0 = process_latents(latents_0_path) 
latents_k = process_latents(latents_k_path) 
latents_k_offline_aligned = process_latents(latents_k_offline_aligned_path) 

n_trials_0= len(latents_0)
n_trials_k= len(latents_k)
# n_trials_on= len(latents_k_online_aligned)
n_trials_off= len(latents_k_offline_aligned)

with open(targets_day0_path, 'rb') as f:
    targets_0 = pickle.load(f)
with open(targets_dayk_path, 'rb') as f:
    targets_k = pickle.load(f)


#Separate the latents by direction:
data_straight_day0, data_right_day0, data_left_day0= separate_by_target(n_trials_0, targets_0, latents_0)
data_straight_dayk, data_right_dayk, data_left_dayk= separate_by_target(n_trials_k, targets_k, latents_k)
# data_straight_on_day0, data_right_on_day0, data_left_on_day0= separate_by_target(n_trials_on, targets_k, latents_k_online_aligned)
data_straight_off_day0, data_right_off_day0, data_left_off_day0= separate_by_target(n_trials_off, targets_k, latents_k_offline_aligned)


#Join for easier readibility:
directions_day0 = np.array([data_straight_day0, data_right_day0, data_left_day0])
directions_dayk = np.array([data_straight_dayk, data_right_dayk, data_left_dayk])
# directions_on_day0 = np.array([data_straight_on_day0, data_right_on_day0, data_left_on_day0])
directions_off_day0 = np.array([data_straight_off_day0, data_right_off_day0, data_left_off_day0])


#Plot the normal latent trajectories: Day0 (Normal) and DayK (normal)
plot_trajectories(ax, directions_day0[0], directions_day0[1], directions_day0[2], colors_0, labels_0)
plot_trajectories(ax, directions_dayk[0], directions_dayk[1], directions_dayk[2], colors_k, labels_k)

# #Plot the latents that were aligned in the simulation script (offline)
plot_trajectories(ax, directions_off_day0[0], directions_off_day0[1], directions_off_day0[2], colors_off, labels_off)
# plot_trajectories(ax, directions_on_day0[0], directions_on_day0[1], directions_on_day0[2], colors_03, labels_03)

#Plot the "correct" latent alignment trajectories (offline as well, aligned in this script)
plot_trajectories(ax, directions_sim_day0_01[0], directions_sim_day0_01[1], directions_sim_day0_01[2], colors_cor, labels_cor)

#There shouldn't be a difference between the latents aligned in this script and the ones 

plt.tight_layout()
plt.show()
