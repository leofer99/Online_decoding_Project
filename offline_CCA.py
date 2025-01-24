import numpy as np
import yaml
import xml.etree.ElementTree as ET
from tnsbmi.onlinedecoding import localconfig, tasks, util
from tnsbmi.tnsbmi import bintrials, dataconversion
from tnsbmi.tnsbmi import bintrials, nevdata
from tnsbmi.tnsbmi import modeling
import os
import pickle
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
from bcidecode.preprocessing.ratesTransformer import (EpochTransformer,
                                                      RatesTransformer)
from bcidecode.online.models import Model
from bcidecode.tools import ccaTools as cca

##############################################
# To align CCA, latents can't have NaN values and need to have the same shape.
# In this code, I align the Day0 trials with the DayK trials 
##############################################

def process_latents(neural_file):

    #load latent variables of model0:
    common_path=os.path.dirname(neural_file)
    latents_0_path= os.path.join(common_path, "latents.pkl")
    with open(latents_0_path, 'rb') as f:
        latents = pickle.load(f)

    #equalize size
    min_timebins = min( trial.shape[0] for trial in latents)
    latents_cut= [ trial[:min_timebins, :] for trial in latents ]
    latents_cut = np.array(latents_cut)    

    return latents_cut

def CCA_coefficients(latents_0, latents_k):
    
    #2.1 Ensure Day-0 and Day-K have the same number of trials:
    min_trials= min(latents_0.shape[0], latents_k.shape[0])

    #2.2 Ensure Day-0 and Day-K have the same number of timebins:
    min_timebins= min(latents_0.shape[1], latents_k.shape[1])

    latents_0_cut= latents_0[:min_trials, :min_timebins, :]
    latents_k_cut= latents_k[:min_trials, :min_timebins, :]

    latents_0_2d= latents_0_cut.reshape(-1, latents_0_cut.shape[-1])
    latents_k_2d= latents_k_cut.reshape(-1, latents_k_cut.shape[-1])

    #3. Perform CCA:
    A, B, r, U, V = cca.canoncorr(latents_0_2d, latents_k_2d, fullReturn=True)
    CCA=[A, B, min_timebins]

    return CCA

def CCA(CCA, latents_0, latents_k):
    U, _, Vh = linalg.svd(CCA[0], full_matrices=False, compute_uv=True, overwrite_a=False)
    U2, _,Vh2 = linalg.svd(CCA[1], full_matrices=False, compute_uv=True, overwrite_a=False)

    # latents_k_canonical=[]
    # latents_0_canonical=[]
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

def plot_trajectories(ax, data_straight, data_left, data_right, colors, labels):

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

def full_process(neural_file_0, neural_file_k, targets_day0_path, targets_dayk_path):

    #1. Equalize number of timebins across the trials:
    latents_0 = process_latents(neural_file_0)
    latents_k = process_latents(neural_file_k)
    n_trials_0= len(latents_0)
    n_trials_k= len(latents_k)

    # 2. CCA Coefficients + Apply CCA:
    CCA_ = CCA_coefficients(latents_0, latents_k)
    latents_sim0_canonical, latents_simk_canonical= CCA(CCA_, latents_0, latents_k)

    #3. Plot Results:
    # Identify the targets:
    with open(targets_day0_path, 'rb') as f:
        targets_0 = pickle.load(f)
    with open(targets_dayk_path, 'rb') as f:
        targets_k = pickle.load(f)

    data_straight_dayK, data_right_dayK, data_left_dayK= separate_by_target(n_trials_k, targets_k, latents_k)
    data_straight_day0, data_right_day0, data_left_day0= separate_by_target(n_trials_0, targets_0, latents_0)
    data_straight_sim_day0, data_right_sim_day0, data_left_sim_day0= separate_by_target(n_trials_k, targets_k, latents_sim0_canonical)
    data_straight_sim_dayk, data_right_sim_dayk, data_left_sim_dayk= separate_by_target(n_trials_0, targets_0, latents_simk_canonical)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    colors_day0=['green', 'blue', 'red']
    colors_sim_day0=['lightgreen', 'lightsteelblue', 'pink']
    colors_dayk=['darkgreen', 'navy', 'maroon']
    labels_day0= 'Day0'
    labels_sim_day0= 'Sim Day0'
    labels_dayk= 'DayK'

    plot_trajectories(ax, data_straight_dayK, data_left_dayK, data_right_dayK, colors_dayk, labels_dayk)
    plot_trajectories(ax, data_straight_sim_day0, data_left_sim_day0, data_right_sim_day0, colors_sim_day0, labels_sim_day0)
    plot_trajectories(ax, data_straight_day0, data_left_day0, data_right_day0, colors_day0, labels_day0)
    
    date_day0 = neural_file_0.split("navtrainingsphere_")[-1].split(".pkl")[0].split('_')[1]
    date_dayk = neural_file_k.split("navtrainingsphere_")[-1].split(".pkl")[0].split('_')[1]

    ax.set_title('CCA: Comparing Day-K ('+date_dayk+'), Day-0 ('+date_day0+') and Aligned Day-K')

    directions_sim_day0 = np.array([data_straight_sim_day0, data_left_sim_day0, data_right_sim_day0])

    return fig, directions_sim_day0


#0. Get the latent variables:
neural_file_list=[
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241119_1007_A\navtrainingsphere_Maui_20241119_1007_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\navtrainingsphere_Maui_20241121_0953_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\navtrainingsphere_Vino_20240827_0926_A_trials.pkl"
]

neural_file_0 = neural_file_list[0] 
neural_file_1 = neural_file_list[1] 
neural_file_2 = neural_file_list[2] 
neural_file_3 = neural_file_list[3] 
# neural_file_4 = neural_file_list[4] 
neural_file_5 = neural_file_list[5] 
neural_file_6 = neural_file_list[6] #DIFFERENT MONKEY


targets_day0_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\targets_training.pkl"
targets_day1_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\targets_training.pkl"
targets_day2_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\targets_training.pkl"
targets_day3_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\targets_training.pkl"
# targets_day4_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\targets_training.pkl"
targets_day5_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\targets_training.pkl"
targets_DIFMONKEY_day6_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\targets_training.pkl"

neural_file_0=neural_file_0
targets_day0_path=targets_day0_path



# PART 1 - Plot the Day0, DayK and the simulated Day0 (that is transformed from DayK)
#Each full_process() will show a figure:

_, directions_sim_day0_00 = full_process(neural_file_0, neural_file_0, targets_day0_path, targets_day0_path) #works, but not great
# fig_01, directions_sim_day0_01 = full_process(neural_file_0, neural_file_1, targets_day0_path, targets_day1_path) #works, but not great
# fig_02, directions_sim_day0_02 = full_process(neural_file_0, neural_file_2, targets_day0_path, targets_day2_path) # a good example of it working!
# fig_03, directions_sim_day0_03 = full_process(neural_file_0, neural_file_3, targets_day0_path, targets_day3_path) #works
# fig_05, directions_sim_day0_05 = full_process(neural_file_0, neural_file_5, targets_day0_path, targets_day5_path) #works
fig_06, directions_sim_day0_06 = full_process(neural_file_0, neural_file_6, targets_day0_path, targets_DIFMONKEY_day6_path) #works



# PART 2 - Plot the Day0, and all simulated Day0 together in 1 figure 
# (just to visualize that they are all similar between each other)

fig = plt.figure()
ax = plt.axes(projection='3d')

colors_00=['green', 'blue', 'red']
colors_01=['olive', 'violet', 'peru']
colors_02=['lightgreen', 'lightsteelblue', 'pink']
colors_03=['darkgreen', 'navy', 'maroon']
colors_05=['mediumseagreen', 'teal', 'indianred']
colors_06=['mediumseagreen', 'teal', 'indianred']

labels_00='00'
labels_01='01'
labels_02='02'
labels_03='03'
labels_05='05'
labels_06='06'

# Adds to the figure the Day0 latents (SimDay00 is just the regular Day0 latents)

plot_trajectories(ax, directions_sim_day0_00[0], directions_sim_day0_00[1], directions_sim_day0_00[2], colors_00, labels_00)
# plot_trajectories(ax, directions_sim_day0_01[0], directions_sim_day0_01[1], directions_sim_day0_01[2], colors_01, labels_01)
# plot_trajectories(ax, directions_sim_day0_02[0], directions_sim_day0_02[1], directions_sim_day0_02[2], colors_02, labels_02)
# plot_trajectories(ax, directions_sim_day0_03[0], directions_sim_day0_03[1], directions_sim_day0_03[2], colors_03, labels_03)
# plot_trajectories(ax, directions_sim_day0_05[0], directions_sim_day0_05[1], directions_sim_day0_05[2], colors_05, labels_05)

#DIFFERENT MONKEY
# Adds to the figure the simulated Day0 latents (transformed from DayK)
# (SimDay0_06 is the DayK latents transformed into the Day0 space)
plot_trajectories(ax, directions_sim_day0_06[0], directions_sim_day0_06[1], directions_sim_day0_06[2], colors_06, labels_06)

ax.set_title('Comparison of all aligned_day0s')

plt.tight_layout()
plt.show()

