from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import tnsbmi.tnsbmi.dataconversion
import tnspython.tns.os
import pickle
import time
import re
# from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import numpy as np
from bcidecode.kalman.filters import PSID_DecoderPositions, KalmanRegressor, PSID_DecoderVelocities
from bcidecode.kalman.pipelines import PSIDPipelineDec
from bcidecode.modeling.defaults import (FILTERING_SCORERS, PREPROCESSORS,
                                         REGRESSORS)
from bcidecode.modeling.pipeline_builder import PipelineBuilder
from bcidecode.online.models import Model
from bcidecode.optimization.axUtils import load_best_config
from bcidecode.preprocessing.data import cursor_preprocessing, load_data
from bcidecode.preprocessing.ratesTransformer import (EpochTransformer,
RatesTransformer)
# from bcidecode.preprocessing.lfpTransformer_PSID import LFPTransformer
from bcidecode.preprocessing.PSIDmethod import psdi_method
from numpy.lib.arraysetops import unique
from omegaconf import OmegaConf
from pandas.core.indexing import convert_from_missing_indexer_tuple
from scipy.spatial import distance
from sklearn.pipeline import Pipeline
from tnsbmi.tnsbmi import bintrials, nevdata
from pathlib import Path
import shutil
from bcidecode.preprocessing.kinematics import compute_modified_velocities
from bcidecode.online.utils import _filterSingleTrial
import logging
from scipy.spatial.distance import euclidean
from tnsbmi.onlinedecoding import localconfig
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA, KernelPCA
from bcidecode.dPCA.python.dPCA import dPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap

def processData(neural_file, latents_path, type_of_latent):
    trials, taskparameters = load_data(neural_file)
    task = taskparameters["task"]
    monkey = re.search(r'_(\w+)_\d{8}_\d{4}_\w_', neural_file).group(1) #monkey identifier
    logging.info(f"Number of trials {len(trials)}")  
    # if task == "reach":
    #     trials = trials[-120:] #selects the last 120 trials
    # else:
    trials = trials[-90:]
    targets = [np.array([trial.targetPosition[0], trial.targetPosition[1], trial.targetPosition[2]]) for trial in trials]


    if os.path.exists(latents_path):
        print("File exists.")
    else:
        print("File does not exist.")

    with open(latents_path, 'rb') as f:
        latents = pickle.load(f)

    if type_of_latent=='DPAD':
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

#None only shows the 3 leading neural modes
def none(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    

    ax = fig.add_subplot(2, 3, 1, projection='3d')

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

    ax.set_title('None: Chooses only 3 of the latent variables')
    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')
    ax.set_zlabel('LD 3')

    # plt.legend()
    # plt.show()

def PCA_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):

    ax1 = fig.add_subplot(2, 3, 2, projection='3d')

    #Prepare data and perform PCA:
    combined_data = np.concatenate((average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents), axis=0)
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

    pca = PCA(n_components=3)
    transformed_data = pca.fit_transform(combined_data)

    transformed_left = transformed_data[:len(average_left_trial_latents)]
    transformed_straight = transformed_data[len(average_left_trial_latents):(len(average_left_trial_latents)+len(average_straight_trial_latents))]
    transformed_right = transformed_data[(len(average_left_trial_latents)+len(average_straight_trial_latents)):]

    # Plot each trajectory:
    for trajectory_data, color, label in zip(
        [transformed_left, transformed_straight, transformed_right],
        ['b', 'g', 'r'],
        ['Left', 'Straight', 'Right']
    ):
        ax1.scatter3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], marker='o', c=color, label=label)
        ax1.scatter3D(trajectory_data[5, 0], trajectory_data[5, 1], trajectory_data[5, 2], marker='X', c=color, s=100)
        ax1.plot3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], color)

    ax1.set_title('PCA')
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')

    # plt.legend()
    # plt.show()

    return ax1


def kPCA_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    
    ax2 = fig.add_subplot(2, 3, 6, projection='3d')

    #with PCA:
    combined_data = np.concatenate((average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents), axis=0)
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

    kernel_pca = KernelPCA(
        n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
    )
    transformed_data = kernel_pca.fit_transform(combined_data)

    transformed_left = transformed_data[:len(average_left_trial_latents)]
    transformed_right = transformed_data[len(average_left_trial_latents):(len(average_left_trial_latents)+len(average_right_trial_latents))]
    transformed_straight = transformed_data[(len(average_left_trial_latents)+len(average_right_trial_latents)):]

    # Plot each trajectory
    for trajectory_data, color, label in zip(
        [transformed_left, transformed_straight, transformed_right],
        ['b', 'g', 'r'],
        ['Left', 'Straight', 'Right']
    ):
        ax2.scatter3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], marker='o', c=color, label=label)
        ax2.scatter3D(trajectory_data[5, 0], trajectory_data[5, 1], trajectory_data[5, 2], marker='X', c=color, s=100)
        ax2.plot3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], color)

    ax2.set_title('kPCA')
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')

    plt.legend()
    plt.show()


def TSNE_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    
    ax1 = fig.add_subplot(2, 3, 3, projection='3d')

    # Combine your data
    combined_data = np.concatenate((average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents), axis=0)
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

    # Apply t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    transformed_data = tsne.fit_transform(combined_data)

    # Split transformed data back into the original categories for plotting
    transformed_left = transformed_data[:len(average_left_trial_latents)]
    transformed_straight = transformed_data[len(average_left_trial_latents):(len(average_left_trial_latents)+len(average_straight_trial_latents))]
    transformed_right = transformed_data[(len(average_left_trial_latents)+len(average_straight_trial_latents)):]

    # Plot each trajectory
    for trajectory_data, color, label in zip(
        [transformed_left, transformed_straight, transformed_right],
        ['b', 'g', 'r'],
        ['Left', 'Straight', 'Right']
    ):
        ax1.scatter3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], marker='o', c=color, label=label)
        ax1.scatter3D(trajectory_data[5, 0], trajectory_data[5, 1], trajectory_data[5, 2], marker='X', c=color, s=100)
        ax1.plot3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], color)

    # Set labels and legend
    ax1.set_title('t-SNE')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_zlabel('t-SNE 3')

    # plt.legend()
    # plt.show()

    return ax1

def Isomap_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    
    ax1 = fig.add_subplot(2, 3, 4, projection='3d')

    combined_data = np.concatenate((average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents), axis=0)
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

    n_neighbors = 14  #4, 5, 8
    isomap = Isomap(n_components=3, n_neighbors=n_neighbors)
    transformed_data = isomap.fit_transform(combined_data)

    # Split transformed data back into the original categories for plotting
    transformed_left = transformed_data[:len(average_left_trial_latents)]
    transformed_straight = transformed_data[len(average_left_trial_latents):(len(average_left_trial_latents)+len(average_straight_trial_latents))]
    transformed_right = transformed_data[(len(average_left_trial_latents)+len(average_straight_trial_latents)):]

    # Plot each trajectory
    for trajectory_data, color, label in zip(
        [transformed_left, transformed_straight, transformed_right],
        ['b', 'g', 'r'],
        ['Left', 'Straight', 'Right']
    ):
        ax1.scatter3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], marker='o', c=color, label=label)
        ax1.scatter3D(trajectory_data[5, 0], trajectory_data[5, 1], trajectory_data[5, 2], marker='X', c=color, s=100)
        ax1.plot3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], color)

    # Set labels and legend
    ax1.set_title('Isomap')
    ax1.set_xlabel('Isomap 1')
    ax1.set_ylabel('Isomap 2')
    ax1.set_zlabel('Isomap 3')

    plt.legend()
    # plt.show()

    return ax1

def LDA_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    
    ax= fig.add_subplot(2, 3, 5, projection='3d')
    
    average_left_trial_latents = average_left_trial_latents[~np.isnan(average_left_trial_latents).any(axis=1)] #keeps only rows that contain 0 nans
    average_straight_trial_latents = average_straight_trial_latents[~np.isnan(average_straight_trial_latents).any(axis=1)] #keeps only rows that contain 0 nans
    average_right_trial_latents = average_right_trial_latents[~np.isnan(average_right_trial_latents).any(axis=1)] #keeps only rows that contain 0 nans

    combined_X=  np.concatenate((average_left_trial_latents, average_straight_trial_latents, average_right_trial_latents), axis=0)

    y_left = [['Left'] for _ in range(average_left_trial_latents.shape[0])]
    y_straight = [['Straight'] for _ in range(average_straight_trial_latents.shape[0])]
    y_right = [['Right'] for _ in range(average_right_trial_latents.shape[0])]

    left_labels = np.array(y_left).reshape(-1, 1)
    straight_labels = np.array(y_straight).reshape(-1, 1)
    right_labels = np.array(y_right).reshape(-1, 1)

    combined_y=  np.concatenate((left_labels, straight_labels, right_labels), axis=0)

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(combined_X, combined_y)

    lda_left= X_lda[:len(y_left),:]
    lda_straight= X_lda[len(y_left):(len(y_left)+len(y_straight)),:]
    lda_right= X_lda[(len(y_left)+len(y_straight)):, :]


    # Plot each trajectory
    for trajectory_data, color, label in zip(
        [lda_left, lda_straight, lda_right],
        ['b', 'g', 'r'],
        ['Left', 'Straight', 'Right']
    ):
        ax.scatter3D(trajectory_data[5:, 0], trajectory_data[5:, 1], marker='o', c=color, label=label)
        ax.scatter3D(trajectory_data[5, 0], trajectory_data[5, 1], marker='X', c=color, s=100)
        ax.plot3D(trajectory_data[5:, 0], trajectory_data[5:, 1], color)

    # Add labels and title
    ax.set_title('LDA')
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    #ax.set_zlabel('LD3')
    ax.set_title('LDA: 2D Visualization of Classes')

    # plt.legend()
    # plt.show()

def LLE_(fig2, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents):
    
    # Combine your data
    combined_data = np.concatenate((average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents), axis=0)
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

    params = {
    "n_components": 3,
    "eigen_solver": "auto",
    "random_state": 0,
    }

##################
    # #Define N
    # neighbors_a=20
    # neighbors_b=50
    # neighbors_c=40
    # neighbors_d=45

    # # Apply LLE
    # lle_standard_a = LocallyLinearEmbedding(method="ltsa", n_neighbors=neighbors_a, **params)
    # data_standard_a = lle_standard_a.fit_transform(combined_data)
    # lle_standard_b = LocallyLinearEmbedding(method="ltsa", n_neighbors=neighbors_b, **params)
    # data_standard_b = lle_standard_b.fit_transform(combined_data)
    # lle_standard_c = LocallyLinearEmbedding(method="ltsa", n_neighbors=neighbors_c, **params)
    # data_standard_c = lle_standard_c.fit_transform(combined_data)
    # lle_standard_d = LocallyLinearEmbedding(method="ltsa", n_neighbors=neighbors_d, **params)
    # data_standard_d = lle_standard_d.fit_transform(combined_data)
##################

    #Define N
    neighbors_standard=25  #N= 20-30+ are the cleanest
    neighbors_ltsa=35  #N=20-25 or 30+ are the cleanest
    neighbors_hessian=45 #N=20-25 are the cleanest or 40+
    neighbors_modified=20  #N=10, 30+

    lle_standard = LocallyLinearEmbedding(method="standard", n_neighbors=neighbors_standard, **params)
    data_standard = lle_standard.fit_transform(combined_data)
    lle_ltsa = LocallyLinearEmbedding(method="ltsa", n_neighbors=neighbors_ltsa, **params)
    data_ltsa = lle_ltsa.fit_transform(combined_data)
    lle_hessian = LocallyLinearEmbedding(method="hessian", n_neighbors=neighbors_hessian, **params)
    data_hessian = lle_hessian.fit_transform(combined_data)
    lle_mod = LocallyLinearEmbedding(method="modified", n_neighbors=neighbors_modified, **params)
    data_mod = lle_mod.fit_transform(combined_data)

    lle_methods = [
    (f"Standard locally linear embedding N={neighbors_standard}", data_standard),
    (f"Local tangent space alignment N={neighbors_ltsa}", data_ltsa),
    (f"Hessian eigenmap N={neighbors_hessian}", data_hessian),
    (f"Modified locally linear embedding N={neighbors_modified}", data_mod),
]

    # lle_methods = [
    #     (f"Standard N={neighbors_a}", data_standard_a),
    #     (f"Standard N={neighbors_b}", data_standard_b),
    #     (f"Standard N={neighbors_c}", data_standard_c),
    #     (f"Standard N={neighbors_d}", data_standard_d),
    # ]

    for i, (name, transformed_data) in enumerate(lle_methods):
        ax1 = fig2.add_subplot(2, 2, i + 1, projection='3d')

        # Split transformed data back into the original categories for plotting
        transformed_left = transformed_data[:len(average_left_trial_latents)]
        transformed_straight = transformed_data[len(average_left_trial_latents):(len(average_left_trial_latents)+len(average_straight_trial_latents))]
        transformed_right = transformed_data[(len(average_left_trial_latents)+len(average_straight_trial_latents)):]

        # Plot each trajectory
        for trajectory_data, color, label in zip(
            [transformed_left, transformed_straight, transformed_right],
            ['b', 'g', 'r'],
            ['Left', 'Straight', 'Right']
        ):
            ax1.scatter3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], marker='o', c=color, label=label)
            ax1.scatter3D(trajectory_data[5, 0], trajectory_data[5, 1], trajectory_data[5, 2], marker='X', c=color, s=100)
            ax1.plot3D(trajectory_data[5:, 0], trajectory_data[5:, 1], trajectory_data[5:, 2], color)

            # Set labels and legend
        ax1.set_title(name)
        ax1.set_xlabel('LLE 1')
        ax1.set_ylabel('LLE 2')
        ax1.set_zlabel('LLE 3')

    plt.legend()
    # plt.show()

    return ax1


if __name__ == "__main__":


    logging.basicConfig(level=logging.INFO)

    #new data:
    # neural_file= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl""C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl"
    # latents_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\latents.pkl"

    neural_file_list = [
        r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl",
        r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl",
        r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl",
        r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl",
        r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\navtrainingsphere_Maui_20241121_0953_A_trials.pkl"
    ]

    neural_file= neural_file_list[1]
    dest_dir = os.path.dirname(neural_file)
    PSID_latents_path= rf"{dest_dir}\latents.pkl"
    DPAD_latents_path= rf"{dest_dir}\DPAD_latents.pkl" #only for files 0,1,2
    PCA_latents_path= rf"{dest_dir}\PCA_latents.pkl" #only for files 0,1,2

    type_of_latent='PSID' 
    # type_of_latent='DPAD' 
    # type_of_latent='PCA' 

    if type_of_latent=='PSID':
        latents_path=PSID_latents_path
    elif type_of_latent=='DPAD': #fix error
        latents_path=DPAD_latents_path
    elif type_of_latent=='PCA':
        latents_path=PCA_latents_path

    extracted_part = neural_file.split("navtrainingsphere_")[-1].split(".pkl")[0]
    date = extracted_part.split('_')[1]

    left_trials, right_trials, straight_trials = processData(neural_file, latents_path, type_of_latent)

    average_left_trial_latents = np.mean(left_trials, axis=0)
    average_straight_trial_latents = np.mean(straight_trials, axis=0)
    average_right_trial_latents = np.mean(right_trials, axis=0)


    #DR Techniques (from 6D-data to 3D-data):

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 12), facecolor="white")
    fig.suptitle(f"3D Visualizations of {type_of_latent} latents - File: {date}", size=13)

    axnone=none(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)
    axPCA= PCA_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)
    axTSNE=TSNE_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)
    axIso=Isomap_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)
    axLDA=LDA_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)
    axkPCA= kPCA_(fig, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)

    fig2 = plt.figure(figsize=(9, 9), facecolor="white")
    fig2.suptitle(f"3D Visualizations of {type_of_latent} latents - File: {date}", size=13)
    axLLE=LLE_(fig2, average_left_trial_latents, average_right_trial_latents, average_straight_trial_latents)


    plt.tight_layout()
    plt.show()



