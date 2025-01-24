from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# print(sys.path)

import tnsbmi.tnsbmi.dataconversion
import tnspython.tns.os
import pickle
import time
import re
from bs4 import BeautifulSoup
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

from sklearn.decomposition import PCA
from bcidecode.dPCA.python.dPCA import dPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
import scipy.linalg as linalg
import random

from bcidecode.tools import utilityTools as utility
from bcidecode.tools import dataTools as dt
from bcidecode.tools import ccaTools as cca

def processData(neural_file, latents_path):

    """
    Function that determines information about the data: 
    monkey, task and the latent variables 
    """

    trials, taskparameters = load_data(neural_file)
    task = taskparameters["task"]
    monkey = re.search(r'_(\w+)_\d{8}_\d{4}_\w_', neural_file).group(1) #monkey identifier
    logging.info(f"Number of trials {len(trials)}")  
    # if task == "reach":
    #     trials = trials[-120:] #selects the last 120 trials
    # else:
    #     trials = trials[-90:]
    targets = [np.array([trial.targetPosition[0], trial.targetPosition[1], trial.targetPosition[2]]) for trial in trials]


    if os.path.exists(latents_path):
        print("File exists.")
    else:
        print("File does not exist.")

    with open(latents_path, 'rb') as f:
        latents = pickle.load(f)

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

def add_gridspec_abs(fig, nrows=1, ncols=1, left=0, bottom=0, right=None, top=None, width=1, height=1, **kwargs):
    """
    Equivalent to `fig.add_gridspec` except  all the inputs are in absolute values in inches.
    """
    figw, figh = fig.get_size_inches()
    if right is None or top is None:  # aligned on bottom left
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, 
                              left=left/figw, bottom=bottom/figh,
                              right=(left/figw) + (width/figw),
                              top=(bottom/figh)+(height/figh),
                              **kwargs
                             )
    else:  # aligned on top right
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, 
                              left=(right/figw)-(width/figw), 
                              bottom=(top/figh)-(height/figh),
                              right=right/figw, top=top/figh,
                              **kwargs
                             )
    
    return gs

def get_ccs_upper_bound(data, n_components):

    n_iter=1000
    min_trials= data.shape[0]
    trialList = np.arange(min_trials)

    #get ccs
    CCsU=[]
    # for sessionData in data:
    r = []
    for n in range(n_iter):
            random.shuffle(trialList)
            # Create 2 non-overlapping randomised trials
            trial1 = trialList[:min_trials//2]
            trial2 = trialList[-(min_trials//2):]
            data1 = np.reshape(data[trial1,:,:], (-1,n_components))
            data2 = np.reshape(data[trial2,:,:], (-1,n_components))

            r.append(cca.canoncorr(data1, data2))
    CCsU.append(r)
    CCsU = np.array(CCsU)
    CCsU = np.percentile(CCsU, 99, axis=1).T

    return CCsU

def plot_fig_CCA(CCAdata_days, gs, full_data, date0, datek):

    fig1=gs.figure
    axes=[]
    
    #UNALIGNED PART
    for i, (day, _) in enumerate(CCAdata_days.items()):

        left_trials= full_data[day]['Left']
        straight_trials= full_data[day]['Straight']
        right_trials= full_data[day]['Right']

        ax1 = fig1.add_subplot(gs[i], projection='3d',fc='None')
        ax2 = fig1.add_subplot(gs[i+5], projection='3d',fc='None')
        axes.append(ax1)
        axes.append(ax2)

        #Plots each target direction: (separately) 
        for target_trials, color, label in zip(
            [left_trials, straight_trials, right_trials],
            ['b', 'g', 'r'],
            ['Left', 'Straight', 'Right']
        ):   
            ex= np.mean(target_trials, axis=0)
            ax1.plot(ex[5:,0],ex[5:,1],ex[5:,2],color=color,lw=1, label=label)
            ax1.view_init(60,-47)
            ax2.plot(ex[5:,3],ex[5:,4],ex[5:,5],color=color,lw=1, label=label)
            ax2.view_init(60,-47)

    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0))

    data1=CCAdata_days['Day0']
    data2=CCAdata_days['Day1']

    min_data= min(data1.shape[0], data2.shape[0])
    data1_=data1[:min_data, :]
    data2_=data2[:min_data, :]

    A,B,*_ = cca.canoncorr(data1_, data2_, fullReturn=True)
    coef_=np.array([A, B])

    data_aligned = {
        'Day0': {
            'Left': [],
            'Straight': [],
            'Right': []
        },
        'Day1': {
            'Left': [],
            'Straight': [],
            'Right': []
        }
    }

    #ALIGNED PART
    for i,(day, _) in enumerate(CCAdata_days.items()):
        U, _, Vh = linalg.svd(coef_[i], full_matrices=False, compute_uv=True, overwrite_a=False)
        ax3 = fig1.add_subplot(gs[i+3], projection='3d',fc='None')
        ax4 = fig1.add_subplot(gs[i+8], projection='3d',fc='None')
        axes.append(ax3)
        axes.append(ax4)

        left_trials= full_data[day]['Left']
        straight_trials= full_data[day]['Straight']
        right_trials= full_data[day]['Right']

        #Plots each target direction: (separately) 
        for target_trials, color, label in zip(
            [left_trials, straight_trials, right_trials],
            ['b', 'g', 'r'],
            ['Left', 'Straight', 'Right']
        ):   
            ex= np.mean(target_trials, axis=0)
            ex = ex @ U @ Vh

            ax3.plot(ex[5:,0],ex[5:,1],ex[5:,2],color=color,lw=1, label=label)
            ax3.view_init(60,-47)
            ax4.plot(ex[5:,3],ex[5:,4],ex[5:,5],color=color,lw=1, label=label)
            ax4.view_init(60,-47)

            data_aligned[day][label]= ex

    fig1.suptitle(f'CCA Alignment of Day0 ({date0}) and DayK ({datek})')

    titles=[f'Day0 unaligned',f'Day0 unaligned',
            f'DayK unaligned',f'DayK unaligned', 
            f'Day0 aligned',f'Day0 unaligned',
            f'DayK aligned',f'DayK unaligned',
            ]
    
    labels = ['NM','NM','NM','NM',
              'Aligned NM','Aligned NM','Aligned NM','Aligned NM',
              ]

    for i, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(titles[i], pad=0, loc='center')

        if i%2==0:
            ax.set_xlabel(f'{labels[i]}1', labelpad=-15)
            ax.set_ylabel(f'{labels[i]}2', labelpad=-15)
            ax.set_zlabel(f'{labels[i]}3', labelpad=-15)
        else:
            ax.set_xlabel(f'{labels[i]}4', labelpad=-15)
            ax.set_ylabel(f'{labels[i]}5', labelpad=-15)
            ax.set_zlabel(f'{labels[i]}6', labelpad=-15)

    # ======== Add the arrow
    ax = fig1.add_subplot(2,5,3, fc='None')
    ax = utility.phantom_axes(ax)
    ax.arrow(0,0,1,0,length_includes_head=True, width=.005, head_width=.015,head_length=.1, ec='k', fc='k')
    ax.set_ylim([-.1,.1])
    ax.set_xlim([-.5,1.1])

    ax.text(0.5,0.01,'CCA', ha='center', va='bottom')
    ax.text(0.5,-0.01,'(alignment)', ha='center', va='top')

    ax = fig1.add_subplot(2,5,8, fc='None')
    ax = utility.phantom_axes(ax)
    ax.arrow(0,0,1,0,length_includes_head=True, width=.005, head_width=.015,head_length=.1, ec='k', fc='k')
    ax.set_ylim([-.1,.1])
    ax.set_xlim([-.5,1.1])

    ax.text(0.5,0.01,'CCA', ha='center', va='bottom')
    ax.text(0.5,-0.01,'(alignment)', ha='center', va='top')
    
    return fig1, data_aligned

def plot_dims_CCA(CCAdata_days, min_time, full_data, date0, datek):

    time_bins= np.arange(5, min_time)
    n_components=6
    figs=[]
    color=np.array(['lightsteelblue', 'rosybrown', 'navy', 'darkred'])

    data1=CCAdata_days['Day0']
    data2=CCAdata_days['Day1']
    A,B,*_ = cca.canoncorr(data1, data2, fullReturn=True)
    coef_=np.array([A, B])
    U, _, Vh = linalg.svd(coef_[0], full_matrices=False, compute_uv=True, overwrite_a=False) #for day0
    U_, _, Vh_ = linalg.svd(coef_[1], full_matrices=False, compute_uv=True, overwrite_a=False) #for dayK
    
    #Different figure for each direction
    for _, direction in zip(   
            [left_trials, straight_trials, right_trials],
            ['Left', 'Straight', 'Right']
            ): 
        
        fig = plt.figure(figsize=(10,9))
        fig.suptitle(f"Neural Modes of Day0 ({date0}) and DayK ({datek}) - Direction: {direction}", fontsize=11)

        axes=[]
        gs= add_gridspec_abs(fig, nrows=3, ncols=2, left=0.8, bottom=0.5, width=9, height=8)

        for dim in range(n_components):
            ax = fig.add_subplot(gs[dim], fc='None')
            # ax.set_title(f'Dimension {dim+1}', pad=0, loc='center')
            ax.set_xlabel('Time bins', labelpad=0)
            ax.set_ylabel(f'Neural Mode {dim+1}', labelpad=0)
            axes.append(ax)

            unaligned_day0= full_data['Day0'][direction]
            unaligned_dayk= full_data['Day1'][direction]

            unaligned_trials0= np.mean(unaligned_day0, axis=0)
            unaligned_trialsk= np.mean(unaligned_dayk, axis=0)

            aligned_trials0 = unaligned_trials0 @ U @ Vh @ np.linalg.pinv(U_ @ Vh_)
            aligned_trialsk = unaligned_trialsk @ U_ @ Vh_ @ np.linalg.pinv(U @ Vh)


            ax.plot(time_bins, unaligned_trials0[5:,dim], color=color[0],lw=1, marker='o', label='Unaligned Day0')
            ax.plot(time_bins, unaligned_trialsk[5:,dim], color=color[1],lw=1, marker='o', label='Unaligned DayK')
            ax.plot(time_bins, aligned_trials0[5:,dim], color=color[2],lw=1, marker='x', label='Aligned Day0')
            ax.plot(time_bins, aligned_trialsk[5:,dim], color=color[3],lw=1, marker='x', label='Aligned DayK')

        ax.legend(loc='center')
        plt.legend()

    
    return axes

def plot_graph_CCA(D_CCAdata_days, CCAdata_days,  data_aligned, n_components, date0, datek):
    
    data1=CCAdata_days['Day0']
    data2=CCAdata_days['Day1']
    D_CCAdata_day0= D_CCAdata_days['Day0']
    D_CCAdata_day1= D_CCAdata_days['Day1']

    #Part 1: Get CCs
    # allCCs = cca.canoncorr(data1, data2)
    ccs=[]
    ccs.append(cca.canoncorr(data1, data2))
    allCCs = np.array(ccs).T 

    #Part 2: Get Upper
    CCsU_1 = get_ccs_upper_bound(D_CCAdata_day0, n_components)
    CCsU_2 = get_ccs_upper_bound(D_CCAdata_day1, n_components)
    CCsU= np.array([CCsU_1, CCsU_2])

    #Part 3: Get Lower
    n_iter=9000
    # rng = np.random.default_rng(12345)
    rng = np.random.default_rng(np.random.SeedSequence(12345))
    CCsL=[]

    # for D_CCAdata_day0_, D_CCAdata_day1_ in zip(D_CCAdata_day0,D_CCAdata_day1):
    r = []
    for n in range(n_iter):
            D_CCAdata_day0_sh = rng.permutation(D_CCAdata_day0)
            D_CCAdata_day1_sh = rng.permutation(D_CCAdata_day1)

            data1 = np.reshape(D_CCAdata_day0_sh, (-1,n_components))
            data2 = np.reshape(D_CCAdata_day1_sh, (-1,n_components))
            r.append(cca.canoncorr(data1, data2))
    CCsL.append(r)
    CCsL = np.array(CCsL)
    CCsL = np.percentile(CCsL, 1, axis=1).T

    #Part 4: Get CCs for aligned
    data1_aligned= np.concatenate([data_aligned['Day0']['Left'], data_aligned['Day0']['Straight'], data_aligned['Day0']['Right']])
    data2_aligned= np.concatenate([data_aligned['Day1']['Left'], data_aligned['Day1']['Straight'], data_aligned['Day1']['Right']])

    ccs_aligned=[]
    ccs_aligned.append(cca.canoncorr(data1_aligned, data2_aligned))
    allCCs_aligned = np.array(ccs_aligned).T 

    #Plotting
    fig, ax = plt.subplots(figsize=(7,7))
    fig.suptitle(f'Canonical Correlation of each Neural Mode - Day0 ({date0}) and DayK ({datek})', fontsize=11)


    x_ = np.arange(1,n_components+1)
    ax.plot(x_, allCCs_aligned, color='green', marker = 'o', label=f'Comparing Day0 and DayK (Aligned)')
    ax.plot(x_, allCCs, color='red', marker = 'o', label=f'Comparing Day0 and DayK (Unaligned)')
    ax.plot(x_, CCsU[0, :], color='lightblue', marker = '<', ls='--', label=f'Comparing within Day0')
    ax.plot(x_, CCsU[1, :], color='darkblue', marker = '<', ls='--', label=f'Comparing within DayK')
    ax.plot(x_, CCsL, color='black', marker = '>', ls=':', label=f'Control')

    ax.set_ylim([-.05,1])
    ax.set_xlim([.6, n_components+.6])
    ax.set_xlabel('Neural Modes')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_title(f'{defs.areas[2]} Alignment')
    ax.legend(loc=(.55,.67))
    ax.set_ylabel('Canonical Correlation')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds([1, n_components])
    ax.spines['left'].set_bounds([0,1])

    plt.legend()

    return fig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_components=6
    n_targets=3

    #new: PSID
    neural_day08_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl"
    neural_day12_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl"
    neural_day13_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl"
    neural_day18_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl"
    neural_day21_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\navtrainingsphere_Maui_20241121_0953_A_trials.pkl"
    # neural_diffmonkey_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\navtrainingsphere_Vino_20240827_0926_A_trials.pkl"

    latents_day08_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\latents.pkl"
    latents_day12_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\latents.pkl"
    latents_day13_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\latents.pkl"
    latents_day18_path= r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\latents.pkl"
    latents_day21_path=r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\latents.pkl"
    # latents_diffmonkey_path=  r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\latents.pkl"

    days = ['Day0', 'Day1']

    neural_file = {
        'Day0': neural_day08_path,
        'Day1': neural_day21_path 
    }

    latents_paths = {
        'Day0': latents_day08_path,
        'Day1': latents_day21_path 
    }

    # Initialize dictionaries to store day_data and averages
    full_data={} #Shape: (day x monkey x target x trials x time x latents)
    average_latents = {}

    # full_data['Day0']['Left']
    for day in days:
        left_trials, right_trials, straight_trials = processData(neural_file[day], latents_paths[day])
        
        full_data[day] = {
            'Left': left_trials, 
            'Right': right_trials, 
            'Straight': straight_trials
        }

    #CCA ALIGNMENT:
    latents_day0 = {'Left': full_data['Day0']['Left'],
                    'Straight': full_data['Day0']['Straight'],
                    'Right': full_data['Day0']['Right']}

    latents_day1 = {'Left': full_data['Day1']['Left'],
                    'Straight': full_data['Day1']['Straight'],
                    'Right': full_data['Day1']['Right']}

    #REMOVE TIMEBINS THAT HAVE NAN VALUES: (cant be used with CCA)
    nan_timebins0 = {key: np.any(np.isnan(value), axis=(0, 2)) for key, value in latents_day0.items()}
    nan_timebins1 = {key: np.any(np.isnan(value), axis=(0, 2)) for key, value in latents_day1.items()}
    latents_day0_clean = {key: np.array(value)[:, ~nan_timebins0[key], :] for key, value in latents_day0.items()}
    latents_day1_clean = {key: np.array(value)[:, ~nan_timebins1[key], :] for key, value in latents_day1.items()}

    #CUT DATA: (ensure all days have same number of trials and timebins)
    #(also needs same number for each target?)
    min_trials = min(latents_day0_clean['Left'].shape[0], latents_day1_clean['Left'].shape[0],
                     latents_day0_clean['Straight'].shape[0], latents_day1_clean['Straight'].shape[0],
                     latents_day0_clean['Right'].shape[0], latents_day1_clean['Right'].shape[0]
                     )
    min_time = min(latents_day0_clean['Left'].shape[1], latents_day1_clean['Left'].shape[1],
                   latents_day0_clean['Straight'].shape[1], latents_day1_clean['Straight'].shape[1],
                   latents_day0_clean['Right'].shape[1], latents_day1_clean['Right'].shape[1]
                   )
    
    n_components = latents_day0_clean['Left'].shape[2]  # Assuming all parts have the same number of components

    latents_day0_cut = {key: value[:min_trials, :min_time, :] for key, value in latents_day0_clean.items()}
    latents_day1_cut = {key: value[:min_trials, :min_time, :] for key, value in latents_day1_clean.items()}

    D_CCAdata_day0=np.concatenate(list(latents_day0_cut.values()))
    D_CCAdata_day1=np.concatenate(list(latents_day1_cut.values()))

    CCAdata_day0 = np.reshape(D_CCAdata_day0, (-1, n_components))
    CCAdata_day1 = np.reshape(D_CCAdata_day1, (-1, n_components))

    #CREATE DICTIONARIES:
    full_cut_data = {    
    'Day0': {
        'Left': latents_day0_cut['Left'],
        'Straight': latents_day0_cut['Straight'],
        'Right': latents_day0_cut['Right']
    },
    'Day1': {
        'Left': latents_day1_cut['Left'],
        'Straight': latents_day1_cut['Straight'],
        'Right': latents_day1_cut['Right']
    }
}

    D_CCAdata_days= {'Day0': D_CCAdata_day0, 'Day1': D_CCAdata_day1}
    CCAdata_days= {'Day0': CCAdata_day0, 'Day1': CCAdata_day1}


    fig = plt.figure(figsize=(11,6))
    gs2 = add_gridspec_abs(fig, nrows=2, ncols=5, 
                            left=0.5, bottom=1,
                            width=9, height=4)  
    monkey_date0 = os.path.basename(neural_file['Day0']).split('_')
    monkey_date0= monkey_date0[1]+'_'+monkey_date0[2]
    monkey_datek = os.path.basename(neural_file['Day1']).split('_')
    monkey_datek= monkey_datek[1]+'_'+monkey_datek[2]


    fig1, data_aligned= plot_fig_CCA(CCAdata_days, gs2, full_cut_data, monkey_date0, monkey_datek)
    axes3= plot_dims_CCA(CCAdata_days, min_time, full_cut_data, monkey_date0, monkey_datek)
    axes4= plot_graph_CCA(D_CCAdata_days, CCAdata_days, data_aligned, n_components, monkey_date0, monkey_datek)

    plt.show()







