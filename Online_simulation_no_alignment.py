## Simulate online decoding to test several things: automatic selection of channels, selection of LFP channels, neuron-dropping method, ...
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from tnsbmi.onlinedecoding import localconfig, tasks, util
from tnsbmi.tnsbmi import bintrials, dataconversion
from tnsbmi.tnsbmi import bintrials, nevdata
from tnsbmi.tnsbmi import modeling
import os
from os.path import dirname
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
from bcidecode.preprocessing.ratesTransformer import (EpochTransformer,
                                                      RatesTransformer)
from bcidecode.online.models import Model
from bcidecode.tools import ccaTools as cca


def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path, label_name="targetObject", take_answers=[1, 3, 5]):
  # Load data
  trials, taskparameters, _ = dataconversion.LoadLastNTrials([data_path], 10000,answerNumbers=take_answers) 
  return trials, taskparameters

def plot_square(ax, center, width_length, color, label=None ,style = None):
    half_width_length = width_length / 2
    vertices = np.array([
        [center[0] - half_width_length, center[1] - half_width_length],
        [center[0] - half_width_length, center[1] + half_width_length],
        [center[0] + half_width_length, center[1] + half_width_length],
        [center[0] + half_width_length, center[1] - half_width_length],
        [center[0] - half_width_length, center[1] - half_width_length]  # To close the square
    ])
    ax.fill(vertices[:, 0], vertices[:, 1], 'white', zorder=2) 
    if style == None:
        ax.plot(vertices[:, 0], vertices[:, 1], color, zorder=3) #label =str(label)
    else:
        ax.plot(vertices[:, 0], vertices[:, 1], style, color, zorder=3)

def update_configuration(target_config, source_config):
    # Get all attributes from source_config
    for attr in dir(source_config):
        # Skip special and internal attributes
        if not attr.startswith('__') and not callable(getattr(source_config, attr)):
            # Set the attribute on the target_config
            setattr(target_config, attr, getattr(source_config, attr))

def is_within_target_window(states, target, experimentType, window_size = 5.6, min_consecutive = 10): 

    """
    Check if the point (x, y) is within the target window around (x_target, y_target).
    
    Parameters:
    - x, y: Coordinates of the point to check.
    - x_target, y_target: Coordinates of the target point.
    - window_size: Size of the target window (side length of the square).

    Returns:
    - True if (x, y) is within the target window, False otherwise.
    """
    if experimentType == "fixedCamera":
        window_size = 4.2 #measured in unity
    elif experimentType == "movingCamera":
        window_size = 5.6  #target window radius = 2.5 (measured in unity)
    else:
        print("Incorrect experimentType.")
    half_window_size = window_size / 2.0

    consecutive_count = 0

    for x, y in states:
        if (
            (target[0] - half_window_size) <= x <= (target[0] + half_window_size) and (target[1] - half_window_size) <= y <= (target[1] + half_window_size)
        ):
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                return True
        else:
            consecutive_count = 0

    return False

def compute_success_rate(trials, allPredictions, task, execution):
    success_count = 0
    total_trials = len(trials)
    dt = 0.05
    for trial_index in range(total_trials):
        trial = trials[trial_index]
        predictions = allPredictions[trial_index]
        
        if execution == "offline":
            # Compute positions from predictions
            x_positions_pred = [0.0]
            y_positions_pred = [0.0]
            z_positions_pred = [0.0]
            
            for j in range(1, len(predictions)):
                prev_x, prev_y, prev_z = x_positions_pred[-1], y_positions_pred[-1], z_positions_pred[-1]
                current_x_velocity = float(predictions[j]['x'])
                current_y_velocity = float(predictions[j]['y'])
                current_z_velocity = float(predictions[j]['z'])
                new_x_position = prev_x + current_x_velocity * dt
                new_y_position = prev_y + current_y_velocity * dt
                new_z_position = prev_z + current_z_velocity * dt
                x_positions_pred.append(new_x_position)
                y_positions_pred.append(new_y_position)
                z_positions_pred.append(new_z_position)
        else:
            x_positions_pred = predictions['x'] 
            z_positions_pred = predictions['z'] 
        # Check if computed trajectory is within target window
        target = [trial.targetPosition[0], trial.targetPosition[2]]
        is_success = is_within_target_window(
            list(zip(x_positions_pred, z_positions_pred)), 
            target, 
            task,
        )
        
        if is_success:
            success_count += 1

    return success_count / total_trials

def prepare_model(neural_file0, trials):
 
    # Prepare model:
    common_path=os.path.dirname(neural_file0)

    preloaded_config_path= os.path.join(common_path, "bin_params.pkl")
    extra_config_path= os.path.join(common_path, "config.pkl")
    output_directory= common_path

    with open(preloaded_config_path, 'rb') as config_file:
        preloaded_config = pickle.load(config_file)
    with open(extra_config_path, 'rb') as config_file:
        extra_config = pickle.load(config_file)
    configuration = localconfig.Configuration(**preloaded_config)
    configuration.modelDirectory= os.path.join(common_path, "model.pkl") 
    update_configuration(configuration, extra_config)

    # Model
    model = Model(output_directory, configuration)
    configuration.channels = model.DataConfiguration()['channels']
    configuration.binWidth = model.DataConfiguration()['binWidth']
    configuration.withLfp = model.DataConfiguration()['withLfp']
    configuration.combined = model.DataConfiguration()['combined']
    configuration.lfpLength = model.DataConfiguration()['lfpLength']
    configuration.frequencyBand = model.DataConfiguration()['frequencyBand']
    configuration.samplingRate = model.DataConfiguration()['samplingRate']
    configuration.withSpikes = model.DataConfiguration()['withSpikes']

    ## Bin online decoding data
    pipeline_params = {
            "preprocessing__rates_comp__t_int": 8000,
            "preprocessing__rates_comp__t_skip": 0,
            "preprocessing__rates_comp__event": "GoCue",
            "preprocessing__rates_comp__stop_flag": "stop",
            "preprocessing__rates_comp__channels": configuration.channels,
            "preprocessing__rates_comp__channels_lfp": configuration.channels_lfp,
            "preprocessing__rates_comp__bin_size": configuration.binWidth,
            "preprocessing__rates_comp__withLfp": configuration.withLfp,
            "preprocessing__rates_comp__lfpLength": configuration.lfpLength,
            "preprocessing__rates_comp__combined": configuration.combined,
            "preprocessing__rates_comp__samplingRate": configuration.samplingRate,
            "preprocessing__rates_comp__frequencyBand": configuration.frequencyBand,
        }
    bin_params = {
        key.split("__")[-1]: value
        for key, value in pipeline_params.items()
        if "rates_comp" in key
    }
    binner = RatesTransformer(**bin_params).fit(trials0,task)
    #Day-0:
    timeStamps_spikes, timeStamps_lfps = binner.get_binned_data(trials, configuration.channels_lfp)

    return model, configuration, timeStamps_spikes, timeStamps_lfps

## Load online decoding data:
neural_file_list=[
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\navtrainingsphere_Maui_20241121_0953_A_trials.pkl",
#5
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\navtrainingsphere_Vino_20240827_0926_A_trials.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\Maui_20240215\navtrainingsphere_Maui_20240215_1210_A_trials.pkl",
#7
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Loki_20230825_0945_B\navtraining_Loki_20230825_0945_B_trials_training_model.pkl",
r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Loki_20230830_1059_B\navtrainingsphere_Loki_20230830_1059_B_trials_training_model.pkl",
]
#08: 187 channels
#12: 182 channels
#13: 177 channels
#18: 189 channels
#19: 193 channels
#21: 187 channels

neural_file0 = neural_file_list[4] 
# neural_filek = neural_file_list[5] 

#Where to save
dest_dir= dirname(neural_file0)

#Load Day-0 data
trials0, taskparameters = load_data(neural_file0)
task = taskparameters["task"]

#Load Day-0 model
model0, configuration0, timeStamps_spikes0, timeStamps_lfps0= prepare_model(neural_file0, trials0)


Kalman_latents=False 
CCA={'Offline': False}
model=model0

configuration=configuration0
trials=trials0
timeStamps_spikes=timeStamps_spikes0
timeStamps_lfps=timeStamps_lfps0


## Predictions: For loop over trials and over bins
allPredictions = []
all_unaligned_latents=[]
all_aligned_latents=[]

for trial_index in range(0,len(trials)):
    model.TrialInit(trials[trial_index])
    trialPredictions = []  
    trial_unaligned_latents=[]
    trial_aligned_latents=[]

    binWidth = 50
    timeStamps_spikes_trial = timeStamps_spikes[trial_index]
    spikeHistograms_t = [bintrials.SpikeRates_offline(evt, binWidth, configuration.channels) for evt in timeStamps_spikes_trial]
    spikeHistograms = np.array(np.transpose([np.array(list(histogram.values()), dtype=float) for histogram in spikeHistograms_t]))
    timeStamps_lfps_trial = timeStamps_lfps[trial_index]
    lfpPower = [bintrials.LfpPowers_offline(evt, configuration.channels_lfp, configuration.frequencyBand, configuration.samplingRate) for evt in timeStamps_lfps_trial]
    lfpPower = np.array(np.transpose([np.array(list(power.values()), dtype=float) for power in lfpPower]))
    # Compute velocities (for each bin)
    for bin_index in range(0,len(lfpPower[1])):
            spikeHistogram_0 = np.array([histogram[bin_index + 4] for histogram in spikeHistograms])
            lfpFeatures = np.array([lfp[bin_index] for lfp in lfpPower])
            predictions = model.Predict(np.array(spikeHistogram_0), np.array(lfpFeatures), configuration.withSpikes,  trial_index=trial_index, bin_index=bin_index, CCA=CCA, Kalman_latents=Kalman_latents)
            
            unaligned_latents=model.get_unaligned_latents()
            aligned_latents=model.get_aligned_latents()

            trialPredictions.append(predictions)
            trial_unaligned_latents.append(unaligned_latents)
            trial_aligned_latents.append(aligned_latents)

    allPredictions.append(trialPredictions)   
    all_unaligned_latents.append(trial_unaligned_latents)
    all_aligned_latents.append(trial_aligned_latents)
    model.Reset() 


dest_dir = r"C:\GBW_MyDownloads\Code\0N_Data\Models"
Path(dest_dir).mkdir(parents=True, exist_ok=True)
with open(os.path.join(dest_dir, "latents_calculated_with_kalman_filter.pkl"), "wb") as f:
    pickle.dump(all_unaligned_latents, f)     


# Compute success rate
success_rate_offline = compute_success_rate(trials, allPredictions, task, execution = "offline")
onlinePositions = np.array([trial.avatarTrajectory for trial in trials])
success_rate_online = compute_success_rate(trials, onlinePositions, task, execution = "online")
print(f'Online success rate: ' + str(success_rate_online) + ' , Offline success rate: ' + str(success_rate_offline))

# Plot trajectories
# Sample trial indexes
# trial_indexes = [0, 8, 15, 20, 35, 50, 65, 70, 90, 95]
trial_indexes = [0, 8, 15, 20, 35, 45, 50, 65, 70, 80]
dt = 0.05 # 50ms = 0.05s

fig, axs = plt.subplots(2, 5, figsize=(20, 10))
axs = axs.flatten()
for i, trial_idx in enumerate(trial_indexes):
    # Retrieve the specific trial
    trial = trials[trial_idx]
    predictions = allPredictions[trial_idx]
    
    # Initialize position arrays
    x_positions = [0.0]
    y_positions = [0.0]
    z_positions = [0.0]

    # Iterate over the predictions and integrate velocities to get positions
    for j in range(1, len(predictions)):
        prev_x, prev_y, prev_z = x_positions[-1], y_positions[-1], z_positions[-1]
        current_x_velocity = float(predictions[j]['x'])
        current_y_velocity = float(predictions[j]['y'])
        current_z_velocity = float(predictions[j]['z'])
        # Compute the new positions based on velocity and time
        new_x_position = prev_x + current_x_velocity * dt
        new_y_position = prev_y + current_y_velocity * dt
        new_z_position = prev_z + current_z_velocity * dt
        # Append the new positions
        x_positions.append(new_x_position)
        y_positions.append(new_y_position)
        z_positions.append(new_z_position)
    
     # Initialize position arrays
    x_positions_original = [0.0]
    y_positions_original = [0.0]
    z_positions_original = [0.0]

    # Iterate over the predictions and integrate velocities to get positions
    for j in range(1, len(predictions)):
        prev_x, prev_y, prev_z = x_positions_original[-1], y_positions_original[-1], z_positions_original[-1]
        current_x_velocity = float(trial.avatarVelocity['vx'][j])
        current_y_velocity = float(trial.avatarVelocity['vy'][j])
        current_z_velocity = float(trial.avatarVelocity['vz'][j])
        # Compute the new positions based on velocity and time
        new_x_position = prev_x + current_x_velocity * dt
        new_y_position = prev_y + current_y_velocity * dt
        new_z_position = prev_z + current_z_velocity * dt
        # Append the new positions
        x_positions_original.append(new_x_position)
        y_positions_original.append(new_y_position)
        z_positions_original.append(new_z_position)

    # Create time series based on bin indices
    time_series = np.arange(len(predictions)) * dt  # Time in seconds

    # Plot the square at the target position
    plot_square(axs[i], np.array([trial.targetPosition[0], trial.targetPosition[2]]), width_length=4.2, color='r', label='Target') #4.2/5.6

    # Plot the trajectory
    axs[i].plot(x_positions, z_positions, marker='o', linestyle='-', color='b', label='Trajectory')
    axs[i].plot(trial.avatarTrajectory['x'], trial.avatarTrajectory['z'], marker='o', linestyle='-', label='Original trajectory')
    # axs[i].plot(x_positions_original, z_positions_original, marker='o', linestyle='-', label='Original trajectory vel')
    # Customize plot
    axs[i].set_title(f'Trial {trial_idx}')
    axs[i].set_xlabel('x position')
    axs[i].set_ylabel('z position')
    axs[i].legend()
    axs[i].set_aspect('equal')
    axs[i].grid(True)

date=neural_file0.split("navtrainingsphere_")[-1].split(".pkl")[0].split('_')[1]
fig.suptitle('Test Day-0 model using Day-0 data (no alignment) - File: '+date)
# Adjust layout
plt.tight_layout()
plt.show()
print("Done")
