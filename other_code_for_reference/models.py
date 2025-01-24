
import logging
import os
import pickle
from re import L
import shutil
import subprocess
from copy import copy
from numba.core import config

import numpy as np
from numpy.core.fromnumeric import put
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import bezier
import logging
from scipy.spatial.distance import euclidean
from navbcidecode.bcidecode.preprocessing.kinematics import diff
from navbcidecode.bcidecode.preprocessing.ratesTransformer import RatesTransformer
from navbcidecode.bcidecode.preprocessing.data import get_alignOn
from navbcidecode.bcidecode.preprocessing.kinematics import compute_modified_velocities, compute_modified_positions
from .utils import getCountVectors, _filterSingleTrial, fileCopy

def compute_positions(estimated_velocities, initial_position, bin_size):
    delta = bin_size / 1000
    estimated_pos = [initial_position]
    for index, vel in enumerate(estimated_velocities):
        new_pos = estimated_pos[index] + delta * vel
        estimated_pos.append(new_pos)
    estimated_pos = np.vstack(estimated_pos)
    return estimated_pos

# def getRobotDir(baseDir, robotRoot):
#     ## Online path of type c:/nx3k/debugdata\fix_centerout_avatar_Loki_20210825_1753_N\fix_centerout_avatar_Loki_20210825_1753_N_online_trials.pkl.tmp
#     basename = os.path.dirname(baseDir)
#     basename, exp = os.path.split(basename)
#     # exp = exp.split("_trials")[0]
#     expDate = exp.split("_")[-3]
#     # expDate = os.path.basename(basename)
#     return os.path.join(robotRoot, expDate, exp)

class _BaseModel:
  answersForTraining = [1, 3 ,5, 6]

  def __init__(self, baseDir, config):
    self.trainingProcess = None
    self.baseDir = baseDir
    self.config = config
    self.model = self.LoadData("model.pkl")
    self.binParams = self.LoadData("bin_params.pkl")
    # if config.withLfp == False:
    self.binner = RatesTransformer(**self.binParams)
    # else:
    #   self.binner = LFPTransformer(**self.bin_params)
    # self.nTargets = config.modelNTargets
    self.axes = config.modelAxes
    self.smoothing = config.modelSmoothing
    # self.distances = config.modelDistances
    self.retrainPeriod = config.modelRetrainPeriod
    # self.centerCue = np.array(config.modelCenterCue)
    # self.trialPredictions = np.array(config.modelCenterCue)
    self.covScaling = config.modelCovScaling
    # self.task = self.config.modelTaskType

    self.SaveConfig()

  def SaveConfig(self):
      configFile = os.path.join(self.baseDir, "config.pkl")
      self.SaveData(self.config, configFile)
    
  def LoadData(self, dataFile):
    try:
      with open(os.path.join(self.baseDir, dataFile), "rb") as f:
        data = pickle.load(f)
      return data
    except:
      return []

  def SaveData(self, data, dataFile):
    with open(os.path.join(self.baseDir, dataFile), "wb") as f:
      pickle.dump(data, f)

  def DataConfiguration(self):
    return self.binner.get_default_params()._asdict()

  def _formatOutput(self, predictions):
    # TO DO: Filter out velocities ??
    return {key: value for key, value in zip(self.axes, predictions.T)}

  
  def Train(self, dataFile, numTrials=500):
    if self.trainingProcess:
      raise RuntimeError(
          'Cannot start training process when another training is still running'
      )

    # Copy online trials
    dataFileCopy = dataFile + '.tmp'
    shutil.copy(src=dataFile, dst=dataFileCopy)
    
    # # Copy logs from avatar task
    # robotRoot = r"\\192.168.1.13\robot_vgrasp"
    # if self.config.modelTaskType == "avatar":
    #   robotDir = getRobotDir(dataFileCopy, robotRoot)
    #   toCopy = ["_continuous.dat", "_events.dat"]
    #   for file in toCopy:
    #     # From robot_dir/file.dat --> base_dir/file.dat.tmp
    #     srcFile = os.path.join(robotDir, file)
    #     dstFile = os.path.join(self.baseDir, f"{file}.tmp")
    #     shutil.copy(src=srcFile, dst=dstFile)

    logging.info("Starting Retraining")
    logging.info(f"ONLINE PATH {dataFileCopy}")
    self.trainingProcess = subprocess.Popen(
        ['python', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'navigation_retraining.py'),
         '--onlinedata', dataFileCopy, '--datadirectory', self.baseDir,
         "--numtrials", str(numTrials), "--retrainperiod", str(self.retrainPeriod),
         "--logfile", os.path.join(self.baseDir, f"training.log")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


class DummyModel(_BaseModel):
  def __init__(self, baseDir, config):
    pass


class Model(_BaseModel):
  def TrialInit(self, trial):
    self.centerCue = np.array([0.0, 0.0, 0.0])
    self.trialPredictions = self.centerCue
    self.numPredictions = 0 
    
  def _Reset(self):
    # Reset Model
    try:
      self.model["decoder"].reset()
        # self.model.reset()

    except:
      self.model.reset()
    # Reset trial predictions
    self.trialPredictions = self.centerCue

  def Reset(self):
    if self.trainingProcess:
      try:
        self.trainingProcess.wait(timeout=0)
        # Process.communicate()
        if self.trainingProcess.returncode != 0:
          stdout, stderr = self.trainingProcess.communicate()
          # raise RuntimeError(f"Training process error {str(stderr)}")
          ## Skip retrain, keep error
          logging.debug(f"Training process error {str(stderr)}")
        self.trainingProcess = None
        logging.info("Retraining Finished")
        self.model = self.LoadData("model.pkl")

      except subprocess.TimeoutExpired:
        # Training is still running
        logging.info(f"Training process still running")
        pass

    # Reset Model
    self._Reset()
    self.numPredictions = 0
    return self

  def unbiasTrials(self, onlineTrials):
    # Load previous trial metadata
    self.bufferTrials = self.LoadData("bufferTrials.pkl")
    currentTrials = self.LoadData("filtered_trials.pkl")
    # Compute count vectors
    currentCounts, _ = getCountVectors(currentTrials, self.nTargets)
    bufferCounts, bufferLabels = getCountVectors(self.bufferTrials, self.nTargets)
    onlineCounts, onlineLabels = getCountVectors(onlineTrials, self.nTargets)
    # Compute next current count vector
    totalCounts = onlineCounts + currentCounts + bufferCounts
    nextCounts = np.array([np.min(totalCounts)] * self.nTargets)
    deltaCounts = nextCounts - currentCounts
    deltaCounts[deltaCounts < 0] = 0
    # Update counts & trials (online)
    takeCounts = np.minimum(deltaCounts, onlineCounts)
    toBuffer = onlineCounts - takeCounts
    deltaCounts = deltaCounts - takeCounts
    keepOnline = np.concatenate([
        onlineTrials[np.argwhere(onlineLabels == target)[:count, 0]]
        for target, count in zip(range(self.nTargets), takeCounts)
    ])
    bufferTrials = np.concatenate([
        onlineTrials[np.argwhere(onlineLabels == target)[count:, 0]]
        for target, count in zip(range(self.nTargets), takeCounts)
    ])

    # Update counts & trials (buffer)
    takeCounts = np.minimum(deltaCounts, bufferCounts)
    bufferCounts = bufferCounts - takeCounts + toBuffer
    if bufferLabels.size > 0 & sum(deltaCounts) > 0:
      keepBuffer = np.concatenate([
          self.bufferTrials[np.argwhere(bufferLabels == target)[:count, 0]]
          for target, count in zip(range(self.nTargets), takeCounts)
      ])
      self.bufferTrials = np.concatenate([
          self.bufferTrials[np.argwhere(bufferLabels == target)[count:, 0]]
          for target, count in zip(range(self.nTargets), takeCounts)
      ])
      # Update meta for next training iteration
      self.debug(keepOnline, keepBuffer)
      unbiasedTrials = np.concatenate(keepOnline, keepBuffer)
      self.bufferTrials = np.concatenate(self.bufferTrials, bufferTrials)
    else:
      unbiasedTrials = keepOnline
      self.bufferTrials = bufferTrials
    self.SaveData(self.bufferTrials, "bufferTrials.pkl")
    return list(unbiasedTrials)

  def computeStates(self, onlineTrials, onlineObservations):
    # Compute raw kinematics using previous model
    nan_index = np.argmax(np.isnan(onlineObservations), axis=1)

    # Find the minimum index along the third axis (ignoring NaN values)
    min_index = np.nanmin(nan_index, axis=0)

    # Reshape the matrix to stop before the first NaN value along the third axis
    onlineObservations_without_nan = onlineObservations[:, :min_index.min()]

    predictedStates = [
        self.model.predict(data, real_time=True)
        for data in np.transpose(onlineObservations_without_nan)
    ]
    return predictedStates

  def modifyStates(self, onlineStates, onlineTrials, positionsIds, velocitiesIds):
    # Compute positions
    # start_positions = [[trial.avatarTrajectory["x"][0],trial.avatarTrajectory["y"][0],trial.avatarTrajectory["z"][0]] for trial in onlineTrials]
    start_position = [0,0,0]
    # predicted_positions = [compute_positions(
    #                 state, start_position, 50.0) for state,start_position in zip(onlineStates,start_positions)]
    predicted_positions = [compute_positions(state, start_position, 50.0) for state in onlineStates]
    # predicted_positions = [
    #     _filterSingleTrial(states,np.array([trial.unityTargetPosition[0][ax] for ax in self.axes]),positionsIds,r_max = self.config.modelRmax) 
    #     for states, trial in zip(predicted_positions,onlineTrials)
    # ]
    predicted_positions = [
        _filterSingleTrial(states,trial.targetPosition,positionsIds,r_max = self.config.modelRmax) 
        for states, trial in zip(predicted_positions,onlineTrials)
    ]
    onlinePositions = [predicted_positions[idx][1:] for idx in range(len(predicted_positions))]
    ## ReFit
    onlineVelocitiesMod = [
        compute_modified_velocities(
            trial, states, self.binner.event, self.binner.bin_size, positionsIds, velocitiesIds, self.axes
        )
        for trial, states in zip(onlineTrials, onlinePositions)
    ]
    onlineStatesMod = [
        np.vstack([(positions[:,idx], velocities[:,idx]) for idx in range(len(positionsIds))]).T
        for positions, velocities in zip(onlinePositions, onlineVelocitiesMod)
    ]
    return onlineStatesMod

  # def SmoothPredictions(self, predictions):
  #   # Append current predictions
  #   self.trialPredictions.append(predictions)
  #   trialPredictions = np.concatenate(copy(self.trialPredictions))

  #   # Apply filter
  #   predictions = np.empty((1, len(self.axes)))
  #   # Return last filtered prediction
  #   for idx in range(len(self.axes)):
  #     filteredPreds = gaussian_filter1d(trialPredictions[:, idx], 6)
  #     predictions[0, idx] = filteredPreds[-1]

  #   return predictions

  def Predict(self, trial, lfp, withSpikes, trial_index=False, bin_index=False, current_position=None, CCA={'Offline': False}, Kalman_latents=False):
    try:
      if self.numPredictions >= 4:
        if np.any(lfp) == True and withSpikes == True:
          trial_and_lfp = np.concatenate((lfp,trial))
          predictions = self.model["decoder"].predict(trial_and_lfp, trial_index, bin_index, current_position, real_time=True, CCA=CCA, Kalman_latents=Kalman_latents)
        elif np.any(lfp) == False:
          predictions = self.model["decoder"].predict(trial, trial_index, bin_index, current_position, real_time=True, CCA=CCA, Kalman_latents=Kalman_latents)
        else:
          predictions = self.model["decoder"].predict(lfp, trial_index, bin_index, current_position, real_time=True, CCA=CCA, Kalman_latents=Kalman_latents)
        
        # Update positions
        self.trialPredictions = predictions
        # Smoothing
        if self.smoothing:
          predictions = self.SmoothPredictions(predictions)
      else:
        predictions = np.array((0,0,0))
      self.numPredictions +=1
      return self._formatOutput(predictions)

    except:
      logging.exception('Exception while predicting')
      return -1
    
  def get_unaligned_latents(self):
    return self.model["decoder"].get_unaligned_latents()
  
  def get_aligned_latents(self):
    return self.model["decoder"].get_aligned_latents()
  
  def get_Kalman_coefficients(self):
    return self.model["decoder"].get_Kalman_coefficients()

