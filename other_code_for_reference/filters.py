from bcidecode.preprocessing.reachTuningTransformer import KinematicsFilter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin, MultiOutputMixin
from .system_fit import fit_steady_gain, mle_fit
import PSID
from bcidecode.preprocessing.kinematics import compute_positions
from pathlib import Path
import pickle
import os
import scipy.linalg as linalg


class LinearKalman:
    """Kalman filter with steady state gain 
       for a discrete time linear system with the following dynamics:
              x_{k+1} = A*x_k + b + w_n
              z_k = C*x_k + d + v_n 
    """

    def __init__(self,
                 transition_mat,
                 measurement_mat,
                 kalman_gain,
                 initial_state,
                 transition_off=None,
                 measurement_off=None
                 ):
        self.A = transition_mat
        self.C = measurement_mat
        self.K = kalman_gain
        self.x0 = initial_state
        self.x = initial_state
        self.set_b(transition_off)
        self.set_d(measurement_off)

    def set_b(self, transition_off):
        if transition_off == None:
            self.b = np.zeros((self.A.shape[0]))

    def set_d(self, measurement_off):
        if measurement_off == None:
            self.d = np.zeros(self.C.shape[0])

    # @jit
    def predict(self):
        return self.A @ self.x + self.b

    # @jit
    def update(self, x, z):
        y = z - (self.C @ x + self.d)   # residual
        x = x + self.K @ y  # posterior
        return x

    # @jit
    def filter(self, z):
        x_tilde = self.predict()
        x_hat = self.update(x_tilde, z)
        # Update filtered state
        self.x = x_hat
        return x_hat

class KalmanRegressor(MultiOutputMixin, RegressorMixin):
    """Kalman filter with steady state gain 
       for a discrete time linear system with the following dynamics:
              x_{k+1} = A*x_k + b + w_n
              z_k = C*x_k + d + v_n 
    """

    def fit(self, latents, kinematics):
        # Fit Kalman system
        states_mat = np.concatenate(kinematics)
        observations_mat = np.concatenate(latents)
        # Create system of linear equations
        # State transition equation
        x_states = states_mat[:-1]
        y_states = states_mat[1:]
        # Measurement equation
        x_measurement = states_mat
        y_measurement = observations_mat
        # Solve system equations
        A, Q, C, R, x0, m0 = mle_fit(
            kinematics, x_states, y_states, x_measurement, y_measurement)
        # Compute steady state kalman gain
        K = fit_steady_gain(A, Q, C, R, m0)
        self.kf_ = LinearKalman(A, C, K, x0)
        return self

    def _system_update(self, observations, step, x_hat):
        obs = observations[step]
        x_hat = self.kf_.filter(x_hat, obs)
        return x_hat

    def predict(self, observations):
        predictions = []
        # velocity_ids = [1, 3]
        for step in range(len(observations)):
            obs = observations[step]
            x_hat = self.kf_.filter(obs)
            # predictions.append(x_hat[velocity_ids])
            predictions.append(x_hat)
        return np.asarray(predictions)

class PSID_Decoder(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(
        self, 
        nx=20,
        n1=10,
        i=10,
        tuning_threshold=0.3,
        preprocessor=None,
        regressor=None,
        start_positions=None,#start_position = [0,0,0]
        start_velocity=[0.0, 0.0, 0.0],
        velocities_ids=[1, 3, 5],
        bin_size = 50,
        axes=["X","Y","Z"], 
        nb_trials = None
    ):
        """
        """
        self.nx = nx
        self.n1 = n1
        self.i = i
        self.timebin=0
        self.current_trial=0
        self.preprocessor = preprocessor
        self.regressor = regressor
        self.preprocessor_ = None
        self.regressor_ = None
        self.start_positions = start_positions
        self.start_velocity = start_velocity
        self.velocities_ids = velocities_ids
        self.position_ids = self._get_position_ids()
        self.bin_size = bin_size
        self.tuning_threshold = tuning_threshold
        self.axes=axes
        self.nb_trials = nb_trials

        self.observations=[]
        self.states=[]
        self.unaligned_latents=[]
        self.aligned_latents=[]

    def _get_position_ids(self): #Get index of position (x, y, z) in states
        all_ids = range(3 + len(self.velocities_ids))
        positions_ids = [id for id in all_ids if not(id in self.velocities_ids)]
        return positions_ids

    def _fit_preprocessor(self, observations, states):
        self.preprocessor_ = self.preprocessor.fit(observations)
        observations = [self.preprocessor_.transform(
            [obs]) for obs in observations] #for each trial -> for each 50ms bin (remove NaN bins)-> spike rate for each electrode
        observations = [obs[:trial_states.shape[0]]
                        for trial_states, obs in zip(states, observations)] #only keep bins that have a corresponding kinematic state (only while trial is running)        
        states = [states[:trial_obs.shape[0]]
                        for trial_obs, states in zip(observations, states)] #only keep bins that have corresponding observation
        
        # Plot snelheden vs neural signal (eg if vx<0.1 and vz >4 -> set in a list and average) or (plot in time from start to end -> average over all right)
        dest_dir=r"C:\GBW_MyDownloads\Code\0N_Data\Models"

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dest_dir, "all_states.pkl"), "wb") as f:
            pickle.dump(states, f)
        with open(os.path.join(dest_dir, "all_observations.pkl"), "wb") as f:
            pickle.dump(observations, f)

        observations_path=r"C:\GBW_MyDownloads\Code\0N_Data\Models\all_observations.pkl"
        states_path=r"C:\GBW_MyDownloads\Code\0N_Data\Models\all_states.pkl"

        # with open(observations_path, 'rb') as f:
        #     observations1 = pickle.load(f)
        # with open(states_path, 'rb') as f:
        #     states1 = pickle.load(f)
        # Kinematics filter 
        #observations_conc = np.concatenate(observations) #set all trials one after the other in a long array
        # self.kinematics_filter = KinematicsFilter(threshold=self.tuning_threshold).fit(observations_conc, np.concatenate(states)) # search for most selective channels
        # observations = [self.kinematics_filter.transform(obs) for obs in observations] # for each trial only keep observations of selected channels
        
        
        
        return observations, states

    def _fit_latents_model(self, observations, states):
        self.model_ = PSID.PSID(
            observations,
            states,
            nx=self.nx,
            n1=self.n1,
            i=self.i
        )
        
    def _fit_regressor(self, observations, states):
        latents_trials = [self.predict_latents(obs, real_time=False) for obs in observations]
        latents = np.concatenate([self.predict_latents(obs, real_time=False) for obs in observations])
        states = np.concatenate([st[:, self.velocities_ids] for st in states]) # only keep velocity states (vx,vy,vz)
        self.regressor_ = self.regressor.fit(latents, states)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.kernel_approximation import Nystroem
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        import pandas as pd
        import matplotlib.pyplot as plt
        nystroem = Nystroem(kernel='rbf', gamma=0.34, n_components=700, random_state=42)
        latents_transformed = nystroem.fit_transform(latents)
        # Assuming 'states_data' is your list of arrays with shape (90, Ax6)
        transposed_data = np.transpose(latents_transformed)  # Transpose to shape (Ax90x6)
        # transposed_data = [transposed_data[0], transposed_data[2], transposed_data[4], transposed_data[1], transposed_data[3], transposed_data[5]]
        # Concatenate along the second axis (trials) to get a list of 6 arrays
        # concatenated_trials = [np.concatenate(trial, axis=0) for trial in transposed_data]
        transposed_data = transposed_data[0:20,:500]
        # Initialize offset
        offset = 0  

        # Transposed data
        transposed_data = np.random.rand(10, 20)

        # Plot transposed data
        # plt.figure(figsize=(8, 6))  
        # for i, data in enumerate(transposed_data):      
        #     plt.plot(data + i * offset * 10, color='#C4A69D')  # Increase scaling factor
        #     offset += max(data.max(), 0.1)  
        # plt.xticks([])  
        # plt.yticks([])  
        # plt.show()

        # Number of points for interpolation
        num_points = 100

        # Number of random lines
        num_lines = 20

        # Generate random lines
        plt.figure(figsize=(8, 6))  
        for i in range(num_lines):
            # Generate random x values
            x = np.linspace(0, 10, num_points)
            
            # Generate random y values with different random noise for each line
            y = np.sin(x) + np.random.normal(scale=0.5, size=num_points) + i * 0.5
            
            # Interpolate to create a smooth curve
            t = np.linspace(0, 10, 10*num_points)
            interpolated_y = np.interp(t, x, y)
            
            # Plot the interpolated curve with a larger vertical offset
            plt.plot(t, interpolated_y + i * 10, color='#98A886')  # Increase scaling factor

        # Show the plot
        # plt.show()
        # # latents_trans =pd.DataFrame(latents_transformed)
        # # X = add_constant(latents_trans)
        # # X = X[:300]
        # # vif_data = pd.DataFrame()
        # # vif_data["Variable"] = X.columns
        # # vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        # # print(vif_data)
        # # Assuming X and y are your feature matrix and target variable
        # X_train, X_test, y_train, y_test = train_test_split(latents_transformed, states, test_size=0.2, random_state=42)

        # # Create and fit the Ridge regression model
        # ridge_model = Ridge(alpha=0.1)  # You can adjust the alpha parameter
        # ridge_model.fit(X_train, y_train)

        # # Make predictions on the test set
        # y_pred = ridge_model.predict(X_test)

        # # Evaluate the model
        # mse = mean_squared_error(y_test, y_pred)
        # print(f"Mean Squared Error: {mse}")
        dest_dir=r"C:\GBW_MyDownloads\Code\0N_Data\Models"
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dest_dir, "latents.pkl"), "wb") as f:
            pickle.dump(latents_trials, f)

    def fit(self, observations, states):
        """
        Fit global model on X features to minimize 
        a given function on Y.

        @param X
        @param Y
        """

        # Fit preprocessor
        if self.preprocessor:
            observations, states = self._fit_preprocessor(observations, states) # remove all NaN values and keep the same number of state and observation bins within each trial

            self.observations=observations
            self.states=states
            
            neural_file_list=[
            r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241108_0950_A\navtrainingsphere_Maui_20241108_0950_A_trials.pkl",
            r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241112_0950_A\navtrainingsphere_Maui_20241112_0950_A_trials.pkl",
            r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241113_0948_A\navtrainingsphere_Maui_20241113_0948_A_trials.pkl",
            r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241118_1002_A\navtrainingsphere_Maui_20241118_1002_A_trials.pkl",
            r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A\navtrainingsphere_Maui_20241121_0953_A_trials.pkl",
            # r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Vino_20240827_0926_A\navtrainingsphere_Vino_20240827_0926_A_trials.pkl"
            ]
    
            neural_file = neural_file_list[4]
            dest_dir=os.path.dirname(neural_file)

  
            # # # dest_dir = r"C:\GBW_MyDownloads\Code\0N_Data\navigation\navtrainingsphere_Maui_20241121_0953_A"
            # Path(dest_dir).mkdir(parents=True, exist_ok=True)
            # with open(os.path.join(dest_dir, f"observations_belo.pkl"), "wb") as f:
            #     pickle.dump(observations, f)      
            #     f.flush()
            # with open(os.path.join(dest_dir, f"states_belo.pkl"), "wb") as f:
            #     pickle.dump(states, f) 
            #     f.flush()     
            # import joblib
        
            # observations_path= os.path.join(dest_dir, f"observations_belo.pkl")
            # states_path= os.path.join(dest_dir, f"states_belo.pkl")
            # observations=joblib.load(observations_path)
            # states=joblib.load(states_path)

            # Path(dest_dir).mkdir(parents=True, exist_ok=True)
            # with open(observations_path, 'rb') as f:
                # observations = pickle.load(f)


        # Fit latents model
        self._fit_latents_model(observations, states) # compute parameters of the model + parameters of the Kalman filter

        # Fit regressor
        if self.regressor:
            self._fit_regressor(observations, states) # compute the parameters of the regression model
        return self

    def predict_latents(self, observations, real_time, Kalman=False):
        # Extract all latent variables for observations (xk)
        if real_time:
            latent_states = self.model_.kalman_rt(observations.reshape(-1), Kalman=False)  
            # latent_states = self.model_.kalman_rt(np.transpose(observations))      
        else:
            _, _, latent_states = self.model_.predict(observations) #predicted kinematic states, spike rate of selected electrodes, latent states (for each bin)
        # Return just the behaviouraly relevant components(xk[n1])
        latents_ = latent_states[:, :self.n1]
        return latents_

    def transform(self, observations_data, real_time=False):
        if self.preprocessor_:
            observations_data = [self.preprocessor_.transform(
                [obs]) for obs in observations_data]
            # if self.kinematics_filter:
            #  observations_data = [self.kinematics_filter.transform(obs) for obs in observations_data]
        return [self.predict_latents(data, real_time) for data in observations_data]

    def reset(self):
        """Wrapper for reset initial state of kalman filter at each trial"""
        try:
            self.model_._system_init()
        except:
            pass
        return self

class PSID_DecoderPositions(PSID_Decoder):        
    """Decode position of (cursor/avatar) at each timestep"""    
    def predict(self, observations, previous_positions=None, real_time=False):
        if self.preprocessor_:
            observations = self.preprocessor_.transform(
                observations[np.newaxis, ...])
            # if self.kinematics_filter:
            #  observations = self.kinematics_filter.transform(observations)

        if self.regressor_:
            # # TO DO ... put previous velocity for kalman system...
            latents = self.predict_latents(observations, real_time)
            velocities = self.regressor_.predict(latents)
            if real_time:
                delta = self.bin_size / 1000
                positions = self.start_positions + delta * velocities
                # Update current position
                self.start_positions = positions
                pass
            else:
                positions = compute_positions(
                    velocities, self.start_positions, self.bin_size)
            return positions

        else:
            return self.predict_latents(observations, real_time)

latents_all_unaligned = []
latents_all = []
spike_rate_all = []
all_observations=[]

# trial_data=[]
# trial_of_previous_latent=0


class PSID_DecoderVelocities(PSID_Decoder):
    """Decode velocity of (cursor/avatar) at each timestep""" 
    ## Velocity model

    # def __init__(self):
    #     super().__init__()
    #     self.unaligned_latents=[]
    #     self.aligned_latents=[]
        # self.trial_of_previous_latent = 0  # Initialize as an attribute
        # self.latents_all_unaligned = []
        # self.trial_data = []


    def predict(self, observations, trial_index=False, bin_index=False, previous_positions=None, real_time=False, CCA={'Offline': False}, Kalman_latents=False):
        
    
        if self.preprocessor_:
            A=0
            ################
            #USE TO DETERMINE THE KALMAN LATENTS, OTHERWISE KEEP COMMENTED (will give error due to dif channel size)
            # observations = self.preprocessor_.transform(
            #     observations[np.newaxis, ...])
            # all_observations.append(observations)
            ################

            # observations = self.preprocessor_.transform(
            #     observations)
            # if self.kinematics_filter:
            #  observations = self.kinematics_filter.transform(observations)

        if self.regressor_:
            ################
            #USE TO DETERMINE THE KALMAN LATENTS, OTHERWISE KEEP COMMENTED (will give error due to dif channel size)
            # latents = self.predict_latents(observations, real_time) #predict latents from spike rates
            # self.unaligned_latents=latents
            ################


            #Offline latent alignment: uses the latents we aligned offline to predict the velocities
            if CCA['Offline']:   

                dest_dir = r"C:\GBW_MyDownloads\Code\0N_Data\Models"
                Path(dest_dir).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(dest_dir, "latents_sim0_canonical.pkl"), "rb") as f:
                    latents_sim0_canonical = pickle.load(f)

                try:
                    latents=latents_sim0_canonical[trial_index, bin_index, :].reshape(1, 6)
                except:
                    dest_dir = r"C:\GBW_MyDownloads\Code\0N_Data\Models"
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(dest_dir, "CCA_coefficients.pkl"), "rb") as f:
                       CCA = pickle.load(f)

                    U= CCA['U']
                    Vh= CCA['Vh']
                    U2= CCA['U2']
                    Vh2= CCA['Vh2']

                    ################
                    #USE TO DETERMINE THE KALMAN LATENTS, OTHERWISE KEEP COMMENTED (will give error due to dif channel size)
                    # latents_0 = latents @ U2 @ Vh2 @ np.linalg.pinv(U @ Vh) #transform from day-K to day-0
                    # latents=latents_0
                    ################
                    #COMMENT WHEN DETERMINING THE KALMAN LATENTS
                    latents= np.array([0, 0, 0, 0, 0, 0]).reshape(1, 6) 
                    ################



            self.aligned_latents= latents

            velocities = self.regressor_.predict(latents)
            return self.regressor_.predict(latents) #predict velocities from latents

        else:
            return self.predict_latents(observations, real_time)
        
    def get_unaligned_latents(self):
        return self.unaligned_latents #if self.unaligned_latents else []

    def get_aligned_latents(self):

        return self.aligned_latents 
    
    def get_Kalman_coefficients(self):
        Kalman_coef= self.model_.get_Kalman_coefficients()

        return Kalman_coef
