import os
import lab3
from lab3.experiment.group import Mouse, ExperimentGroup
from lab3.experiment.base import BehaviorExperiment, ImagingExperiment, ImagingOnlyExperiment, fetch_trials
from lab3.experiment.virtual import VirtualBehaviorMixin
from lab3.analysis.spatial.abstract_spatial_tuning import SpatialTuning
from lab3.filters.time_filters import IsRunning
from lab3.signal.dfof import Suite2pDFOF
import matplotlib.pyplot as plt
from lab3.misc.progressbar import ProgressBar
import pickle
import numpy as np
import pandas as pd 
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from scipy.signal import fftconvolve, gaussian
from sklearn import svm, metrics, linear_model
from sklearn.model_selection import LeaveOneOut
np.set_printoptions(threshold=np.inf)
# ignore cvxpy warning

df_experiments = pd.read_csv("data.csv")

R_ctx = ['ctxA', 'ctxB']
NR_ctx = ['ctxC', 'ctxD']

def get_variables(bd_exp):
    # Variables from expt
    laps = bd_exp['lap']
    numLaps = bd_exp['json']['laps']

    rw_dur = bd_exp['json']['contexts']['reward']['max_duration']

    if 'water' in bd_exp.keys():
        water = bd_exp['water']

    treadmillPosition = np.array(bd_exp['treadmillPosition'])
    tracklength = bd_exp['json']['track_length']

    if 'licking' in bd_exp.keys():
        licking = bd_exp['licking']
        lickstarts = np.array(bd_exp['licking'])
        numLicks = np.count_nonzero(bd_exp['licking'])
        lickpositions = treadmillPosition[lickstarts]*tracklength
    else:
        licking = []
        lickstarts = []
        numLicks = 0
        lickpositions = treadmillPosition[lickstarts]*tracklength
    velocity = bd_exp['velocity']

    R_lap_list = []
    NR_lap_list = []
    for ctx_id in range(len(bd_exp['json']['contexts'])-1):
        ctx = (list(bd_exp['json']['contexts'].keys())[ctx_id])
        ctx_lap_list = bd_exp['json']['contexts'][ctx]['decorators'][0]['lap_list']
        if ctx in R_ctx:
            R_lap_list.extend(ctx_lap_list)
        else:
            NR_lap_list.extend(ctx_lap_list)
    numRepeat = bd_exp['json']['contexts'][ctx]['decorators'][0]['repeat']

    return laps, numLaps, rw_dur, water, treadmillPosition, tracklength, licking, lickstarts, numLicks, lickpositions, velocity, R_lap_list, NR_lap_list, numRepeat

# calculate Behavioral Summary Table (ICRWL)
def make_ICRWL(bd_exp):
    laps, numLaps, rw_dur, water, treadmillPosition, \
        tracklength, licking, lickstarts, numLicks, lickpositions, \
        velocity, R_lap_list, NR_lap_list, numRepeat = get_variables(bd_exp)
    
    # Extract event timeline
    def collect_event_timeline(bd_exp):
        ctx_list = list(bd_exp['json']['contexts'].keys())

        for i in range(len(bd_exp['json']['contexts'][ctx_list[0]]['decorators'])):
            if 'iti_time' in bd_exp['json']['contexts'][ctx_list[0]]['decorators'][i]:
                iti_time = bd_exp['json']['contexts'][ctx_list[0]]['decorators'][i]['iti_time']

        laps = bd_exp['lap']
        numLaps = bd_exp['json']['laps']

        # find onset/offset during "vr_on" condition 
        vr_on_start = []
        vr_on_start = np.append(vr_on_start,laps[0])
        for i in range(1,len(laps)-1):
            vr_on_start = np.append(vr_on_start, laps[i]+iti_time*10+1)
        vr_on_start = vr_on_start.astype(int)

        vr_on_end = laps[1:]

        # find onset/offset during "vr_off" condition 
        vr_off_start = laps[1:]
        vr_off_end = laps[1:]+iti_time*10

        events = pd.DataFrame({"vr_on_start":vr_on_start, "vr_on_end":vr_on_end, "vr_off_start":vr_off_start, "vr_off_end":vr_off_end })

        # To exclude the last lap when it is incompletely done  
        if events.loc[numLaps-1].max() > bd_exp['lap'][-1]: 
            events.drop(numLaps-1, axis=0, inplace=True)

        return events

    def collect_lick_info(events):
        vr_on_lick = []
        vr_off_lick = []
        for i in range(numLaps-1):
            vr_on_lick.append(lickstarts[events['vr_on_start'][i]:events['vr_on_end'][i]])
            vr_off_lick.append(lickstarts[events['vr_off_start'][i]:events['vr_off_end'][i]])
            
        lick_info = pd.DataFrame({"vr_on":vr_on_lick, "vr_off":vr_off_lick})
        return lick_info

    def collect_treadmill_position(events):
        vr_on_treadmillPosition = []
        vr_off_treadmillPosition = []
        for i in range(numLaps-1):
            vr_on_treadmillPosition.append(treadmillPosition[events['vr_on_start'][i]:events['vr_on_end'][i]])
            
            curr_position = treadmillPosition[events['vr_off_start'][i]:events['vr_off_end'][i]]
            # To find the number of cycles during "vr_off" 
            iti_block = 1
            iti_block_offset = []
            for iti_frame in range(len(curr_position)-1):
                if curr_position[iti_frame]>0.9 and curr_position[iti_frame+1]<0.2:
                    iti_block+=1
                    iti_block_offset.append(iti_frame+1)

            # To reconstruc the position depending on the cycle number during "vr_off" 
            if iti_block == 0:
                vr_off_treadmillPosition.append(curr_position+1)
            else: #in the case of many cyles within ITI period #sometimes 2 cycles during 4s-ITI
                recon_curr_position = []
                for n in range(iti_block):
                    # print(n)
                    if n == len(iti_block_offset): # the last block (inclduing ITI-end)
                        recon_curr_position.extend(curr_position[0:]+(n+1)*1)
                    else:
                        recon_curr_position.extend(curr_position[0:iti_block_offset[n]]+(n+1)*1)
                        # print(recon_curr_position)
                        curr_position = np.delete(curr_position,np.s_[0:iti_block_offset[n]],axis = None)
                        # print('deleted')
                        # print(curr_position)
                vr_off_treadmillPosition.append(np.array(recon_curr_position))
                
        recon_treadmillPosition = pd.DataFrame({"vr_on":vr_on_treadmillPosition, "vr_off":vr_off_treadmillPosition})
        return recon_treadmillPosition

    def collect_lick_pos(tread_position, lick_info):
        vr_on_lick_pos = []
        vr_off_lick_pos = []
        for i in range(numLaps-1):
            curr_vr_on_lick_pos = tread_position['vr_on'][i]*lick_info['vr_on'][i]
            curr_vr_on_lick_pos = curr_vr_on_lick_pos[curr_vr_on_lick_pos!= 0.] # to prevent 0.0 being counted as location '0'
            vr_on_lick_pos.append(curr_vr_on_lick_pos*tracklength) 
            curr_vr_off_lick_pos = tread_position['vr_off'][i]*lick_info['vr_off'][i]
            curr_vr_off_lick_pos = curr_vr_off_lick_pos[curr_vr_off_lick_pos!=0.] # to prevent 0.0 being counted as location '0'
            vr_off_lick_pos.append(curr_vr_off_lick_pos*tracklength)

        lick_pos = pd.DataFrame({"vr_on":vr_on_lick_pos, "vr_off":vr_off_lick_pos})
        return lick_pos

    def collect_water_info(events):
        vr_on_water = []
        vr_off_water = []
        for i in range(numLaps-1):
            vr_on_water.append(water[events['vr_on_start'][i]:events['vr_on_end'][i]])
            vr_off_water.append(water[events['vr_off_start'][i]:events['vr_off_end'][i]])
            
        water_info = pd.DataFrame({"vr_on":vr_on_water, "vr_off":vr_off_water})
        return water_info

    events = collect_event_timeline(bd_exp)
    lick_info = collect_lick_info(events)
    tread_position = collect_treadmill_position(events)
    lick_pos = collect_lick_pos(tread_position, lick_info)
    water_info = collect_water_info(events)

    sequenceUnit = list(range(0,numRepeat))
    stim_seq=sequenceUnit*((numLaps-1)//numRepeat) + sequenceUnit[0:numLaps-len(sequenceUnit*((numLaps-1)//numRepeat))-1]
    
    # Which context is presented?
    ctx_info = []
    for i in range(numLaps-1):
        for ctx in R_ctx + NR_ctx:
            if np.sum(bd_exp[ctx][events['vr_on_start'][i]:events['vr_on_end'][i]]) != 0:
                ctx_info.append(ctx)   
                
    # Is this  reward or no-reward trial?
    r_or_nr = []
    for lap in range(numLaps-1):
        if stim_seq[lap] in R_lap_list:
            r_or_nr.append("True")
        else:
            r_or_nr.append("False")

    # Extract water information
    getWater = [] #whether water reward come or not
    for i in range(numLaps-1):
        if True in water[events['vr_off_start'][i]:events['vr_off_end'][i]]:
            getWater.append("True")
        else:
            getWater.append("False")
            
    # licking behavior during RZ
    lick_or_not = []
    for i in range(numLaps-1):
        if len(lick_pos['vr_off'][i]) == 0:
            lick_or_not.append("False")
        else:
            lick_or_not.append("True")

    # Generate a behavior summary table -- ICRWL: lap_index, context, reward_trial, water, lick_or_not
    ICRWL = pd.DataFrame({"lap_index":stim_seq, "context":ctx_info,"reward_trial":r_or_nr,"water":getWater, "lick_or_not":lick_or_not})
    return ICRWL

# calculate place maps for each neuron -- delta florocence over baseline fluorescence (dF/F0)
def make_place_maps(
    expt, 
    n_position_bins=100, 
    sigma=3,
    velocity_threshold=5
    ):
    
    # Find frames_to_include
    beh_data = expt.format_behavior_data()
    laps = beh_data['lap_bin']
    velocity = beh_data['velocity']
    licking = beh_data['licking']

    ctx_id = beh_data['ctxA'].astype('int')\
    -beh_data['ctxB'].astype('int')\
    -beh_data['ctxC'].astype('int')\
    -beh_data['ctxD'].astype('int')
    iti = ctx_id == 0

    # extend ITI by one frame to correct an error in behavior data formatting
    iti[np.where(np.diff(iti.astype('int'))==-1)[0] + 1] = True

    # discretize position into n_position_bins bins
    pos = (beh_data['treadmillPosition'] * n_position_bins).astype('int')

    # load dfof
    dfof = expt.signals(label='suite2p', signal_type='dfof')

    # iterate over laps to create spatial tuning array
    print(f'Calculation spatial rasters for {expt}')

    place_maps = np.zeros((len(np.unique(laps)), dfof.shape[0], 
                       n_position_bins))
    p = ProgressBar(len(np.unique(laps)) )
    for lap in range(len(np.unique(laps))):

        # only count signals on the current lap while the animal is 
        # running > velocity_threshold cm/sec, and not during ITI periods 
        valid_samples = (laps==lap) & ~iti & (velocity > velocity_threshold) & ~licking

        # calculate spike rate at each position
        place_maps[lap, :] = np.stack(
            [np.mean(dfof.iloc[:, (valid_samples) & (pos==p)], axis=1) 
             for p in range(n_position_bins)]).T

        p.update(lap)
    p.end()

    # replace nans with zeros and smooth with a gaussian
    place_maps = np.nan_to_num(np.swapaxes(place_maps, 0, 1))
    k = gaussian(12, sigma)
    k /= k.sum()
    smooth_place_maps = np.apply_along_axis(fftconvolve, -1, place_maps, k, mode='same') 
    
    return place_maps, smooth_place_maps

def get_or_make_ICRWL(expt, bd_exp):
    ICRWL = None
    ICRWL_file = os.path.join(expt.sima_path,"ICRWL.csv")
    if os.path.isfile(ICRWL_file):
        print("ICRWL.h5 is successfully loaded")
        ICRWL = pd.read_csv(ICRWL_file)
    else:
        ICRWL = make_ICRWL(bd_exp)
        ICRWL.to_csv(ICRWL_file)
        print("ICRWL created")
    return ICRWL

def get_or_make_place_maps(expt):
    place_maps = None
    smooth_place_maps = None
    place_maps_file = os.path.join(expt.sima_path,"place_maps.pkl")
    if os.path.isfile(place_maps_file):
        with open(place_maps_file,'rb') as f:
            place_maps_dict = pickle.load(f)
        print("place_maps.pkl is successfully loaded")

        place_maps = place_maps_dict['raw']
        smooth_place_maps = place_maps_dict['smooth']
    else:
        with open(place_maps_file,'wb') as f:
            place_maps, smooth_place_maps = make_place_maps(expt)
            place_maps_dict={"raw":place_maps,"smooth":smooth_place_maps}
            pickle.dump(place_maps_dict,f)
        print("place_maps.pkl created")
    return place_maps, smooth_place_maps

def decode(rasters,ctx_id,type='Logistic'):
    """To do:
    update "rasters" to extract dff information from specific cell population such as neurons with significant d' value

    IDEA:
    if not n_list:
        rasters = rasters
    elif n_list = signif_n_list: # should load ctx_A_signif_idx and ctx_B_signif_idx from DI.h5 file
        rasters = rasters[signif_n_list,:,:]
    """
    loo = LeaveOneOut()
    X = np.arange(0,len(ctx_id)) #Number of trials

    Accuracy = []

    p = ProgressBar(rasters.shape[2])
    for spatial_bin in range(rasters.shape[2]): #iterate every spatial_bin (e.g. n_position_bins = 100)
        accuracy = []
        for i, (train_index, test_index) in enumerate(loo.split(X)): #i-th fold
            # Splitting Data
            X_train = np.transpose(rasters[:,train_index,spatial_bin],(1,0)) #input dataframe in (num_lap x num_neuron) dimension
            X_test = np.transpose(rasters[:,test_index,spatial_bin],(1,0)) #input dataframe in (num_lap x num_neuron) dimension
            y_train = ctx_id[train_index] 
            y_test = ctx_id[test_index] 

            # Generating Model
            if type=='Logistic':
                # Create a Logistic Classifier
                clf = linear_model.LogisticRegression(solver='liblinear')
            elif type=='SVM':
                # Create a SVM Classifier
                clf = svm.SVC(kernel='linear') # linear kernel

            # Train the model using training sets
            clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # Evaluating the Model
            if np.array_equal(y_test,y_pred):
                accuracy.append(1)
            else:
                accuracy.append(0)
        Accuracy.append(np.mean(np.array(accuracy)))    
        p.update(spatial_bin)
    p.end
    Accuracy = np.array(Accuracy)
    return Accuracy


