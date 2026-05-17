import neo 
import numpy as np
import mne 
import re
import pandas as pd
import os
from datetime import datetime
from scipy.stats import ttest_ind


EXTRA_COLUMNS_DESCRIPTIONS = {
    'stim_onset_unity': 'Stimulus onset time in seconds since the start of the Unity game',
    'resp_onset_unity': 'Response onset time in seconds since the start of the Unity game',
    'duration': 'Duration between stimulus and response in seconds',
    'condition': 'Condition of the trial (Current, Past, Distant Past, Futur, Distant Futur)',
    'correct': 'Whether the response is correct or not (True/False)',
    'validation': 'whether the response is correct or not(Eprime)',
    'stim': 'Stimulus presented to the subject',
    'resp': 'Response given by the subject',
    'correct_year': 'Year that the subject should have answered',
    'year': 'Year during which the subject was asked to answer',
    'cross_time': 'duration of the fixation cross before trial onset',
}

def raw_from_neo(fname):
    seg_micromed = neo.MicromedIO(filename=fname)
    # Date
    date = seg_micromed.raw_annotations['blocks'][0]['rec_datetime']
    print("Date: ", date)
    # Convert the date to UTC
    segment = seg_micromed.read_segment()

    # Because here we have the same on all chan
    sfreq = segment.analogsignals[0].sampling_rate

    data = np.asarray(segment.analogsignals)[0].T
    data *= 1e-6  # putdata from microvolts to volts

    ch_names = [channel[0] for channel in seg_micromed.header['signal_channels']]
    ch_types = ['stim' if 'MKR' in ch_name else 'ecg' if 'ECG' in ch_name else 'misc' if 'EX' in ch_name else 'stim' if 'PHOTO' in ch_name else 'stim' if 'photo' in ch_name else 'seeg' for ch_name in ch_names]
 
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return(raw)

def seeg_ch_name_split(name):
    elec, idx = re.match(r'([A-Za-z]+)(\d+)', name).groups()
    return elec, int(idx)

def find_anodes_cathodes(raw):
    anodes, cathodes = [], []
    for i in range(len(raw.ch_names)-1):
        if raw.get_channel_types()[i:i+2]==['seeg', 'seeg']:
            (e1, i1), (e2, i2) = map(seeg_ch_name_split, raw.ch_names[i:i+2])
            if e1==e2:
                anodes.append(e1 + str(i1))
                cathodes.append(e2 + str(i2))
    return (anodes, cathodes)

def read_montage(subject, subjects_dir):
    # ii. import electrodes
    ch_coords_df = pd.read_csv(os.path.join(subjects_dir, subject, 'elec_recon', subject + '.PIAL'), sep=' ', header=1)
    ch_coords = ch_coords_df[['R', 'A', 'S']].to_numpy(dtype=float)/1000. # put in mm
    ch_names_df = pd.read_csv(os.path.join(subjects_dir, subject, 'elec_recon', subject + '.electrodeNames'), sep=' ', header=1, names=['name', 'Depth', 'hemisphere'])
    ch_names = ch_names_df['name'].tolist()
    ch_pos = dict(zip(ch_names, ch_coords))
    lpa, nasion, rpa = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
    lpa, nasion, rpa = lpa['r'], nasion['r'], rpa['r']
    montage_head = mne.channels.make_dig_montage(ch_pos, coord_frame='mri', nasion=nasion, lpa=lpa, rpa=rpa)
    return(montage_head)

def read_events(fname):
    dicts = list()
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            if line.startswith("UniqueID"):
                ID, Q, year, stim, _, _, _, resp, correct_year, validation, Game, cross_time, stim_onset, resp_time, resp_onset = line.split(" | ")
                
                ID = int(ID.split(":")[1])
                Q = int(Q.split("#")[1])
                year = int(year.split(":")[1])
                stim = stim.split(":")[1]
                resp = resp.split(":")[1].split("(")[0].strip()
                correct_year = correct_year.split(":")[1].strip()
                validation = validation.split(":")[1]
                Game = Game.split("#")[1]
                cross_time = float(cross_time.split(":")[1])
                stim_onset = datetime.strptime(stim_onset.split("displayed:")[1].strip(), "%H:%M:%S:%f")
                resp_time = float(resp_time.split(":")[1])
                resp_onset = datetime.strptime(resp_onset.split("answered:")[1].strip(), "%H:%M:%S:%f")
                data ={
                        "ID": ID,
                        "Q": Q,
                        "year": year,
                        "stim": stim,
                        "resp": resp,
                        "correct_year": correct_year,
                        "validation": validation,
                        "Game": Game,
                        "cross_time": cross_time,
                        "stim_onset": stim_onset,
                        "resp_time": resp_time,
                        "resp_onset": resp_onset
                    }
                dicts.append(data)

    unity_start = content[0].split('|')[0].split('Time:')[1].strip()
    unity_start = datetime.strptime(unity_start, "%H:%M:%S:%f")

    first_photo = content[2].split('completed:')[1].strip()
    first_photo = datetime.strptime(first_photo, "%H:%M:%S:%f")

    df = pd.DataFrame(dicts)

    df['stim_onset_unity'] = df['stim_onset'] - unity_start
    df['stim_onset_unity'] = [stim.total_seconds() for stim in df['stim_onset_unity']]
    df['resp_onset_unity'] = df['resp_onset'] - unity_start
    df['resp_onset_unity'] = [resp.total_seconds() for resp in df['resp_onset_unity']]
    df['duration'] = df['resp_onset_unity'] - df['stim_onset_unity']

    conditions = []
    correct = []
    for r, row in df.iterrows():
        current_year = row['year']
        correct_year = row['correct_year']
        resp = row['resp']
        # Condition
        if correct_year in ['Never', 'Always']:
            condition = correct_year
        else:
            distance = int(correct_year) - int(current_year)
            if distance == 0:
                condition = "Current"
            elif distance == -1:
                condition = "Past"
            elif distance  < -1:
                condition = "Distant Past"
            elif distance == 1:
                condition = "Futur"
            elif distance > 1:
                condition = "Distant Futur"
        conditions.append(condition)

        if resp == 'C':
            response = 'Current'
        elif resp == 'A':
            response = 'Always'
        elif resp == 'N':
            response = 'Never'
        elif resp == '+1':
            response = 'Futur'
        elif resp == '-1':
            response = 'Past'
        elif resp == '+2':
            response = 'Distant Futur'
        elif resp == '-2':
            response = 'Distant Past'
        else:
            raise ValueError(f"Unknown response: {resp}")
        
        if response == condition:
            correct.append(True)
        else:
            correct.append(False)

    df['condition'] = conditions
    df['correct'] = correct
    df['validation'] = df['validation'].astype(bool)

    return(df)

def signed_ttest(X_a, X_b):
    """
    MNE passes two data arrays, not (data, labels).
    X_a: (n_trials_a, n_times) — first group (Temporal)
    X_b: (n_trials_b, n_times) — second group (Current)
    """
    t, _ = ttest_ind(X_a, X_b, axis=0)
    return t