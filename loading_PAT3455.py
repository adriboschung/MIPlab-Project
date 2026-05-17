from functions import read_events,raw_from_neo
import mne 
import numpy as np
import pandas as pd
from pathlib import Path
import mne_bids 
from mne_bids import (
    find_matching_paths,
    read_raw_bids,
)

def PAT_3455_loading(data_source, bids_root, bids_path, experiment_dir, trc_dir, fs_subjects_dir, subject, patient, session, task):    
# Read data
    if data_source == "BIDS" or data_source == "Check":
        print(f"\nLoading data from: {bids_path.fpath}")
        raw = read_raw_bids(bids_path, verbose=False)
        raw_bids = raw.copy()

    if data_source == "TRC" or data_source == "Check":
        fname_events = experiment_dir
        fname_raw = trc_dir

        print(f"\n--- Traitement TRC ---")
        df = read_events(fname_events)
        raw = raw_from_neo(fname_raw)

        if 'photo' in raw.ch_names:
            raw.rename_channels({'photo': 'photodiode'})
        if 'PHOTO' in raw.ch_names:
            raw.rename_channels({'PHOTO': 'photodiode'})
        
        if 'ECG-' in raw.ch_names and 'ECG+' in raw.ch_names:
            raw = mne.set_bipolar_reference(raw, 'ECG-', 'ECG+', ch_name='ECG', drop_refs=True)
            raw.set_channel_types({'ECG': 'ecg'})

        if subject == "01":
            negPhotoSig = raw.get_data(picks=['photodiode'])[0] <-0.003

            # Get suspected beginning and end of fixation cross
            mrkTCbeg = np.where(np.diff(np.int64(negPhotoSig),n=1)>0)[0]
            mrkTCend = np.where(np.diff(np.int64(negPhotoSig),n=1)<0)[0]

            onsets  = []
            durations = []

            for onset, end in zip(mrkTCbeg, mrkTCend):
                duration = (end - onset) / raw.info['sfreq']
                if duration > 1.6 and duration < 2.4:
                    if onset > 1546240.0:
                        onsets.append(onset / raw.info['sfreq'])
                        durations.append(duration)

            delay = np.abs(onsets[0] - df['resp_onset_unity'].values[0])
            df['stim_onset_abs'] = df['stim_onset_unity'] + delay
            df['resp_onset_abs'] = df['resp_onset_unity'] + delay
            
            tmin_abs = 700
            tmax_abs = 1600
        
        if subject == "02":
            delay = (1844.475 - 333.112) # TODO: Comment est calculé ce délai?
            df['stim_onset_abs'] = df['stim_onset_unity'] + delay
            df['resp_onset_abs'] = df['resp_onset_unity'] + delay

            tmin_abs = 1600
            tmax_abs = 3200
        
        print(f"Original signal length : {raw.times[-1]:.2f}s")
        raw.crop(tmin=tmin_abs, tmax=tmax_abs)
        print(f"Length after crop ({tmin_abs}-{tmax_abs}) : {raw.times[-1]:.2f}s")

        raw = raw.copy() 

        mask = (df['stim_onset_abs'] >= tmin_abs) & (df['stim_onset_abs'] <= tmax_abs)
        df_filtered = df[mask].reset_index(drop=True)

        # Keep only correct trials
        df_filtered = df_filtered[df_filtered['correct'] == True].reset_index(drop=True)
        print(f"Correct trials: {len(df_filtered)} / {mask.sum()}")
        
        
        if len(df_filtered) == 0:
            print(f"ERREUR CRITIQUE : Aucun événement ne tombe dans la fenêtre [{tmin_abs}, {tmax_abs}]s !")
            print(f"Vérifiez vos colonnes 'stim_onset_unity' et le délai calculé ({delay}).")
            print(f"Min stim_onset_abs: {df['stim_onset_abs'].min()}, Max: {df['stim_onset_abs'].max()}")
        else:
            df_filtered['onset_rel'] = df_filtered['stim_onset_abs'] - tmin_abs
            
            onsets = df_filtered['onset_rel'].values
            durations = df_filtered['duration'].values
            descriptions = df_filtered['condition'].values

            valid_idx = ~np.isnan(onsets) & ~np.isnan(durations)
            onsets = onsets[valid_idx]
            durations = durations[valid_idx]
            descriptions = descriptions[valid_idx]

            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions.astype(str)
            )
            
            raw.set_annotations(annotations)

            print(f"Nombre d'annotations dans l'objet raw : {len(raw.annotations)}")
            if len(raw.annotations) > 0:
                print(f"Exemple d'annotation : {raw.annotations[0]}")
            else:
                print("ÉCHEC : Les annotations n'ont pas été enregistrées malgré la création.")

        raw_trc = raw.copy()

    trc_channel_names = raw.ch_names
    print(f"TRC channel names (first 10): {trc_channel_names[:10]}")
    print(f"Total channels: {len(trc_channel_names)}")


    # Path to electrode reconstruction files
    elec_recon_path = bids_root / "sourcedata" / "reconstructions" / patient / "elec_recon"
    coord_type = "LEPTO"  # Use LEPTO !!!!!
    coord_file = elec_recon_path / f"{patient}.{coord_type}"
    print(f"\nLoading coordinates from: {coord_file}")
    print(f"File exists: {coord_file.exists()}")

    # Load bad annotations
    bad_annotation_path = bids_root / f"sub-{subject}" / f"ses-{session}" / "ieeg" / f"sub-{subject}_ses-{session}_task-{task}_annot.csv"

    if bad_annotation_path.exists():
        # Check if file is empty (size 0) or has only headers
        if bad_annotation_path.stat().st_size == 0:
            print(f"Warning: {bad_annotation_path} is empty. Skipping.")
            bad_annotations = mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.annotations.orig_time)
        else:
            try:
                # Try reading with pandas first to check content
                df_test = pd.read_csv(bad_annotation_path)
                if df_test.empty:
                    print(f"Warning: {bad_annotation_path} has no data rows. Skipping.")
                    bad_annotations = mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.annotations.orig_time)
                else:
                    # File has data, proceed with MNE
                    bad_annotations = mne.read_annotations(bad_annotation_path)
                    
                    # # Sync orig_time if needed
                    # if raw.annotations.orig_time is not None:
                    #     bad_annotations = mne.Annotations(
                    #         onset=bad_annotations.onset,
                    #         duration=bad_annotations.duration,
                    #         description=bad_annotations.description,
                    #         orig_time=raw.annotations.orig_time
                    #     )
            except Exception as e:
                print(f"Error reading {bad_annotation_path}: {e}")
                bad_annotations = mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.annotations.orig_time)

        # Add to raw
        # raw.set_annotations(raw.annotations + bad_annotations)
        raw.annotations.append(
            onset=bad_annotations.onset,
            duration=bad_annotations.duration,
            description=bad_annotations.description
        )
        print(f"Added {len(bad_annotations)} bad segments.")

        # Save annotations
        annotations_path = f"/home/aboschun/MIPlab-Project/data/annotations/sub-{subject}_annot.fif"
        raw.annotations.save(annotations_path, overwrite=True)
        print(f"Saved annotations to {annotations_path}")


    # Clean the coordinates and creates a montage that matches the TRC channels
    if coord_file.exists():
        # Read the coordinate file
        # These are typically text files with coordinates in mm
        coords = []
        with open(coord_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        # Parse x y z coordinates
                        parts = line.split()
                        if len(parts) >= 3:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            coords.append([x, y, z])
                    except:
                        continue
        
        coords = np.array(coords)
        print(f"Loaded {len(coords)} coordinates")
        
        # Also load electrode names if available
        names_file = elec_recon_path / f"{patient}.electrodeNames"
        if names_file.exists():
            with open(names_file, 'r') as f:
                electrode_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"Loaded {len(electrode_names)} electrode names")
        else:
            electrode_names = [f"ELEC{i+1:03d}" for i in range(len(coords))]
        
        # Check if number matches TRC channels
        print(f"\nCoordinate count: {len(coords)}")
        print(f"TRC channel count: {len(trc_channel_names)}")
        print(f"Electrode names count: {len(electrode_names)}")
        
        if len(coords) == len(trc_channel_names):
            print(" Coordinate count matches TRC channels - using as-is")
            montage_coords = coords
            montage_names = trc_channel_names
        else:
            print(" Count mismatch - need to map coordinates to TRC channels")
            
            # Create mapping dictionary
            coord_dict = dict(zip(electrode_names, coords))
            
            # Map to TRC order
            montage_coords = []
            montage_names = []
            
            for ch_name in trc_channel_names:
                # Try exact match first
                if ch_name in coord_dict:
                    montage_coords.append(coord_dict[ch_name])
                    montage_names.append(ch_name)
                else:
                    # Try without spaces/special chars
                    ch_clean = ch_name.replace(' ', '').replace('-', '').replace("'", "")
                    found = False
                    for coord_name in coord_dict.keys():
                        coord_clean = coord_name.replace(' ', '').replace('-', '').replace("'", "")
                        if ch_clean in coord_clean or coord_clean in ch_clean:
                            montage_coords.append(coord_dict[coord_name])
                            montage_names.append(ch_name)
                            print(f"  Matched {ch_name} to {coord_name}")
                            found = True
                            break
                    
                    if not found:
                        print(f"  Warning: No match for {ch_name}")
                        montage_coords.append([np.nan, np.nan, np.nan])
                        montage_names.append(ch_name)
            
            montage_coords = np.array(montage_coords)
        montage_coords = montage_coords / 1000  # Convert mm to m for MNE
        
        # Create montage
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(montage_names, montage_coords)),
            coord_frame='mri'  # These are in MRI coordinates
        )
        
        # Set montage to raw
        raw.set_montage(montage)
        print(f"\n✓ Created montage with {len(montage_names)} electrodes")
        
        # Verify first few electrodes
        print("\nFirst 10 electrode positions:")
        for i, ch_name in enumerate(montage_names[:10]):
            pos = montage.get_positions()['ch_pos'][ch_name]
            print(f"  {ch_name}: ({pos[0]}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        # Save montage for future use
        montage_path = Path.cwd() / f"sub-{subject}_montage.fif"
        montage.save(montage_path, overwrite=True)
        print(f"Saved montage to {montage_path}")
        
        # Now you can get volume labels if FreeSurfer subject exists
        fs_subject = patient
        subjects_dir = bids_root / "sourcedata" / "reconstructions"
        
        aparcaseg_path = f"/media/RCPNAS/sEEG_MARS_Alison/sourcedata/reconstructions/{patient}/mri/aparc+aseg.mgz"

        if Path(aparcaseg_path).exists():
            try:
                labels, colors = mne.get_montage_volume_labels(
                    montage,
                    patient,
                    subjects_dir=str(subjects_dir),
                    aseg="aparc+aseg"
                )
                
                # labels and colors are dicts keyed by channel name
                # Build DataFrame from the dicts directly
                results_df = pd.DataFrame({
                    'channel': list(labels.keys()),
                    'label': [labels[ch] for ch in labels.keys()],
                    'color': [colors[ch] for ch in labels.keys()]
                })
                
                output_path = Path.cwd() / f"sub-{subject}_electrode_labels.csv"
                results_df.to_csv(output_path, index=False)
                print(f"Saved electrode labels to {output_path}")
                print(results_df.head())
                
            except Exception as e:
                print(f"Could not get volume labels: {e}")

    if coord_file.exists() == False:
        montage = None

    # Now visualize electrodes on brain
    print(f"\nCreating brain visualization for {fs_subject}")

    # Create brain
    brain = mne.viz.Brain(
        fs_subject,
        subjects_dir=str(fs_subjects_dir),
        cortex="low_contrast",
        alpha=0.25,
        background="white",
        figure=1,
    )

    # Add electrodes if montage exists
    if montage is not None:
        try:
            # TODO: Check the transform
            trans = mne.transforms.Transform('head', 'mri')  # TODO: Why does the inverse trans give the same result
            brain.add_sensors(raw.info, trans=trans)
            print("Added electrodes to brain")
        except Exception as e:
            print(f"Could not add electrodes: {e}")

    print(f"Montage exists: {montage is not None}")

    if montage is None:
        print("\nNo montage found in raw data.")
        print("Looking for electrodes.tsv file...")
        
        # Look for electrodes.tsv
        electrodes_path = bids_root / f"sub-{subject}" / f"ses-{session}" / "ieeg" / f"sub-{subject}_ses-{session}_task-{task}_electrodes.tsv"
        print(f"Electrodes file exists: {electrodes_path.exists()}")
        
        if electrodes_path.exists():
            # Read electrodes.tsv
            electrodes_df = pd.read_csv(electrodes_path, sep='\t')
            print(f"\nElectrodes file contents:")
            print(f"Columns: {electrodes_df.columns.tolist()}")
            print(f"Number of electrodes: {len(electrodes_df)}")
            print("\nFirst few electrodes:")
            print(electrodes_df.head())
            
            # Check coordinate columns
            coord_cols = ['x', 'y', 'z']
            if all(col in electrodes_df.columns for col in coord_cols):
                print("\n✓ Found coordinate columns (x, y, z)")
                
                # Create montage
                montage = mne.channels.make_dig_montage(
                    ch_pos=dict(zip(electrodes_df['name'], 
                                electrodes_df[coord_cols].values)),
                    coord_frame='mri'  # or 'head' - check which one
                )
                raw.set_montage(montage)
                print(f"Created and set montage with {len(electrodes_df)} electrodes")
            else:
                print(f"Expected coordinate columns not found. Available columns: {electrodes_df.columns.tolist()}")
        else:
            print("No electrodes.tsv file found in BIDS directory")
            
            # Look for alternative coordinate files
            coord_files = list(Path(bids_root / f"sub-{subject}").rglob("*coord*"))
            print(f"Alternative coordinate files found: {coord_files}")

    # Save images
    output_dir = Path.cwd() / "figures" / f"{subject}"
    output_dir.mkdir(exist_ok=True)
    
    # Save different views
    views = [
        ("lateral_right", dict(azimuth=90, elevation=90, distance=400)),
        ("lateral_left", dict(azimuth=-90, elevation=90, distance=400)),
        ("top", dict(azimuth=0, elevation=90, distance=400)),
        ("front", dict(azimuth=0, elevation=0, distance=400)),
        ("back", dict(azimuth=180, elevation=0, distance=400)),
    ]

    for view_name, view_kwargs in views:
        brain.show_view(**view_kwargs)
        output_file = output_dir / "electrodes" /f"{subject}_electrodes_{view_name}.png"
        brain.save_image(str(output_file))
        print(f"Saved: {output_file}")

    brain.close()
    print(f"\nAll images saved to {output_dir}")

    # Save raw 
    raw_save_path = f"/home/aboschun/MIPlab-Project/data/raw/raw_sub-{subject}.fif"
    raw.save(raw_save_path, overwrite=True)
    print(f"Saved raw to {raw_save_path}")

    return df, raw, montage, annotations