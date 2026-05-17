import mne 
import numpy as np
import matplotlib.pyplot as plt

def get_epochs_from_raw_filtered(raw_filtered, events, event_id, tmin, tmax, subject="02", reference="bipolar", pad=0.5):
    # Get averages for each condition
    epochs = mne.Epochs(raw_filtered, events=events, event_id=event_id, detrend=1, baseline=None, tmin=tmin-pad, tmax=tmax+pad, preload=True)

    if reference == "bipolar":
        print("Bipolar Referencing")
        # TODO: Automic Shaft detection and referencing

        if subject == "01":
            epochs_ref = mne.set_bipolar_reference(
                epochs,
                ["HAG1", "HAG2", "HAG3", "HAG4", "HPG1", "HPG2", "HPG3", "PHG1", "PHG2", "PHG3", "PHG4", "TOL2"],
                ["HAG2", "HAG3", "HAG4", "HAG5", "HPG2", "HPG3", "HPG4", "PHG2", "PHG3", "PHG4", "PHG6", "TOL5"],
                ch_name=["HAG1-HAG2", "HAG2-HAG3", "HAG3-HAG4", "HAG4-HAG5", "HPG1-HPG2", "HPG2-HPG3", "HPG3-HPG4", "PHG1-PHG2", "PHG2-PHG3", "PHG3-PHG4", "PHG4-PHG6", "TOL2-TOL5"]
            )

        if subject == "02":
            epochs_ref = mne.set_bipolar_reference(
                epochs,
                ["TOD2", "TOD3", "PHD1", "PHD2", "PHD3", "PHD4", "HPD1", "HPD2", "HPD3", "HPD4", "HAD1", "HAD2", "HAD3"],
                ["TOD3", "TOD4", "PHD2", "PHD3", "PHD4", "PHD5", "HPD2", "HPD3", "HPD4", "HPD5", "HAD2", "HAD3", "HAD4"],
                ch_name=["TOD2-TOD3", "TOD3-TOD4", "PHD1-PHD2", "PHD2-PHD3", "PHD3-PHD4", "PHD4-PHD5", "HPD1-HPD2", "HPD2-HPD3", "HPD3-HPD4", "HPD4-HPD5", "HAD1-HAD2", "HAD2-HAD3", "HAD3-HAD4"]
            )

        epochs_ref_no_hga = epochs_ref.copy()

        # Filter (80-200Hz)
        # epochs_ref.filter(80, 200, picks='seeg', method="iir", iir_params=None, n_jobs=9)

        # ----------------------------------------------------------
        # New filter: from 70Hz to 200Hz, do a bandpass filter every 10Hz, keep the envelope aside, and then average the envelopes
        # Each band is normalized by its own mean amplitude across time and trials
        envelopes = []
        band_mean_amps = []

        for ifreq in np.arange(70, 201, 10):
            # Always filter from the original, unmodified epochs
            epochs_filt = epochs_ref.copy().filter(
                ifreq, ifreq + 10,
                picks='seeg',
                method='iir',
                iir_params=None,
                n_jobs=9
            )
            
            # Compute the analytic signal envelope via Hilbert transform
            epochs_filt.apply_hilbert(picks='seeg', envelope=True)

            # Mean over band
            data = epochs_filt.get_data(picks='seeg')           # (n_epochs, n_ch, n_times)
            mean_amp = data.mean(axis=-1, keepdims=True)
            band_mean_amps.append(mean_amp)

            # Baseline normalization TODO: Check if needed
            # baseline_mask = epochs_filt.times < 0               # pre-stimulus samples (t < 0)

            # baseline_mean = data[:, :, baseline_mask].mean(axis=-1, keepdims=True)  # (n_epochs, n_ch, 1)
            # data_db = data - baseline_mean   

            data_norm = data / (mean_amp + 1e-10)

            # Put normalized data back into the epochs object
            #epochs_filt._data[:, :len(epochs_filt.ch_names), :] = data_db  # update in place
            
            # Store the envelope data: shape (n_epochs, n_channels, n_times)
            #envelopes.append(epochs_filt.get_data(picks='seeg'))
            envelopes.append(data_norm)

        # Average envelopes across frequency bands → shape (n_epochs, n_channels, n_times)
        envelopes = np.array(envelopes)          # (n_bands, n_epochs, n_channels, n_times)
        band_mean_amps = np.array(band_mean_amps)
        global_mean_amp = band_mean_amps.mean(axis=0)

        mean_envelope = envelopes.mean(axis=0)         # (n_epochs, n_ch, n_times)
        mean_envelope = mean_envelope * global_mean_amp  # restore physical units

        # Baseline normalization across all bands together
        baseline_mask = epochs_ref.times < 0  # pre-stimulus samples
        baseline_mean = mean_envelope[:, :, baseline_mask].mean(axis=-1, keepdims=True)
        mean_envelope = (mean_envelope - baseline_mean) / (baseline_mean + 1e-10) * 100  # % change

        # Change mean_envelope back to epochs_ref._data shape and update in place
        # Warning: all information of FB below 70Hz is lost --> epochs_ref now ONLY for HGA
        epochs_ref._data[:, :len(epochs_ref.ch_names), :] = mean_envelope

        print(f"Computed HGA envelope: {mean_envelope.shape}  →  (epochs, channels, times)")
        # ----------------------------------------------------------

        # Hilbert Transform + Absolute Value
        #epochs_ref.apply_hilbert(envelope=True) # envelope=True returns the absolute value of the signal (see doc)

        # Envelope check, more informative than PSD after hilbert
        evoked_env = epochs_ref.average()
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(epochs_ref.times, evoked_env.get_data(picks=['PHD2-PHD3'])[0])
        ax.axvline(0, color='r', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("HFA envelope — PHD2-PHD3")
        plt.savefig(f"figures/{subject}/test_filtering/{subject}_{reference}_envelope_check.png", dpi=300)
        plt.close(fig)

        # Crop to remove filter edge effects
        epochs_ref.crop(tmin=tmin, tmax=tmax)
        #epochs_ref.crop(tmin=tmin, tmax=tmax)

        return epochs_ref, epochs_ref_no_hga, band_mean_amps, envelopes

    if reference == "bipolar_shaft":
        print("Bipolar Shaft Referencing")

        if subject == "01":
            epochs_ref = mne.set_bipolar_reference(
                epochs,
                [],
                [],
                ch_name=[]
            )

        if subject == "02":
            epochs_ref = mne.set_bipolar_reference(
                epochs,
                ["TOD2", "TOD2", "TOD3", "TOD4", "TOD4", "PHD1", "PHD2", "PHD3", "PHD4", "PHD5", "HPD1", "HPD2", "HPD3", "HPD3", "HPD4", "HPD5",],
                ["PHD1", "PHD2", "PHD3", "PHD4", "PHD5", "HPD1", "HPD2", "HPD3", "HPD4", "HPD5", "HAD1", "HAD2", "HAD2", "HAD3", "HAD3", "HAD4"],
                ch_name=["TOD2-PHD1", "TOD2-PHD2", "TOD3-PHD3", "TOD4-PHD4", "TOD4-PHD5", "PHD1-HPD1", "PHD2-HPD2", "PHD3-HPD3", "PHD4-HPD4", "PHD5-HPD5", "HPD1-HAD1", "HPD2-HAD2", "HPD3-HAD2", "HPD3-HAD3", "HPD4-HAD3", "HPD5-HAD4" ]
            )

        # Filter (80-200Hz)
        epochs_ref.filter(80, 200, picks='seeg', method="iir", iir_params=None, n_jobs=9)

        # Hilbert Transform + Absolute Value
        epochs_ref.apply_hilbert(envelope=True) # envelope=True returns the absolute value of the signal (see doc)

        # Envelope check, more informative than PSD after hilbert
        evoked_env = epochs_ref.average()
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(epochs_ref.times, evoked_env.get_data(picks=['PHD3-HPD3'])[0])
        ax.axvline(0, color='r', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("HFA envelope — PHD3-HPD3")
        plt.savefig(f"figures/{subject}/test_filtering/{subject}_{reference}_envelope_check.png", dpi=300)
        plt.close(fig)

        # Baseline normalization TODO: Check if needed
        data = epochs_ref.get_data(picks='seeg')           # (n_epochs, n_ch, n_times)
        baseline_mask = epochs_ref.times < 0               # pre-stimulus samples (t < 0)

        baseline_mean = data[:, :, baseline_mask].mean(axis=-1, keepdims=True)  # (n_epochs, n_ch, 1)
        data_db = data - baseline_mean   

        # Put normalized data back into the epochs object
        epochs_ref._data[:, :len(epochs_ref.ch_names), :] = data_db  # update in place

    if reference == "average": # TODO: Implement average referencing
        print("Average Referencing")
        epochs_ref = epochs.copy().set_eeg_reference(ref_channels='average', projection=False)
        # TODO: Compute average value of all electrodes over time period and substract it (average over all time points or average for each bin/time point?)

    if reference == "no_reference":
        print("No Referencing")
        epochs_ref = epochs
    
    return epochs_ref