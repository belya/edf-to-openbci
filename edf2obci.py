import numpy as np
import mne
import sys
import pandas as pd
from datetime import datetime


CHANNELS_10_20 = [

           'Fp1', 'Fp2', 

    'F7', 'F3', 'Fz', 'F4', 'F8', 

'A2-A1', 'T3', 'C3', 'Cz', 'C4', 'T4', 

    'T5', 'P3', 'Pz', 'P4', 'T6', 

            'O1', 'O2'

]


CHANNELS_MAP_8 = {
    1: "Fp1",
    2: "Fp2",
    3: "C3",
    4: "C4",
    5: "T5",
    6: "T6",
    7: "O1",
    8: "O2"
}


def get_channels_map():
    return [
        CHANNELS_MAP_8[key] 
        for key in sorted(CHANNELS_MAP_8.keys())
    ]


def read_edf_data(path):
    edf = mne.io.read_raw_edf(edf_file, preload=True)
    return edf


def edf_to_data_frame(edf):
    eeg_df = edf.to_data_frame()
    prepared_eeg_df = pd.DataFrame()

    for channel in get_channels_map():
        for column in eeg_df.columns:
            if channel in column:
                prepared_eeg_df[channel] = eeg_df[column].tolist()
                break

    prepared_eeg_df["accel_x"] = 0
    prepared_eeg_df["accel_y"] = 0
    prepared_eeg_df["accel_z"] = 0

    current_timestamp = float(int(datetime.now().timestamp()))
    timestamp = current_timestamp + edf.times
    prepared_eeg_df["time"] = pd.to_datetime(timestamp, unit='s').strftime('%H:%M:%S.%f').str[:-3]
    prepared_eeg_df["timestamp"] = (timestamp).astype(int)

    return prepared_eeg_df


def extract_sample_rate(eeg_df):
    float_frequency = eeg_df.shape[0] / (eeg_df["timestamp"].max() - eeg_df["timestamp"].min())
    int_frequency = int(round(float_frequency / 10) * 10)
    return float(int_frequency)


def save_obci(edf_file, eeg_df):
    obci_name = "converted/" + ".".join(edf_file.split(".")[:-1]) + ".txt"
    print("Writing to", obci_name)

    sample_rate = extract_sample_rate(eeg_df)
    channels = eeg_df.shape[1] - 5

    with open(obci_name, "w") as file:
        file.write("%OpenBCI Raw EEG Data\n")
        file.write("%Number of channels = {}\n".format(channels))
        file.write("%Sample Rate = {} Hz\n".format(sample_rate))
        file.write("%First Column = SampleIndex\n")
        file.write("%Last Column = Timestamp\n")
        file.write("%Other Columns = EEG data in microvolts followed by Accel Data (in G) interleaved with Aux Data\n")

    eeg_df.to_csv(obci_name, mode='a', header=None, index=True, float_format='%.2f')


if __name__ == "__main__":
    edf_file = sys.argv[1] 
    edf = read_edf_data(edf_file)
    eeg_df = edf_to_data_frame(edf)
    save_obci(edf_file, eeg_df)