import numpy as np
import pandas as pd

def calculate_tuning(neuron_trace,stim_table):
    orientations = np.sort(
        stim_table["orientation"].dropna().unique()
    )
    responses = []
    directions = []

    for index, stim in stim_table.iterrows():
        start = int(stim["start"])
        end = int(stim["end"])
        fluorescence = neuron_trace[start:end]
        mean = np.mean(fluorescence)
        if pd.notna(stim["orientation"]):
            directions.append(stim["orientation"])
            responses.append(mean)
    responses = np.asarray(responses)
    directions = np.asarray(directions)
    mean_responses = []
    std_responses = []

    for direction in orientations:
        mask = directions == direction
        mean_responses.append(np.mean(responses[mask]))
        std_responses.append(np.std(responses[mask]))

    return mean_responses, std_responses

def calculate_dsi(mean_responses, orientations):
    preferred_idx = np.argmax(mean_responses)
    preferred_direction = orientations[preferred_idx]
    preferred_response = mean_responses[preferred_idx]
    opposite_direction = (preferred_direction + 180) % 360
    oppoiste_idx = np.where(orientations==opposite_direction)[0][0]
    opposite_response = mean_responses[oppoiste_idx]
    denominator = preferred_response + opposite_response

    if denominator > 1e-10:
        dsi = (preferred_response - opposite_response) / denominator
    else:
        dsi = np.nan
    return dsi