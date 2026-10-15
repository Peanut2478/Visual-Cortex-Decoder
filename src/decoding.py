import numpy as np
from src.tuning import calculate_tuning

def get_pref_dir(dff, stim_table, num_neurons=5):
    orientations = np.sort(
        stim_table["orientation"].dropna().unique()
    )
    preferred_directions = []

    for i in range(num_neurons):
        mean, std = calculate_tuning(dff[i], stim_table)
        preferred_idx = np.argmax(mean)
        preferred_direction = orientations[preferred_idx]
        preferred_directions.append(preferred_direction)

    return  np.array(preferred_directions)
def population_vector_decode(responses, preferred_directions):
    directions_rad = np.deg2rad(preferred_directions)

    x = np.maximum(responses, 0) * np.cos(directions_rad)
    y = np.maximum(responses,0) * np.sin(directions_rad)

    x_total = np.sum(x)
    y_total = np.sum(y)
    theta = np.arctan2(y_total,x_total)
    decoded_dir = np.rad2deg(theta) % 360
    
    if np.hypot(x_total, y_total) < 1e-10:
        return np.nan
    return decoded_dir