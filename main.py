from src.data_loader import load_data
from src.tuning import calculate_tuning
from src.tuning import calculate_tuning, calculate_dsi
from src.visualization import plot_tuning,plot_direction_errors
from src.decoding import get_pref_dir,population_vector_decode
import numpy as np
import pandas as pd 
def main():
    dataset, timestamps, dff, stim_table = load_data()
    print("Fluorescence shape:", dff.shape)
    print("Stimulus presentations:", len(stim_table))

    means, stds = calculate_tuning(dff[0], stim_table)
    print("Neuron 0 mean responses:", means)
    print("Neuron 0 standard deviations:", stds)

    orientations = np.sort(
    stim_table["orientation"].dropna().unique()
    )
    for neuron_idx in range(5):
        means, stds = calculate_tuning(dff[neuron_idx], stim_table)
        dsi = calculate_dsi(means, orientations)
        print(f"Neuron {neuron_idx} DSI: {dsi:.4f}")

    means, stds = calculate_tuning(dff[0], stim_table)

    valid_stims = stim_table[stim_table["orientation"].notna()]
    num_stims = len(valid_stims)
    split_idx = int(0.8 * num_stims)
    train_stims = valid_stims.iloc[:split_idx]
    test_stims = valid_stims.iloc[split_idx:]
    print("Training presentations:", len(train_stims))
    print("Testing presentations:", len(test_stims))
    baseline_direction = train_stims["orientation"].mode()[0]
    print("Baseline direction:", baseline_direction)
    baseline_errors = []

    for index, stim in test_stims.iterrows():
        act_dir = stim["orientation"]
        err = abs(((baseline_direction - act_dir + 180) % 360) - 180)
        baseline_errors.append(err)
    baseline_mean_error = np.mean(baseline_errors)
    print("Baseline mean error:", baseline_mean_error)

    num_neurons = 50
    preferred_directions = get_pref_dir(dff, train_stims, num_neurons=num_neurons)    
    print("Preferred directions:", preferred_directions)
    unique_dirs, counts = np.unique(preferred_directions,return_counts=True)
    for direction, count in zip(unique_dirs, counts):
        print(direction, count)

    # plot_tuning(orientations, means, stds, neuron_idx=0)

    responses = np.array([0.03, 0.11])
    baselines = np.array([0.02, 0.10])
    corrected_responses= np.subtract(responses,baselines)
    print(corrected_responses)

    neuron_baselines =[]
    for i in range(num_neurons):
        mean = np.mean(dff[i])
        neuron_baselines.append(mean)
    print("First 5 neuron baselines:", neuron_baselines[:5])

    first_stim = stim_table[stim_table["orientation"].notna()].iloc[0]
    start = int(first_stim["start"])
    end = int(first_stim["end"])
    actual_direction = first_stim["orientation"]

    responses = []
    for neuron_idx in range(num_neurons):
        fluorescence = dff[neuron_idx][start:end]
        mean = np.mean(fluorescence)
        responses.append(mean)
    responses = np.asarray(responses)
    print("Neural responses:", responses)

    centered_responses = responses - np.asarray(neuron_baselines)
    print("First 5 original responses:", responses[:5])
    print("First 5 centered responses:", centered_responses[:5])

    decoded_direction = population_vector_decode(centered_responses, preferred_directions)
    print("Actual direction:", actual_direction)
    print("Decoded direction:", decoded_direction)
    angular_error = abs(((decoded_direction-actual_direction+180)%360)-180)

    actual_directions = []
    decoded_directions = []
    angular_errors = []

    for index, stim in test_stims.iterrows():
        start = int(stim["start"])
        end = int(stim["end"])
        act_dir = stim["orientation"]
        responses = []
        for i in range(num_neurons):
            fluorescence = dff[i][start:end]
            mean = np.mean(fluorescence)
            responses.append(mean)
        responses = np.asarray(responses)
        #centered_responses = responses - neuron_baselines
        decoded_dir = population_vector_decode(responses,preferred_directions)
        ang_err = abs(((decoded_dir-act_dir+180)%360)-180)
        actual_directions.append(act_dir)
        decoded_directions.append(decoded_dir)
        angular_errors.append(ang_err)

    mean_error = np.nanmean(angular_errors)
    print("Mean error", mean_error)
    num_failed = np.sum(np.isnan(decoded_directions))
    print("Failed predictions:", num_failed)
    for i in range(len(actual_directions)):
        print(
            f"Stimulus {i}: "
            f"Actual = {actual_directions[i]:.0f}°, "
            f"Decoded = {decoded_directions[i]:.1f}°, "
            f"Error = {angular_errors[i]:.1f}°"
        )
    actual_directions = np.asarray(actual_directions)
    angular_errors = np.asarray(angular_errors)
    unique_test_dirs = np.sort(test_stims["orientation"].unique())
    direction_errors = []
    for i in range(len(unique_test_dirs)):
        print(unique_test_dirs[i])
        mask = actual_directions == unique_test_dirs[i]
        errors_for_direction = angular_errors[mask]
        avg_mean_err = np.nanmean(errors_for_direction)
        direction_errors.append(avg_mean_err)
        print("Average mean error:", avg_mean_err)
    plot_direction_errors(unique_test_dirs, direction_errors)
if __name__ == "__main__":
    main()