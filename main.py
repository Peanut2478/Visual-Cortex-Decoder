from src.data_loader import load_data
from src.tuning import calculate_tuning, calculate_dsi
from src.visualization import plot_tuning,plot_direction_errors
from src.decoding import get_pref_dir,population_vector_decode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
def main():
    dataset, timestamps, dff, stim_table = load_data()
    print("Fluorescence shape:", dff.shape)
    print("Stimulus presentations:", len(stim_table))
    num_neurons = dff.shape[0]
    print("Number of neurons:", num_neurons)
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
    preferred_directions = get_pref_dir(dff, train_stims, num_neurons=num_neurons)    

    train_responses = []
    for _, stim in train_stims.iterrows():
        start = int(stim["start"])
        end = int(stim["end"])
        mean_responses = np.mean(dff[:, start:end], axis=1)
        train_responses.append(mean_responses)
    train_responses =  np.array(train_responses)
    print("Training response shape:", train_responses.shape)

    training_baselines = np.mean(train_responses, axis=0)
    centered_train = train_responses - training_baselines
    max_responses = np.max(centered_train, axis=0)
    safe_max = np.where(max_responses <= 0, 1, max_responses)
    normalized_train = centered_train / safe_max
    print("Normalized training shape:", normalized_train.shape)
    print("First 5 normalization factors:", safe_max[:5])

    test_responses = []
    for _, stim in test_stims.iterrows():
        start = int(stim["start"])
        end = int(stim["end"])
        mean_responses = np.mean(dff[:, start:end], axis=1)
        test_responses.append(mean_responses)
    test_responses = np.asarray(test_responses)
    centered_test = test_responses - training_baselines
    normalized_test = centered_test / safe_max
    print("Normalized test shape:", normalized_test.shape)

    normalized_predictions = []

    for response in normalized_test:
        decoded = population_vector_decode(
            response,
            preferred_directions
        )
        normalized_predictions.append(decoded)
    normalized_predictions = np.asarray(normalized_predictions)
    print("First 5 normalized predictions:", normalized_predictions[:5])
    print("NaN count:", np.isnan(normalized_predictions).sum())

    rounded_normalized = (np.round(normalized_predictions / 45) * 45) % 360
    valid_normalized = ~np.isnan(rounded_normalized)
    actual_valid = np.asarray(test_stims["orientation"])[valid_normalized]
    predicted_valid = rounded_normalized[valid_normalized]
    normalized_accuracy = np.mean(actual_valid == predicted_valid) * 100

    print("Normalized decoder accuracy:", normalized_accuracy)
    print("Valid normalized predictions:", valid_normalized.sum())

    baseline_predictions = []
    for response in centered_test:
        decoded = population_vector_decode(
            response,
            preferred_directions
        )
        baseline_predictions.append(decoded)
    baseline_predictions = np.asarray(baseline_predictions)
    rounded_baseline = (np.round(baseline_predictions / 45) * 45) % 360
    valid_baseline = ~np.isnan(rounded_baseline)
    actual_baseline = np.asarray(test_stims["orientation"])[valid_baseline]
    predicted_baseline = rounded_baseline[valid_baseline]
    baseline_accuracy = np.mean(actual_baseline == predicted_baseline) * 100
    print("Fair baseline accuracy:", baseline_accuracy)
    print("Normalized accuracy:", normalized_accuracy)

    baseline_direction = train_stims["orientation"].mode()[0]
    print("Baseline direction:", baseline_direction)
    baseline_errors = []

    for index, stim in test_stims.iterrows():
        act_dir = stim["orientation"]
        err = abs(((baseline_direction - act_dir + 180) % 360) - 180)
        baseline_errors.append(err)
    baseline_mean_error = np.mean(baseline_errors)
    print("Baseline mean error:", baseline_mean_error)

    
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
   # plot_direction_errors(unique_test_dirs, direction_errors)

    print("Actual directions:", actual_directions[:10])
    print("Predicted directions:", decoded_directions[:10])
    print("Number of predictions:", len(decoded_directions))

    redicted_array = np.array(decoded_directions)
    rounded_predictions = []
    for d in decoded_directions:
        rounded = (np.round(d/45) * 45) %360
        rounded_predictions.append(rounded)
    print("Rounded predictions:", rounded_predictions[:10])
    print("Number of rounded predictions:", len(rounded_predictions))
    rounded_array = np.array(rounded_predictions)

    print("NaN count:", np.isnan(rounded_array).sum())
    print("NaN indices:", np.where(np.isnan(rounded_array))[0])
    print("NaN actual directions:", np.array(actual_directions)[np.isnan(rounded_array)])

    valid_mask = ~np.isnan(rounded_array)

    valid_actual = np.array(actual_directions)[valid_mask]
    valid_predicted = rounded_array[valid_mask]
    cm = confusion_matrix(
    valid_actual,
    valid_predicted,
    labels=orientations
    )
    print("Confusion Matrix:")
    print(cm)

    print("Training inputs:", normalized_train.shape)
    print("Testing inputs:", normalized_test.shape)
    print("Training labels:", train_stims["orientation"].shape)
    print("Testing labels:", test_stims["orientation"].shape)

    X_train = torch.tensor(normalized_train, dtype=torch.float32)
    X_test = torch.tensor(normalized_test,dtype=torch.float32)



if __name__ ==  "__main__":
    main()