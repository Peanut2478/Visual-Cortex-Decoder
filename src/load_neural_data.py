from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
manifest_path = os.path.join("data", "manifest.json")
boc = BrainObservatoryCache(
    manifest_file=manifest_path
)
experiment_id = 652094901
dataset = boc.get_ophys_experiment_data(
    ophys_experiment_id=experiment_id
)
print("Experiment loaded!")

cell_ids = dataset.get_cell_specimen_ids()
print("Recorded neurons:", len(cell_ids))
print("First five cell IDs:", cell_ids[:5])

timestamps, dff = dataset.get_dff_traces()
print("Timestam[s shape:", timestamps.shape)
print("Fluorescence shape:", dff.shape)

mask = timestamps <= 30
plt.figure()
plt.plot(timestamps[mask],dff[0][mask])
plt.xlabel("Time (s)")
plt.ylabel("ΔF/F")
plt.title("Neuron 0 — Calcium Activity")
plt.show()

stim_table = dataset.get_stimulus_table("drifting_gratings")
print(stim_table.head())
print("Columns:", stim_table.columns.tolist())
print("Total presentations:", len(stim_table))

orientations = np.sort(stim_table["orientation"].dropna().unique())
print("Unique orientations:", orientations)
print("Number of orientations:", len(orientations))

first_stim = stim_table.iloc[0]
start = int(first_stim["start"])
end = int(first_stim["end"])

first_stim_pres = dff[0][start:end]
mean_1 = np.mean(first_stim_pres)
print("Stimulus direction:",first_stim["orientation"])
#print("Stimulus direction", first_stim_pres)
print("Mean direc:", mean_1)

responses = []
directions = []
for index, stim in stim_table.iterrows():
    start = int(stim["start"])
    end = int(stim["end"])
    fluorescence = dff[0][start:end]
    mean = np.mean(fluorescence)
    if (pd.notna(stim["orientation"])):
        directions.append(stim["orientation"])
        responses.append(mean)
    
responses = np.array(responses)
directions = np.array(directions)

mean_responses = []
for direction in orientations:
    mask = directions == direction
    mean = np.mean(responses[mask])
    mean_responses.append(mean)
print("Directions:", orientations)
print("Mean responses:", mean_responses)

pref_dir = orientations[np.argmax(mean_responses)]
print("Preffered directon:", pref_dir)
plt.figure(figsize=(8, 5))
plt.plot(orientations, mean_responses, marker="o")
plt.xlabel("Grating direction (degrees)")
plt.ylabel("Mean ΔF/F")
plt.title("Neuron 0 - Direction Tuning")
plt.show()


std_responses = []
responses = np.asarray(responses)
directions = np.asarray(directions)

std_responses = []

for direction in orientations:
    mask = directions == direction
    std_dev = np.std(responses[mask])
    std_responses.append(std_dev)

print("Standard deviations:", std_responses)

plt.figure(figsize=(8, 5))
plt.errorbar(
    orientations,
    mean_responses,
    yerr=std_responses,
    fmt="o-",
    capsize=5
)
plt.xlabel("Grating Direction (degrees)")
plt.ylabel("Mean ΔF/F")
plt.title("Neuron 0 - Direction Tuning")
plt.savefig("figures/neuron_0_tuning.png", dpi=300, bbox_inches="tight")
plt.show()