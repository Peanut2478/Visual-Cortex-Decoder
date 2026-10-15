import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

def load_data(experiment_id=652094901):
    manifest_path = os.path.join("data", "manifest.json")
    boc = BrainObservatoryCache(
        manifest_file=manifest_path
    )
    dataset = boc.get_ophys_experiment_data(
        ophys_experiment_id=experiment_id
    )
    timestamps, dff = dataset.get_dff_traces()
    stim_table = dataset.get_stimulus_table(
        "drifting_gratings"
    )
    print("Experiment loaded!")
    print("Recorded neurons:", len(dff))

    return dataset, timestamps, dff, stim_table