from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os

manifest_path = os.path.join("data", "manifest.json")
boc = BrainObservatoryCache(
    manifest_file=manifest_path
)

experiments = boc.get_ophys_experiments()
print(len(experiments))

grating_experiments = boc.get_ophys_experiments(
    stimuli=["drifting_gratings"]
)

print("Drifting grating experiments:", len(grating_experiments))

visp_experiments = [
    exp for exp in grating_experiments
    if exp["targeted_structure"] == "VISp"
]

print("VISp experiments:", len(visp_experiments))
first_exp = visp_experiments[0]

print("Experiment ID:", first_exp["id"])
print("Experiment details:", first_exp)

depths = [exp["imaging_depth"] for exp in visp_experiments]
unique_depths = sorted(set(depths))
print("Available imaging depths:", unique_depths)

shallow_experiments = [
    exp for exp in visp_experiments
    if 175 <= exp["imaging_depth"] <= 225
]
print("Shallow VISp experiments:", len(shallow_experiments))

for exp in shallow_experiments[:5]:
    print(exp["id"])
    print("Imagiung depth:", exp["imaging_depth"])
    print("cre_lin:", exp["cre_line"])

exp_id = 652094901
cells = boc.get_cell_specimens()
selected_exp = next(
    exp for exp in shallow_experiments
    if exp["id"] == exp_id
)

container_id = selected_exp["experiment_container_id"]

print("Experiment ID:", exp_id)
print("Container ID:", container_id)

exp_cells = [
    cell for cell in cells
    if cell["experiment_container_id"] == container_id
]

print("Number of cells:", len(exp_cells))