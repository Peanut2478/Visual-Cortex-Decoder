import matplotlib.pyplot as plt


def plot_tuning(orientations, means, stds, neuron_idx):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        orientations,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5
    )
    plt.xlabel("Grating Direction (degrees)")
    plt.ylabel("Mean ΔF/F")
    plt.title(f"Neuron {neuron_idx} - Direction Tuning")

    plt.show()
def plot_direction_errors(directions, errors):
    plt.figure(figsize=(9, 5))
    plt.bar(directions.astype(str), errors)
    plt.xlabel("Actual Stimulus Direction (degrees)")
    plt.ylabel("Mean Angular Error (degrees)")
    plt.title("Population Decoder Error by Stimulus Direction")
    plt.ylim(0, 180)
    plt.tight_layout()
    plt.savefig("results/figures/direction_errors.png", dpi=300)
    plt.show()