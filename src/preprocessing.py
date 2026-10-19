import numpy as np

def prepare_data(dff, stim_table):
    valid_stims = stim_table[stim_table["orientation"].notna()]
    num_stims = len(valid_stims)
    split_idx = int(0.8 * num_stims)
    train_stims = valid_stims.iloc[:split_idx]
    test_stims = valid_stims.iloc[split_idx:]
    train_responses = []

    for _, stim in train_stims.iterrows():
        start = int(stim["start"])
        end = int(stim["end"])
        mean_responses = np.mean(dff[:, start:end], axis=1)
        train_responses.append(mean_responses)
    train_responses = np.array(train_responses)
    test_responses = []
    for _, stim in test_stims.iterrows():
        start = int(stim["start"])
        end = int(stim["end"])
        mean_responses = np.mean(dff[:, start:end], axis=1)
        test_responses.append(mean_responses)
    test_responses = np.asarray(test_responses)

    training_baselines = np.mean(train_responses, axis=0)
    centered_train = train_responses - training_baselines
    max_responses = np.max(centered_train, axis=0)
    safe_max = np.where(max_responses <= 0, 1, max_responses)
    normalized_train = centered_train / safe_max

    centered_test = test_responses - training_baselines
    normalized_test = centered_test / safe_max

    return normalized_train,normalized_test,train_stims["orientation"],test_stims['orientation']