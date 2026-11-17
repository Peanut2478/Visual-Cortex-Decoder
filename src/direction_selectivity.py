import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path 
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import prepare_data,normalize_data

dataset, timestamps, dff, stim_table = load_data()
X_train_full, X_test, y_train_full, y_test = prepare_data(
    dff, stim_table
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)
X_train, X_val, X_test = normalize_data(
    X_train, X_val, X_test
)
directions = np.arange(0,360,45)
labels = np.asarray(y_train)
mean_responses = np.array([
    np.mean(X_train[labels==direction],axis=0)
    for direction in directions 
])
responses =np.maximum(mean_responses,0)
pref_idx = np.argmax(responses,axis=0)
opp_idx = (pref_idx +4) %8
neuron_idx = np.arange(responses.shape[1])
pref_resp = responses[
    pref_idx,neuron_idx
]
opp_resp = responses[
    opp_idx,neuron_idx
]

#calC DSI 
denominator = pref_resp +opp_resp
dsi = np.divide(
    pref_resp - opp_resp,
    denominator,
    out=np.full_like(denominator,np.nan),
    where=denominator > 1e-8
)
valid_dsi  = dsi[np.isfinite(dsi)]

print("\nDirection Selectivity Analysis")
print("-" * 40)
print(f"Total neurons: {len(dsi)}")
print(f"Valid DSI neurons: {len(valid_dsi)}")

if len(valid_dsi) > 0:
    print(f"Mean DSI: {np.mean(valid_dsi):.4f}")
    print(f"Median DSI: {np.median(valid_dsi):.4f}")
    print(f"High selectivity (DSI > 0.5): "
          f"{np.sum(valid_dsi > 0.5)}")
    print(f"Low selectivity (DSI < 0.2): "
          f"{np.sum(valid_dsi < 0.2)}")

