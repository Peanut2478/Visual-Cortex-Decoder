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

#vecotr based DSI 
angles = np.deg2rad(directions)
vector_x = np.sum(responses * np.cos(angles)[:,None],
axis=0)
vector_y = np.sum(responses * np.sin(angles)[:,None], axis=0)
vector_magnitude = np.sqrt(vector_x **2 + vector_y**2)
tot_resp = np.sum(responses, axis=0)
vector_dsi = np.divide(
    vector_magnitude,
    tot_resp,
    out=np.full_like(tot_resp,np.nan),
    where=tot_resp > 1e-8
)
valid_vector_dsi = vector_dsi[np.isfinite(vector_dsi)]

print("\nVector-Based DSI Analysis")
print("-" * 40)
print(f"Valid neurons: {len(valid_vector_dsi)}")
print(f"Mean DSI: {np.mean(valid_vector_dsi):.4f}")
print(f"Median DSI: {np.median(valid_vector_dsi):.4f}")
print(f"High DSI (>0.5): {np.sum(valid_vector_dsi > 0.5)}")
print(f"Low DSI (<0.2): {np.sum(valid_vector_dsi < 0.2)}")


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

