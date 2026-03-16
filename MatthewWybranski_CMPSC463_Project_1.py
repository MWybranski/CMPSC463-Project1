# Matthew Wybranski - CMPSC 463 - Project 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load Dataset

data = pd.read_csv("rul_hrs.csv")

# Use first 10k rows
data = data.iloc[:10000]

# Get sensor columns
sensor_cols = [c for c in data.columns if "sensor" in c]

# Rul -> Condition Categories

rul = data["rul"]

Q10 = rul.quantile(0.10)
Q50 = rul.quantile(0.50)
Q90 = rul.quantile(0.90)

def rul_category(r):

    if r < Q10:
        return 0  # Extremely Low

    elif r < Q50:
        return 1  # Moderately Low

    elif r < Q90:
        return 2  # Moderately High

    else:
        return 3  # Extremely High

data["condition"] = data["rul"].apply(rul_category)

# Task 1 — Divide‑and‑Conquer Segmentation

def segment(signal, start, end, threshold, segments):

    seg = signal[start:end]

    if len(seg) < 20:
        segments.append((start, end))
        return

    variance = np.var(seg)

    if variance > threshold:

        mid = (start + end) // 2

        segment(signal, start, mid, threshold, segments)
        segment(signal, mid, end, threshold, segments)

    else:
        segments.append((start, end))


def run_segmentation(signal, threshold=0.5):

    segments = []

    segment(signal, 0, len(signal), threshold, segments)

    return segments


def plot_segments(signal, segments, title):

    plt.figure(figsize=(12,4))

    plt.plot(signal)

    for s,e in segments:
        plt.axvline(s, color="red", alpha=0.3)

    plt.title(title)

    plt.show()

# Select 10 Sensors

selected_sensors = random.sample(sensor_cols, 10)

print("\nSelected sensors:", selected_sensors)

# Segmentation

print("\nTASK 1: Divide‑and‑Conquer Segmentation")

segmentation_scores = {}

for s in selected_sensors:

    signal = data[s].values

    segments = run_segmentation(signal, threshold=0.5)

    complexity = len(segments)

    segmentation_scores[s] = complexity

    print(s, "segments:", complexity)

    plot_segments(signal, segments, s)


# Task 2 — Recursive Clustering (Divide‑and‑Conquer Clustering of Segments)

def distance_idx(i, j):
    """Distance between two rows using sensor values"""
    return np.linalg.norm(X[i] - X[j])


def farthest_pair(indices, sample_size=200):
    """
    Approximate farthest pair using a random subset
    prevents O(n^2) explosion
    """

    if len(indices) > sample_size:
        sample = random.sample(indices, sample_size)
    else:
        sample = indices

    max_d = -1
    pair = (sample[0], sample[1])

    for i in range(len(sample)):
        for j in range(i+1, len(sample)):

            d = distance_idx(sample[i], sample[j])

            if d > max_d:
                max_d = d
                pair = (sample[i], sample[j])

    return pair


def split_cluster(indices):

    i, j = farthest_pair(indices)

    c1 = []
    c2 = []

    for idx in indices:

        if distance_idx(idx, i) < distance_idx(idx, j):
            c1.append(idx)
        else:
            c2.append(idx)

    return c1, c2


def recursive_cluster(indices, k):

    clusters = [indices]

    while len(clusters) < k:

        cluster = clusters.pop(0)

        c1, c2 = split_cluster(cluster)

        clusters.append(c1)
        clusters.append(c2)

    return clusters


# Run Clustering

print("\nTASK 2: Divide‑and‑Conquer Clustering of Segments")

# Use subset for faster runtime (still valid for assignment)
X = data[sensor_cols].values[:2000]

indices = list(range(len(X)))

clusters = recursive_cluster(indices, 4)

for i, cluster in enumerate(clusters):

    classes = data.iloc[cluster]["condition"]

    majority = classes.value_counts().idxmax()

    print(
        "Cluster", i,
        "size:", len(cluster),
        "majority class:", majority
    )


# Task 3 — Maximum Subarray (Kadane)

def kadane(arr):

    max_sum = arr[0]
    current = arr[0]

    start = end = temp = 0

    for i in range(1,len(arr)):

        if current + arr[i] < arr[i]:
            current = arr[i]
            temp = i
        else:
            current += arr[i]

        if current > max_sum:

            max_sum = current
            start = temp
            end = i

    return start,end,max_sum


print("\nTask 3 — Maximum Subarray (Kadane)")

sensor_results = {}

for s in sensor_cols:

    signal = data[s].values

    d = np.abs(np.diff(signal))

    x = d - np.mean(d)

    start,end,val = kadane(x)

    rul_slice = data.iloc[start:end]["condition"]

    dominant = rul_slice.value_counts().idxmax()

    sensor_results[s] = (start,end,val,dominant)

    print(s,
          "interval:",start,"-",end,
          "deviation:",round(val,3),
          "dominant class:",dominant)

# Summary

print("\nSegmentation Complexity")

for s,v in segmentation_scores.items():
    print(s,"complexity:",v)

print("\nAnalysis complete.")