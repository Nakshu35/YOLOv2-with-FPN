import numpy as np
import torch
from tqdm import tqdm


def extract_box_wh(dataset, input_size=416):
    """
    Extract width and height of all boxes in dataset.
    Normalize according to input size.
    """
    box_wh = []

    for i in tqdm(range(len(dataset))):
        _, target = dataset[i]
        boxes = target["boxes"]

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            box_wh.append([w, h])

    box_wh = np.array(box_wh)
    box_wh = box_wh / input_size  # normalize

    return box_wh


def iou(box, clusters):
    """
    box: (2,)
    clusters: (k,2)
    """
    min_w = np.minimum(clusters[:, 0], box[0])
    min_h = np.minimum(clusters[:, 1], box[1])

    intersection = min_w * min_h
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    return intersection / (box_area + cluster_area - intersection)


def kmeans(boxes, k=5, max_iter=100):
    n = boxes.shape[0]

    clusters = boxes[np.random.choice(n, k, replace=False)]

    for _ in range(max_iter):
        distances = []
        for box in boxes:
            distances.append(1 - iou(box, clusters))

        distances = np.array(distances)
        nearest_clusters = np.argmin(distances, axis=1)

        new_clusters = []
        for i in range(k):
            cluster_boxes = boxes[nearest_clusters == i]
            if len(cluster_boxes) == 0:
                new_clusters.append(clusters[i])
            else:
                new_clusters.append(np.mean(cluster_boxes, axis=0))

        new_clusters = np.array(new_clusters)

        if np.all(clusters == new_clusters):
            break

        clusters = new_clusters

    return clusters