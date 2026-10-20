import numpy as np

def postprocess(output, min_score_thresh=0.5, min_suppression_threshold=0.3):
    detections = output[0]  # [896, 17]

    # Score filtering
    scores = detections[:, 16]
    mask = scores >= min_score_thresh

    detections = detections[mask]

    if len(detections) == 0:
        return np.empty((0, 17), dtype=np.float32)

    # Weighted NMS
    detections = detections.copy()
    remaining = np.argsort(detections[:, 16])[::-1]

    output_detections = []

    while len(remaining) > 0:
        detection = detections[remaining[0]]
        first_box = detection[:4]

        other_boxes = detections[remaining, :4]

        # IoU
        ymin = np.maximum(first_box[0], other_boxes[:, 0])
        xmin = np.maximum(first_box[1], other_boxes[:, 1])
        ymax = np.minimum(first_box[2], other_boxes[:, 2])
        xmax = np.minimum(first_box[3], other_boxes[:, 3])

        inter_w = np.maximum(xmax - xmin, 0)
        inter_h = np.maximum(ymax - ymin, 0)
        intersection = inter_w * inter_h

        area_a = (
            (first_box[2] - first_box[0])
            * (first_box[3] - first_box[1])
        )

        area_b = (
            (other_boxes[:, 2] - other_boxes[:, 0])
            * (other_boxes[:, 3] - other_boxes[:, 1])
        )

        union = area_a + area_b - intersection

        ious = intersection / (union + 1e-6)

        mask = ious > min_suppression_threshold

        overlapping = remaining[mask]
        remaining = remaining[~mask]

        weighted_detection = detection.copy()

        if len(overlapping) > 1:
            coordinates = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]

            total_score = scores.sum()

            weighted = (
                coordinates * scores
            ).sum(axis=0) / total_score

            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return np.stack(output_detections)