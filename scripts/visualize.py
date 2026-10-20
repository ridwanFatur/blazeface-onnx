import numpy as np
from PIL import Image, ImageDraw

def visualize(
    img, 
    detections,
    target_width,
    target_height,
    original_width,
    original_height,
):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    scale_x = original_width / target_width
    scale_y = original_height / target_height

    num_detections = len(detections)

    for i in range(num_detections):
        ymin = detections[i, 0] * target_height * scale_y
        xmin = detections[i, 1] * target_width * scale_x
        ymax = detections[i, 2] * target_height * scale_y
        xmax = detections[i, 3] * target_width * scale_x

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline="red",
            width=2,
        )

        for k in range(6):
            kp_x = detections[i, 4 + k * 2] * target_width * scale_x
            kp_y = detections[i, 4 + k * 2 + 1] * target_height * scale_y

            radius = 2

            draw.ellipse(
                [
                    kp_x - radius,
                    kp_y - radius,
                    kp_x + radius,
                    kp_y + radius,
                ],
                fill="blue",
            )

    return img       