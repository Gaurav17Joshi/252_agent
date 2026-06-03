import cv2
import numpy as np

def annotate_bbox_from_mask(
    image_path: str,
    mask_path: str,
    output_path: str,
    box_color=(0, 0, 255),  # Red in BGR
    box_thickness=3,
):
    """
    Draw a bounding box on an image using the foreground region from a mask.

    Supports:
    - RGB masks
    - RGBA masks (4th channel can be all zeros)

    Foreground definition:
    - Any non-zero pixel in the first 3 channels (RGB/BGR)

    Args:
        image_path: Path to input image (.png)
        mask_path: Path to mask image (.png)
        output_path: Path to save annotated image
        box_color: Bounding box color in BGR
        box_thickness: Rectangle thickness
    """

    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Read mask with all channels preserved
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    # Handle grayscale masks
    if len(mask.shape) == 2:
        foreground = mask > 0

    else:
        # Use only first 3 channels (ignore alpha if present)
        # rgb_channels = mask[..., :3]
        # foreground = np.any(rgb_channels != 0, axis=-1)
        foreground = np.any(mask[..., 3:4] != 0, axis=-1)

    if not np.any(foreground):
        raise ValueError("Mask contains no foreground pixels.")

    # Get bounding box coordinates
    ys, xs = np.where(foreground)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    # Draw rectangle
    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x_min, y_min),
        (x_max, y_max),
        box_color,
        box_thickness,
    )

    # Save output
    aspect_ratio = annotated.shape[1] / annotated.shape[0]
    if aspect_ratio > 1:
        new_width = 256
        new_height = int(256 / aspect_ratio)
    else:
        new_height = 256
        new_width = int(256 * aspect_ratio)
    annotated = cv2.resize(annotated, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, annotated)

    print(f"Saved annotated image to: {output_path}")
    print(f"Bounding box: ({x_min}, {y_min}) -> ({x_max}, {y_max})")

# Example usage
annotate_bbox_from_mask(
    image_path="/home/keshav06/sam3d/sam-3d-objects/notebook/images/shutterstock_modern_colorful_Interior_2620125197/image.png",
    mask_path="/home/keshav06/sam3d/sam-3d-objects/notebook/images/shutterstock_modern_colorful_Interior_2620125197/18.png",
    output_path="annotated.png",
)