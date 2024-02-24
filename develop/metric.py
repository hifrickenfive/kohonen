import cv2
import numpy as np
import time


def metric_av_gradient_mag(image_path: str) -> float:
    """
    Create a metric to evaluate the average gradient magnitude of an image

    credit: https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/

    Args:
        image_path: str, path to image
    """
    image = cv2.imread(image_path)

    # Find image gradient in x and y direction
    gX = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)  # img shape
    gY = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)

    # Eval gradient magnitude
    gradient_magnitude = np.sqrt(gX**2 + gY**2)
    avg_gradient_magnitude = np.mean(gradient_magnitude)

    return avg_gradient_magnitude


start_time = time.time()

avg_gradient_magnitude = metric_av_gradient_mag(
    "exp\\plot_of_initial_grid_2024-02-23_23-42-59.png"
)  # Worst bc rand
print(f"{avg_gradient_magnitude:.3f}")

elapsed_time = time.time() - start_time
print(
    f"{elapsed_time:.2f}s to eval gradient mag"
)  # 0.11s / im * 1000 im for 1000 iter = 110s / 60s = 1.83 min

avg_gradient_magnitude = metric_av_gradient_mag(
    "exp\\plot_of_trained_grid_2024-02-23_23-42-48.png"
)  # Expect bad
print(f"{avg_gradient_magnitude:.3f}")

avg_gradient_magnitude = metric_av_gradient_mag(
    "exp\\plot_of_trained_grid_2024-02-23_23-43-52.png"
)  # Expect better
print(f"{avg_gradient_magnitude:.3f}")

avg_gradient_magnitude = metric_av_gradient_mag(
    "customer_spec\\benchmark.png"
)  # Expect best
print(f"{avg_gradient_magnitude:.3f}")
