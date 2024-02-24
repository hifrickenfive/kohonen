import cv2
import numpy as np
import time


def av_gradient_mag(image_path):
    """_summary_

    credit: https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/

    Args:
        image_path (_type_): _description_
    """
    image = cv2.imread(image_path)
    image_lab = image

    # Sobel filter gets gradients from img. Do for cardinal dir height width
    gX = cv2.Sobel(image_lab, cv2.CV_64F, dx=1, dy=0, ksize=3)  # img shape
    gY = cv2.Sobel(image_lab, cv2.CV_64F, dx=0, dy=1, ksize=3)

    # Plot gradient img
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    cv2.imshow("Sobel Combined", combined)
    # cv2.waitKey(0)

    # Eval gradient magnitude. Single value for perf
    gradient_magnitude = np.sqrt(gX**2 + gY**2)
    avg_gradient_magnitude = np.mean(gradient_magnitude)

    print(f"{avg_gradient_magnitude:1f}")


start_time = time.time()
av_gradient_mag("exp\\plot_of_initial_grid_2024-02-23_23-42-59.png")  # Worst bc rand
elapsed_time = time.time() - start_time
print(
    f"{elapsed_time:.2f}s to eval gradient mag"
)  # 0.11s / im * 1000 im for 1000 iter = 110s / 60s = 1.83 min

av_gradient_mag("exp\\plot_of_trained_grid_2024-02-23_23-42-48.png")  # Expect bad
av_gradient_mag("exp\\plot_of_trained_grid_2024-02-23_23-43-52.png")  # Expect better
av_gradient_mag("customer_spec\\benchmark.png")  # Expect best
