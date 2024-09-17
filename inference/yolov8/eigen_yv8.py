import os
import time
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image


def get_unique_results_dir(base_dir="results", name_dir="run"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    n = 0
    while True:
        results_dir = os.path.join(base_dir, f"{name_dir}_{n}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            return results_dir
        n += 1


def infer_image(weights_path, image_path, base_dir):

    # Load the model
    model = YOLO(weights_path)
    save_name_img = weights_path.split("/")[-1].split(".pt")[0]
    # Create unique results directory
    results_dir = get_unique_results_dir(base_dir=base_dir, name_dir=save_name_img)

    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    norm_img_path = os.path.join(results_dir, f"{save_name_img}_base.jpg")
    cv2.imwrite(norm_img_path, img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Measure inference time
    start_time = time.time()
    results = model(img_rgb)
    end_time = time.time()
    inference_time = end_time - start_time

    with open(os.path.join(results_dir, f"{save_name_img}_results.txt"), "w") as file:
        # Scrivi attributi dell'oggetto results nel file
        for result in results:
            file.write(f"boxes: {str(result.boxes)}\n")
            file.write(f"keypoints: {str(result.keypoints)}\n")
            file.write(f"masks: {str(result.masks)}\n")
            file.write(f"names: {str(result.names)}\n")
            file.write(f"orig_shape: {str(result.orig_shape)}\n")
            file.write(f"speed: {str(result.speed)}\n")

    # Save inference results
    for result in results:
        result_img = result.plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_img)
        result_path = os.path.join(results_dir, f"{save_name_img}_inference.jpg")
        result_img.save(result_path)

    # If requested, calculate and save Eigen-CAM
    rgb_img = img.copy()
    img = np.float32(img) / 255
    target_layers = [model.model.model[-4]]
    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    # Save Eigen-CAM result
    eigencam_path = os.path.join(results_dir, f"{save_name_img}_eigen_cam.jpg")
    cv2.imwrite(eigencam_path, cam_image)

    # Save inference details to a text file
    with open(os.path.join(results_dir, f"{save_name_img}_time.txt"), "w") as f:
        f.write(f"end   Time: {end_time:.4f} seconds  - \n")
        f.write(f"start Time: {start_time:.4f} seconds  = \n")
        f.write(f"--------------------------------------------\n")
        f.write(f"Inference Time: {inference_time:.4f} seconds\n")

    print(f"Results saved in: {results_dir}")
    print(f"Inference Time: {inference_time:.4f} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script with Eigen-CAM and Time Measurement")
    parser.add_argument("-w", "--weights", type=str, help="Path to the model weights")
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    parser.add_argument("-r", "--result", default="results", type=str, help="Name of the result dir")

    args = parser.parse_args()

    infer_image(args.weights, args.image, args.result)

# python eigen_yv8.py -w weights/yolov8s_200e_64b_SGD_best.pt -i img/CAESAR/nave.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_32b_SGD_best.pt -i img/CAESAR/nave.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_AdamW_best.pt -i img/CAESAR/nave.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_SGD_best.pt -i img/CAESAR/nave.jpg ; python eigen_yv8.py -w weights/yolov8s_300e_64b_SGD_best.pt -i img/CAESAR/nave.jpg
# python eigen_yv8.py -w weights/yolov8s_200e_32b_SGD_best.pt -i img/CAESAR/nave_3_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_64b_SGD_best.pt -i img/CAESAR/nave_3_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_AdamW_best.pt -i img/CAESAR/nave_3_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_SGD_best.pt -i img/CAESAR/nave_3_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_300e_64b_SGD_best.pt -i img/CAESAR/nave_3_rumore.jpg
# python eigen_yv8.py -w weights/yolov8s_200e_32b_SGD_best.pt -i img/CAESAR/nave_grande.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_64b_SGD_best.pt -i img/CAESAR/nave_grande.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_AdamW_best.pt -i img/CAESAR/nave_grande.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_SGD_best.pt -i img/CAESAR/nave_grande.jpg ; python eigen_yv8.py -w weights/yolov8s_300e_64b_SGD_best.pt -i img/CAESAR/nave_grande.jpg
# python eigen_yv8.py -w weights/yolov8s_200e_32b_SGD_best.pt -i img/CAESAR/nave_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_64b_SGD_best.pt -i img/CAESAR/nave_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_AdamW_best.pt -i img/CAESAR/nave_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_200e_128b_SGD_best.pt -i img/CAESAR/nave_rumore.jpg ; python eigen_yv8.py -w weights/yolov8s_300e_64b_SGD_best.pt -i img/CAESAR/nave_rumore.jpg



# python eigen_yv8.py -w weights/yolov8s_200e_64b_SGD_best.pt -i img/OPTICAL/20240523_092652_29_2439_3B_Visual_clip_jpg.rf.ab11dcea908b5167aaba89fe35f46e35.jpg
