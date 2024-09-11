import os
import time
import argparse
import cv2
from ultralytics import YOLOv10
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
    model = YOLOv10(weights_path)
    save_name_img = weights_path.split("/")[-1].split(".pt")[0]
    # Create unique results directory
    results_dir = get_unique_results_dir(base_dir=base_dir, name_dir=save_name_img)

    # Load the image
    image = Image.open(image_path)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_rgb = np.float32(image_np) / 255
    norm_img_path = os.path.join(results_dir, f"{save_name_img}_base.jpg")
    cv2.imwrite(norm_img_path, image_cv)

    # Measure inference time
    start_time = time.time()
    results = model(image_path)
    end_time = time.time()
    inference_time = end_time - start_time

    with open(os.path.join(results_dir, f"{save_name_img}_results.txt"), "w") as file:
        # Write attributes of the results object to the file
        for result in results:
            file.write(f"boxes: {str(result.boxes)}\n")
            file.write(f"keypoints: {str(result.keypoints)}\n")
            file.write(f"masks: {str(result.masks)}\n")
            file.write(f"names: {str(result.names)}\n")
            file.write(f"orig_shape: {str(result.orig_shape)}\n")
            file.write(f"speed: {str(result.speed)}\n")

    # Save inference results with bounding boxes
    for i, result in enumerate(results):
        result_img = result.plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_img)
        result_path = os.path.join(results_dir, f"{save_name_img}_inference.jpg")
        result_img.save(result_path)

    # Generate the CAM for the full image
    target_layer = model.model.model[-2]
    cam = EigenCAM(model, target_layers=[target_layer], task='od')
    grayscale_cam = cam(image_rgb)[0, :, :]
    cam_image = show_cam_on_image(image_rgb, grayscale_cam, use_rgb=True)
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # Save the EigenCAM result for the full image
    eigen_cam_path = os.path.join(results_dir, f"{save_name_img}_eigen_cam_full.jpg")
    cv2.imwrite(eigen_cam_path, cam_image_bgr)

    # Save inference details to a text file
    with open(os.path.join(results_dir, f"{save_name_img}_time.txt"), "w") as f:
        f.write(f"end   Time: {end_time:.4f} seconds  - \n")
        f.write(f"start Time: {start_time:.4f} seconds  = \n")
        f.write(f"--------------------------------------------\n")
        f.write(f"Inference Time: {inference_time:.4f} seconds\n")

    print(f"Results saved in: {results_dir}")
    print(f"Inference Time: {inference_time:.4f} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv10 Inference Script with Eigen-CAM and Time Measurement")
    parser.add_argument("-w", "--weights", type=str, help="Path to the model weights")
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    parser.add_argument("-r", "--result", default="results", type=str, help="Name of the result dir")

    args = parser.parse_args()

    infer_image(args.weights, args.image, args.result)

# python eigencam_v10.py -w ../weights/yolov10s_200e_32b_SGD_best.pt -i ../img/CAESAR/nave_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_64b_SGD_best.pt -i ../img/CAESAR/nave_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_SGD_best.pt -i ../img/CAESAR/nave_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_AdamW_best.pt -i ../img/CAESAR/nave_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_300e_64b_SGD_best.pt -i ../img/CAESAR/nave_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_800e_64b_SGD_best.pt -i ../img/CAESAR/nave_rumore.jpg

# python eigencam_v10.py -w ../weights/yolov10s_200e_32b_SGD_best.pt -i ../img/CAESAR/nave.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_64b_SGD_best.pt -i ../img/CAESAR/nave.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_SGD_best.pt -i ../img/CAESAR/nave.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_AdamW_best.pt -i ../img/CAESAR/nave.jpg ; python eigencam_v10.py -w ../weights/yolov10s_300e_64b_SGD_best.pt -i ../img/CAESAR/nave.jpg ; python eigencam_v10.py -w ../weights/yolov10s_800e_64b_SGD_best.pt -i ../img/CAESAR/nave.jpg

# python eigencam_v10.py -w ../weights/yolov10s_200e_32b_SGD_best.pt -i ../img/CAESAR/nave_3_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_64b_SGD_best.pt -i ../img/CAESAR/nave_3_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_SGD_best.pt -i ../img/CAESAR/nave_3_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_AdamW_best.pt -i ../img/CAESAR/nave_3_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_300e_64b_SGD_best.pt -i ../img/CAESAR/nave_3_rumore.jpg ; python eigencam_v10.py -w ../weights/yolov10s_800e_64b_SGD_best.pt -i ../img/CAESAR/nave_3_rumore.jpg

# python eigencam_v10.py -w ../weights/yolov10s_200e_32b_SGD_best.pt -i ../img/CAESAR/nave_grande.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_64b_SGD_best.pt -i ../img/CAESAR/nave_grande.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_SGD_best.pt -i ../img/CAESAR/nave_grande.jpg ; python eigencam_v10.py -w ../weights/yolov10s_200e_128b_AdamW_best.pt -i ../img/CAESAR/nave_grande.jpg ; python eigencam_v10.py -w ../weights/yolov10s_300e_64b_SGD_best.pt -i ../img/CAESAR/nave_grande.jpg ; python eigencam_v10.py -w ../weights/yolov10s_800e_64b_SGD_best.pt -i ../img/CAESAR/nave_grande.jpg
