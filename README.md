# Background Remover using PyTorch

The `background_remover_pytorch.py` script demonstrates how to remove the background from an image using PyTorch and a pre-trained DeepLabv3 model with a ResNet101 backbone.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- PyTorch
- torchvision

## Usage

1. Install the required dependencies using `pip`:
    ```shell
    pip install opencv-python numpy torch torchvision


2. Place the image you want to process at the specified path: `path to image.png`.

3. Run the script:
    ```shell
    python background_remover_pytorch.py


4. The resulting image with a transparent background will be saved at the specified output path.

---

Please make sure to replace `path to image.png` with the actual path to your input image and `path to output_image.png` with the desired output path for the resulting image.


# Background Remover using TensorFlow

The `background_remover_tensorflow.py` script demonstrates how to remove the background from an image using TensorFlow and a pre-trained DenseNet121 model.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- TensorFlow

## Usage

1. Install the required dependencies using `pip`:
    ```shell
    pip install opencv-python numpy tensorflow


2. Place the image you want to process at the specified path: `path to image.png`.

3. Run the script:
    ```shell
    python background_remover_tensorflow.py


4. The resulting image with a transparent background will be saved at the specified output path.



