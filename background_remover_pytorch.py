import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as transforms

def load_image(image_path):
    # Load an image from the specified path using OpenCV
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    target_size = (512, 512)
    # Resize the image to the target size using OpenCV
    resized_image = cv2.resize(image, target_size)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    return normalized_image

def generate_mask(image, model):
    # Convert the image to a PyTorch tensor and add a batch dimension
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
    image_tensor = image_tensor.to(torch.device)

    # Perform inference on the model to obtain the predicted mask
    with torch.no_grad():
        outputs = model(image_tensor)
        mask = torch.argmax(outputs, dim=1)
        # Convert the mask to a NumPy array and remove the batch dimension
        binary_mask = mask.squeeze().cpu().numpy().astype(np.uint8)

    return binary_mask

def remove_background(image, binary_mask):
    # Concatenate the original image with the binary mask along the channel axis
    image_with_alpha = np.concatenate((image, binary_mask[:, :, np.newaxis]), axis=-1)
    return image_with_alpha

def main():
    image_path = "path to image.png"
    # Load the image
    image = load_image(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained DeepLabv3 model with ResNet101 backbone
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    # Move the model to the specified device and set it to evaluation mode
    model = model.to(device).eval()

    # Generate a binary mask for the image using the model
    binary_mask = generate_mask(preprocessed_image, model)

    # Remove the background from the image using the binary mask
    image_with_transparent_bg = remove_background(image, binary_mask)

    output_path = "path to output_image.png"
    # Save the resulting image with a transparent background
    cv2.imwrite(output_path, image_with_transparent_bg)

if __name__ == "__main__":
    main()
