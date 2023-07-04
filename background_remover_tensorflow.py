import cv2
import numpy as np
import tensorflow as tf

def load_image(image_path):
    # Load an image from the specified path
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    target_size = (512, 512)
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    return normalized_image

def generate_mask(image, model):
    # Convert the image to a tensor and add an extra dimension for batch size
    input_tensor = tf.convert_to_tensor(image[np.newaxis, ...])
    # Use the model to predict the mask for the image
    mask = model.predict(input_tensor)
    # Get the index of the highest predicted class for each pixel
    mask = mask.argmax(axis=-1)
    # Convert the mask to a binary mask by thresholding
    binary_mask = (mask > 0).astype(np.uint8)
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

    # Load a pre-trained DenseNet121 model
    model = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=(512, 512, 3))

    # Generate a binary mask for the image using the model
    binary_mask = generate_mask(preprocessed_image, model)

    # Remove the background from the image using the binary mask
    image_with_transparent_bg = remove_background(image, binary_mask)

    output_path = "path to output_image.png"
    # Save the resulting image with a transparent background
    cv2.imwrite(output_path, image_with_transparent_bg)

if __name__ == "__main__":
    main()
