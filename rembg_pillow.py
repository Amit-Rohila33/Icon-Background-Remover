# Importing Required Modules
from rembg import remove
from PIL import Image
import os

# Store path of the image in the variable input_path
input_path = 'girlicon.png'  # Assuming the image is in the same directory as the script

# Store path of the output image in the variable output_path
output_path = 'girloutput.png'  # Output image path

# Processing the image
input = Image.open(input_path)

# Removing the background from the given Image
output = remove(input)

# Save the image in the given path
output.save(output_path)

# Print the absolute path of the output image
absolute_output_path = os.path.abspath(output_path)
print(f"Output image saved at: {absolute_output_path}")
