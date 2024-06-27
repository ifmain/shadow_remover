# Shadow Remover

This script removes shadows from images using a pre-trained convolutional neural network model.

## Usage
To use the shadow remover script, follow these steps:

1. **Prepare your image:**
    Make sure your input image is in the same directory or provide the full path to the image.

2. **Run the script:**

    Use the following command to run the script:
    ```sh
    python shadow_remover.py -i <input_image_path> -o <output_image_path>
    ```
    Replace `<input_image_path>` with the path to your input image and `<output_image_path>` with the desired path for the output image.

## Script Details
The script consists of the following main components:

1. **ShadowModel Class:**
    - A neural network model to remove shadows from images. The model contains several convolutional layers with ReLU activations.

2. **load_image Function:**
    - Loads an RGBA image and splits it into RGB and alpha channels.

3. **save_image Function:**
    - Combines the processed RGB image with the original alpha channel and saves it as an RGBA image.

4. **process_image Function:**
    - Processes the input image using the ShadowModel and saves the output image.

5. **Main Execution:**
    - The script uses argparse to handle command-line arguments, loads the pre-trained model, and processes the input image.

## Examples

| Before | After | Source |
| ------ | ----- | ------ |
| ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/1_b.webp) | ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/1_a.webp) | SD 3 |
| ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/2_b.png) | ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/2_a.png) | Steam Game Screenshots FNAF 9 Ruin  |
| ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/3_b.png) | ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/3_a.png) | Steam Game Screenshots FNAF 9 Ruin  |
| ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/4_b.png) | ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/4_a.png) | Steam Game Screenshots FNAF 9 Ruin  |
| ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/5_b.png) | ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/5_a.png) | Steam Game Screenshots FNAF 9 Ruin  |
| ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/6_b.png) | ![ex](https://raw.githubusercontent.com/ifmain/shadow_remover/main/ex/6_a.png) | Steam Game Screenshots FNAF 9 Ruin  |

## Prerequisites

Make sure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy

## Installation

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install the required packages:**
    You can install the required packages using pip:

    ```sh
    pip install torch torchvision pillow numpy
    ```

3. **Place the model file:**
    Ensure the pre-trained model file `shadowRM2.pth` is located in the same directory as the script.

## Model

The pre-trained model `shadowRM2.pth` should be located in the same directory as the script. This model is loaded and used to process the input image to remove shadows.
