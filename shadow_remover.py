import sys
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# Shadow remover
class ShadowModel(nn.Module):
    def __init__(self, num_iterations=2):
        super(ShadowModel, self).__init__()
        self.num_iterations = num_iterations
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(self.num_iterations):
            x = self.restore(x)
        return x

    def restore(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def load_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    rgba = np.array(img)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    return rgb, alpha

def save_image(rgb, alpha, output_path):
    rgba = np.dstack((rgb, alpha))
    img = Image.fromarray(rgba, 'RGBA')
    img.save(output_path)

def process_image(image_path, model, output_path):
    rgb, alpha = load_image(image_path)

    # Convert RGB to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(Image.fromarray(rgb)).unsqueeze(0)

    # Run through the model
    with torch.no_grad():
        out_tensor = model(img_tensor)

    # Convert tensor back to image
    processed_img = transforms.ToPILImage()(out_tensor.cpu().squeeze(0))
    processed_rgb = np.array(processed_img)

    # Save the processed image
    save_image(processed_rgb, alpha, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to remove shadows.')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', required=True, help='Output image path')
    
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Load the model
    model = ShadowModel()
    model.load_state_dict(torch.load('shadowRM2.pth', map_location=torch.device('cpu')))
    model.eval()

    # Process the image
    process_image(input_path, model, output_path)
