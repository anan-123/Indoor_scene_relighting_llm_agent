import torch
import os

# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, invert
from chrislib.data_util import load_from_url, load_image

# import model loading and running the pipeline
from intrinsic.pipeline import load_models, run_pipeline 
# download the pretrained weights and return the model (may take a bit to download weights the first time)
from PIL import Image
import numpy as np

def generate_albedo(input_image_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    img = load_image(input_image_path)
    
    # Run the pipeline
    intrinsic_model = load_models('v2')
    result = run_pipeline(
        intrinsic_model,
        img,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Get albedo map
    albedo = view(result['hr_alb'])
    
    # Save the albedo map
    output_path = os.path.join(output_dir, os.path.basename(input_image_path))
    Image.fromarray(albedo).save(output_path)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    input_dir = "image_relighting/scriblit/dataset/data/image"
    output_dir = "image_relighting/scriblit/dataset/data/albedo"
    
    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            generate_albedo(input_path, output_dir)