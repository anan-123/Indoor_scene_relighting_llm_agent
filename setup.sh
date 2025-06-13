# Initialize git submodules
git submodule init
git submodule update

# Install dependencies
pip install groundingdino-py
pip install opencv-python>=4.5.0 scipy>=1.7.0 scikit-image>=0.19.0
pip install torch transformers diffusers opencv-python pillow safetensors segment-anything
wget  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip install groundingdino-py
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
pip install geffnet
pip install Intrinsic/.
