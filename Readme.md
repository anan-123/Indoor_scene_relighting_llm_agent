# Indoor Scene Relighting with LLM Agent

This project implements an agentic system for indoor scene relighting using a combination of vision models and LLM feedback. The system analyzes scenes, generates scribbles, and iteratively improves relighting results through a feedback loop.

## Directory Structure

```
image_relighting/
├── scriblit/
│   ├── dataset/
│   │   └── data/
│   │       ├── image/          # Input images
│   │       ├── normal/         # Generated normal maps
│   │       ├── albedo/         # Generated albedo maps
│   │       └── scribble/       # Generated scribble masks
│   └── inference/
│       └── data/
│           └── output/         # Final relit images
```

## Setup

You can look at setup.sh, but here are the detailed steps:

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/yourusername/Indoor_scene_relighting_llm_agent.git
cd Indoor_scene_relighting_llm_agent
```

If you've already cloned without submodules, initialize them:
```bash
git submodule init
git submodule update
```

2. Install dependencies:
```bash
pip install torch torchvision transformers openai crewai langchain pydantic opencv-python segment-anything
```

3. Download required model checkpoints:
   - GroundingDINO: `groundingdino_swint_ogc.pth`
   - SAM: `sam_vit_h_4b8939.pth`
   Place these in the root directory.

4. Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```


## Pipeline Steps

### 1. Generate Normal Maps
```bash
python image_relighting/DSINE/inference.py -i image_relighting/scriblit/dataset/data/image -o image_relighting/scriblit/dataset/data/normal
```

### 2. Generate Albedo Maps
```bash
python image_relighting/albedo_generator.py
```

### 3. Generate Scribble Masks with LLM Agent
```bash
python image_relighting/scribble_generator.py
```
The agentic system will:
- Analyze scene content and lighting conditions
- Generate initial scribble masks
- Run relighting
- Analyze results using CLIP scores
- Iteratively improve through feedback
- Save results in `image_relighting/scriblit/dataset/data/scribble/`

### 4. Final Relighting
```bash
python image_relighting/scriblit/inference.py -n scribblelight_controlnet -data data
```

## Results and Outputs

### Final Results Location
- Final relit images: `image_relighting/scriblit/inference/data/output/`
- Best scribble masks: `image_relighting/scriblit/dataset/data/scribble/best_scribble.png`
- Iteration results: `image_relighting/scriblit/dataset/data/scribble/iteration_X/`

### Important Notes
1. The agentic system performs up to 3 iterations to find the best relighting result
2. Each iteration includes:
   - Scene analysis using BLIP
   - Visual feature analysis using CLIP
   - Region segmentation using SAM
   - LLM-based planning
   - Result evaluation and feedback
3. The system tracks the best result based on:
   - CLIP similarity scores
   - Lighting-specific metrics
   - Visual quality assessment