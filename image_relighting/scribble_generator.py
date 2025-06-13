import torch
import numpy as np
from PIL import Image, ImageFilter
import json
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import os
import cv2
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamPredictor, sam_model_registry
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Initialize models globally
def initialize_models():
    # GroundingDINO setup
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"
    dino_model = GroundingDINOModel(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    )

    # SAM setup
    SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(sam)
    
    return dino_model, sam_predictor

# Initialize models
dino_model, sam_predictor = initialize_models()

# Tool Definitions
class SceneAnalysisTool(BaseTool):
    name = "scene_analysis"
    description = "Analyzes scene using BLIP model to get scene description and understanding"
    
    def __init__(self, blip_processor, blip_model, device):
        self.processor = blip_processor
        self.model = blip_model
        self.device = device
        
    def _run(self, image: Image.Image) -> str:
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        scene_desc = self.model.generate(**inputs, max_length=50)
        return self.processor.decode(scene_desc[0], skip_special_tokens=True)

class CLIPAnalysisTool(BaseTool):
    name = "clip_analysis"
    description = "Analyzes images using CLIP model for feature extraction and similarity scoring"
    
    def __init__(self, clip_processor, clip_model, device):
        self.processor = clip_processor
        self.model = clip_model
        self.device = device
        self.lighting_prompts = [
            "well lit room", "dark room", "natural lighting", "artificial lighting",
            "warm lighting", "cold lighting", "balanced lighting", "dramatic lighting", "harsh shadows"
        ]
    
    def _run(self, image: Image.Image) -> Dict:
        # Get image features
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        features = image_features.detach().cpu().numpy()
        
        # Get CLIP scores
        inputs = self.processor(images=image, text=self.lighting_prompts, 
                              return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        scores = {prompt: float(score) for prompt, score in zip(self.lighting_prompts, probs[0])}
        
        return {
            "features": features,
            "scores": scores
        }

class SAMTool(BaseTool):
    name = "sam_segmentation"
    description = "Performs image segmentation using SAM model"
    
    def __init__(self, sam_predictor, dino_model):
        self.predictor = sam_predictor
        self.dino_model = dino_model
    
    def _run(self, image: Image.Image, text_prompts: List[str] = None) -> Dict:
        if text_prompts is None:
            text_prompts = ["window", "lamp", "light", "sun", "chandelier"]
            
        image_np = np.array(image)
        
        # DINO Detection
        detections = self.dino_model.predict_with_classes(
            image=image_np,
            classes=text_prompts,
            box_threshold=0.35,
            text_threshold=0.25,
        )
        
        result_list = []
        if len(detections.xyxy) == 0:
            return {"segments": []}
            
        # Set SAM image context
        self.predictor.set_image(image_np)
        
        for idx, (xyxy, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
            if idx >= len(detections.class_id) or detections.class_id[idx] is None:
                continue
                
            class_name = text_prompts[int(detections.class_id[idx])]
            x_min, y_min, x_max, y_max = map(int, xyxy)
            
            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image_np.shape[1], x_max + padding)
            y_max = min(image_np.shape[0], y_max + padding)
            
            try:
                masks, scores, _ = self.predictor.predict(
                    box=np.array([[x_min, y_min, x_max, y_max]]),
                    multimask_output=True
                )
                mask = masks[np.argmax(scores)]
                
                result_list.append({
                    "type": class_name,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "mask": mask,
                    "confidence": float(confidence)
                })
            except Exception as e:
                print(f"Error processing detection {idx}: {str(e)}")
                continue
                
        return {"segments": result_list}

class RelightingAgent:
    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # BLIP setup
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(self.device)
        
        # CLIP setup
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # Initialize tools
        self.scene_analysis_tool = SceneAnalysisTool(self.blip_processor, self.blip_model, self.device)
        self.clip_analysis_tool = CLIPAnalysisTool(self.clip_processor, self.clip_model, self.device)
        self.sam_tool = SAMTool(sam_predictor, dino_model)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Store previous iterations' results
        self.previous_results = []
        self.max_iterations = 3

    def generate_scribble_mask(self, image: Image.Image, recommendations: List[Dict], segments: Dict) -> Image.Image:
        """Generate scribble mask based on recommendations and segments"""
        img_np = np.array(image)
        mask = np.zeros_like(img_np)
        mask[mask==0] = 127  # Initialize with gray
        
        for segment in segments["segments"]:
            # Find matching recommendation
            matching_rec = next(
                (rec for rec in recommendations
                 if segment["type"] in rec["area"]),
                None
            )
            
            if matching_rec:
                bbox = segment["bbox"]
                x1, y1, x2, y2 = bbox
                
                if matching_rec["action"] == "on":
                    mask[y1:y2, x1:x2] = 255  # White for light on
                elif matching_rec["action"] == "off":
                    mask[y1:y2, x1:x2] = 0    # Black for light off
                # Gray (127) for unchanged areas
        
        # Convert to PIL Image and apply smoothing
        scribble = Image.fromarray(mask)
        scribble = scribble.filter(ImageFilter.GaussianBlur(radius=1.5))
        return scribble

    def create_agents(self):
        # Scene Analysis Agent
        scene_analyzer = Agent(
            role='Scene Analysis Expert',
            goal='Analyze scene content and lighting conditions',
            backstory='Expert in scene understanding and lighting analysis',
            tools=[self.scene_analysis_tool],
            verbose=True
        )
        
        # CLIP Analysis Agent
        clip_analyzer = Agent(
            role='Visual Analysis Expert',
            goal='Analyze visual features and lighting characteristics',
            backstory='Expert in visual feature analysis and lighting assessment',
            tools=[self.clip_analysis_tool],
            verbose=True
        )
        
        # Segmentation Agent
        segmentation_expert = Agent(
            role='Segmentation Expert',
            goal='Identify and segment important regions in the scene',
            backstory='Expert in image segmentation and region analysis',
            tools=[self.sam_tool],
            verbose=True
        )
        
        # Planning Agent (LLM)
        planner = Agent(
            role='Lighting Planning Expert',
            goal='Plan and coordinate lighting changes',
            backstory='Expert in lighting design and scene enhancement',
            llm=self.client,
            verbose=True
        )
        
        return scene_analyzer, clip_analyzer, segmentation_expert, planner

    def process_image(self, image: Image.Image, output_dir: str) -> Tuple[Image.Image, Dict]:
        """Process an image with the agentic system"""
        # Create agents
        scene_analyzer, clip_analyzer, segmentation_expert, planner = self.create_agents()
        
        best_result = None
        best_scribble = None
        
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            
            # Create tasks for this iteration
            tasks = [
                Task(
                    description="Analyze the scene content and lighting conditions",
                    agent=scene_analyzer,
                    context={"image": image}
                ),
                Task(
                    description="Analyze visual features and lighting characteristics",
                    agent=clip_analyzer,
                    context={"image": image}
                ),
                Task(
                    description="Identify and segment important regions",
                    agent=segmentation_expert,
                    context={"image": image}
                ),
                Task(
                    description="Plan lighting changes based on analysis",
                    agent=planner,
                    context={
                        "scene_analysis": self.previous_results[-1] if self.previous_results else None,
                        "iteration": iteration
                    }
                )
            ]
            
            # Create and run the crew
            crew = Crew(
                agents=[scene_analyzer, clip_analyzer, segmentation_expert, planner],
                tasks=tasks,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            
            # Generate scribble mask based on the results
            scribble_mask = self.generate_scribble_mask(image, result["recommendations"], result["segments"])
            
            # Save the scribble mask for this iteration
            iteration_output_dir = os.path.join(output_dir, f"iteration_{iteration + 1}")
            os.makedirs(iteration_output_dir, exist_ok=True)
            scribble_path = os.path.join(iteration_output_dir, "scribble.png")
            scribble_mask.save(scribble_path)
            
            # Run relighting and analyze results
            relit_image_path = f"image_relighting/scriblit/inference/data/output_{iteration + 1}.png"
            if os.path.exists(relit_image_path):
                relit_image = Image.open(relit_image_path)
                
                # Analyze the result
                analysis_tasks = [
                    Task(
                        description="Analyze the relighting result",
                        agent=clip_analyzer,
                        context={"original_image": image, "relit_image": relit_image}
                    ),
                    Task(
                        description="Evaluate the lighting changes",
                        agent=planner,
                        context={"analysis": result}
                    )
                ]
                
                analysis_crew = Crew(
                    agents=[clip_analyzer, planner],
                    tasks=analysis_tasks,
                    process=Process.sequential
                )
                
                feedback = analysis_crew.kickoff()
                self.previous_results.append(feedback)
                
                # Store the best result
                if not best_result or feedback.get('score', 0) > best_result.get('score', 0):
                    best_result = feedback
                    best_scribble = scribble_mask
            else:
                print(f"Relit image not found: {relit_image_path}")
                break
        
        # Save the best scribble mask
        if best_scribble:
            best_scribble.save(os.path.join(output_dir, "best_scribble.png"))
        
        return best_scribble, best_result

if __name__ == "__main__":
    # Initialize the agent
    agent = RelightingAgent()
    
    # Process all images in the input directory
    input_dir = "image_relighting/scriblit/dataset/data/image"
    output_dir = "image_relighting/scriblit/dataset/data/scribble"
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path)
            agent.process_image(image, output_dir)


