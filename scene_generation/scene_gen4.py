import os
import re
import json
import base64
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import gc

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai import BaseLLM

import torch
from diffusers import FluxPipeline
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interior_design_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class SystemConfig:
    # Model paths
    flux_model_path: str = os.getenv("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    lora_flux_path: str = os.getenv("LORA_FLUX_PATH", "SedatAl/Interior-Flux-Lora")
    qwen_model_path: str = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Generation settings
    image_width: int = int(os.getenv("IMAGE_WIDTH", "720"))
    image_height: int = int(os.getenv("IMAGE_HEIGHT", "720"))
    num_inference_steps: int = int(os.getenv("INFERENCE_STEPS", "28"))
    guidance_scale: float = float(os.getenv("GUIDANCE_SCALE", "3.5"))
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_8bit: bool = os.getenv("USE_8BIT", "false").lower() == "true"
    enable_cpu_offload: bool = os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "5"))
    
    # Directories - FIXED: Use absolute paths for Gradio
    output_dir: Path = Path(os.path.abspath(os.getenv("OUTPUT_DIR", "./generated_images")))
    final_dir: Path = Path(os.path.abspath(os.getenv("FINAL_DIR", "./final_designs")))
    cache_dir: Path = Path(os.path.abspath(os.getenv("CACHE_DIR", "./../hf_cache")))

config = SystemConfig()

# Ensure directories exist
config.output_dir.mkdir(parents=True, exist_ok=True)
config.final_dir.mkdir(parents=True, exist_ok=True)
config.cache_dir.mkdir(parents=True, exist_ok=True)


class QwenVisionLLM(BaseLLM):
    """Custom LLM wrapper for Qwen2.5-VL with vision capabilities"""
    
    def __init__(self, model_path: str = None):
        # Initialize with dummy values for parent class
        super().__init__(
            model="qwen-vision",
            temperature=0.7,
        )
        self.model_path = model_path or config.qwen_model_path
        self.model = None
        self.processor = None
        self._loaded = False
        
    def load_model(self):
        """Load the Qwen model lazily"""
        if self._loaded:
            return
            
        logger.info("Loading Qwen2.5-VL model...")
        try:
            # Configure quantization if needed
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto" if config.device == "cuda" else "cpu"
            }
            
            if config.use_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                # cache_dir=str(config.cache_dir),
                **model_kwargs
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            self._loaded = True
            logger.info("Qwen2.5-VL loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
            raise
    
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using Qwen2.5-VL"""
        if not self._loaded:
            self.load_model()
        
        try:
            # Handle image if provided in kwargs
            image_path = kwargs.get('image_path', None)
            
            if image_path and os.path.exists(image_path):
                # Process with image
                if isinstance(messages, str):
                    messages = [
                        {"role": "system", "content": "You are a helpful interior design assistant."},
                        {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": messages},
                        ],
                    }
                    ]
                else:
                    # Convert messages to include image
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            # {"type": "text", "text": messages[-1]["content"]},
                        ],
                    })
            else:
                if isinstance(messages, str):
                    messages = [
                        {"role": "system", "content": "You are a helpful interior design assistant."},
                        {"role": "user", "content": messages}
                    ]
            
            # Prepare the messages for the model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process with vision info if image is present
            if image_path and os.path.exists(image_path):
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
            
            # Generate
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode only the generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "qwen_vision"


class ModelManager:
    """Manages model loading and memory optimization"""
    
    def __init__(self):
        self.flux_pipe = None
        self._flux_loaded = False
    
    def load_flux(self) -> FluxPipeline:
        """Load FLUX model with optimizations"""
        if self._flux_loaded and self.flux_pipe is not None:
            return self.flux_pipe
        
        logger.info("Loading FLUX.1-dev model...")
        try:
            # Load model
            self.flux_pipe = FluxPipeline.from_pretrained(
                config.flux_model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                # cache_dir=str(config.cache_dir)
            )
            
            # Load LoRA if available
            if config.lora_flux_path:
                logger.info(f"Loading LoRA weights from {config.lora_flux_path}")
                self.flux_pipe.load_lora_weights(
                    config.lora_flux_path,
                    # cache_dir=str(config.cache_dir)
                )
            
            # Move to device
            if config.device == "cuda":
                if config.enable_cpu_offload:
                    self.flux_pipe.enable_model_cpu_offload()
                else:
                    self.flux_pipe = self.flux_pipe.to(config.device)
                
                # Memory optimizations
                self.flux_pipe.vae.enable_slicing()
                self.flux_pipe.vae.enable_tiling()
            
            self._flux_loaded = True
            logger.info("FLUX model loaded successfully")
            return self.flux_pipe
            
        except Exception as e:
            logger.error(f"Failed to load FLUX model: {e}")
            raise
    
    def clear_memory(self):
        """Clear GPU memory"""
        if config.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


# Global instances
model_manager = ModelManager()
qwen_llm = QwenVisionLLM()


# Tool Input Schemas
class ImageGenerationInput(BaseModel):
    """Input schema for image generation"""
    prompt: str = Field(description="The detailed prompt for image generation")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class PromptEnhancementInput(BaseModel):
    """Input schema for prompt enhancement"""
    original_prompt: str = Field(description="The original user prompt")
    style_preferences: Optional[str] = Field(default="", description="Style preferences")


class FeedbackAnalysisInput(BaseModel):
    """Input schema for feedback analysis"""
    current_prompt: str = Field(description="The current prompt used")
    current_image_path: str = Field(description="Path to current generated image")
    user_feedback: str = Field(description="User feedback about the image")
    previous_prompts: Optional[List[str]] = Field(default=[], description="Previous prompts tried")


# Tools
class ImageGenerationTool(BaseTool):
    """Tool for generating interior design images using FLUX"""
    name: str = "generate_interior_image"
    description: str = "Generate high-quality interior design images using FLUX.1-dev"
    args_schema: type[BaseModel] = ImageGenerationInput
    
    def _run(self, prompt: str, negative_prompt: str = "", seed: Optional[int] = None) -> str:
        """Generate image and return result as JSON string"""
        try:
            # Load model
            pipe = model_manager.load_flux()
            
            # Set seed for reproducibility
            if seed is not None:
                generator = torch.Generator(device=config.device).manual_seed(seed)
            else:
                generator = torch.Generator(device=config.device)
                seed = generator.seed()
            
            enhanced_prompt = f"Professional interior design photograph: {prompt}. High quality."
            # Enhanced negative prompt for interior design
            enhanced_negative = f"low quality, blurry, distorted, amateur, ugly, deformed, cartoon, painting, sketch, {negative_prompt}"
            
            logger.info(f"Generating image with prompt: {prompt}")
            
            # Generate image
            with torch.inference_mode():
                result = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=enhanced_negative,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    width=config.image_width,
                    height=config.image_height,
                    generator=generator
                )
            
            image = result.images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interior_{timestamp}.png"
            filepath = config.output_dir / filename
            image.save(filepath, "PNG", quality=95)
            
            # Save metadata
            metadata = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "negative_prompt": enhanced_negative,
                "seed": seed,
                "timestamp": timestamp,
                "settings": {
                    "guidance_scale": config.guidance_scale,
                    "steps": config.num_inference_steps,
                    "size": f"{config.image_width}x{config.image_height}"
                }
            }
            
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Clear memory
            model_manager.clear_memory()
            
            logger.info(f"Image generated successfully: {filepath}")
            
            # Return as JSON string that CrewAI can parse
            result_dict = {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "seed": seed,
                "metadata": metadata
            }
            
            return json.dumps(result_dict)
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class PromptEnhancementTool(BaseTool):
    """Tool for enhancing prompts for interior design"""
    name: str = "enhance_prompt"
    description: str = "Enhance user prompts with interior design details"
    args_schema: type[BaseModel] = PromptEnhancementInput
    
    def _run(self, original_prompt: str, style_preferences: str = "") -> str:
        """Enhance the prompt with design details"""
        try:
            enhancement_prompt = f"""You are an expert interior designer. Enhance this prompt for AI image generation:

Original request: {original_prompt}
Style preferences: {style_preferences if style_preferences else "Not specified"}

Create a detailed prompt that includes:
1. Specific design style (e.g., Modern, Scandinavian, Industrial)
2. Color palette (3-5 specific colors)
3. Key furniture pieces and materials
4. Lighting description (natural and artificial)
5. Textures and materials (flooring, walls, fabrics)
6. Decorative elements and accessories
7. Mood and atmosphere
8. Camera angle and composition

Keep the prompt under 50 words but make it rich in visual details. Return ONLY the enhanced prompt, nothing else."""

            enhanced = qwen_llm.call(enhancement_prompt)
            
            # Clean up the response
            enhanced = enhanced.strip()
            if len(enhanced) > 500:
                enhanced = enhanced[:497] + "..."
            
            logger.info(f"Prompt enhanced successfully")
            
            return json.dumps({
                "success": True,
                "original_prompt": original_prompt,
                "enhanced_prompt": enhanced
            })
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "enhanced_prompt": original_prompt  # Fallback to original
            })


class FeedbackAnalysisTool(BaseTool):
    """Tool for analyzing feedback and rewriting prompts"""
    name: str = "analyze_feedback"
    description: str = "Analyze user feedback on generated image and create improved prompt"
    args_schema: type[BaseModel] = FeedbackAnalysisInput
    
    def _run(
        self, 
        current_prompt: str, 
        current_image_path: str,
        user_feedback: str,
        previous_prompts: List[str] = []
    ) -> str:
        """Analyze feedback with image and create new prompt"""
        try:
            analysis_prompt = f"""You are an expert interior designer analyzing user feedback on a generated design.

Current prompt: {current_prompt}
User feedback: {user_feedback}
Previous attempts: {len(previous_prompts)}

Looking at the generated image and considering the feedback:
1. Identify what the user wants to change
2. Determine what elements to keep
3. Understand their style preferences from the feedback
4. Try to incorporate the feedback at the start improved prompt
5. Maintain the successful elements from the current design

Create an improved prompt that:
- Addresses ALL the user's feedback points
- Maintains successful elements from the current design
- Adds specific details to realize the user's vision
- Stays under 50 words

Return ONLY the new improved prompt, nothing else."""

            # Use vision capability to analyze the current image
            if os.path.exists(current_image_path):
                logger.info(f"Analyzing feedback with image: {current_image_path}")
                improved_prompt = qwen_llm.call(
                    analysis_prompt,
                    image_path=current_image_path
                )
            else:
                # Fallback to text-only analysis
                improved_prompt = qwen_llm.call(analysis_prompt)
            
            # Clean up
            improved_prompt = improved_prompt.strip()
            if len(improved_prompt) > 500:
                improved_prompt = improved_prompt[:497] + "..."
            
            logger.info(f"Feedback analyzed and prompt improved")
            
            return json.dumps({
                "success": True,
                "original_prompt": current_prompt,
                "user_feedback": user_feedback,
                "improved_prompt": improved_prompt
            })
            
        except Exception as e:
            logger.error(f"Feedback analysis failed: {e}")
            # Simple fallback
            fallback_prompt = f"{current_prompt}. {user_feedback}"
            return json.dumps({
                "success": False,
                "error": str(e),
                "improved_prompt": fallback_prompt
            })


class SaveFinalImageTool(BaseTool):
    """Tool for saving final approved images"""
    name: str = "save_final_image"
    description: str = "Save the final approved image with metadata"
    
    def _run(self, source_path: str, design_name: str, tags: List[str] = []) -> str:
        """Save image to final collection"""
        try:
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source image not found: {source_path}")
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in design_name if c.isalnum() or c in "-_").lower()
            filename = f"{safe_name}_{timestamp}.png"
            destination = config.final_dir / filename
            
            # Copy image
            image = Image.open(source)
            image.save(destination, "PNG", quality=95)
            
            # Load original metadata if exists
            metadata_path = source.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    original_metadata = json.load(f)
            else:
                original_metadata = {}
            
            # Create comprehensive metadata
            final_metadata = {
                "design_name": design_name,
                "original_path": str(source),
                "final_path": str(destination),
                "saved_at": datetime.now().isoformat(),
                "tags": tags,
                "original_metadata": original_metadata
            }
            
            # Save metadata
            metadata_destination = destination.with_suffix('.json')
            with open(metadata_destination, 'w') as f:
                json.dump(final_metadata, f, indent=2)
            
            logger.info(f"Design saved successfully: {destination}")
            
            return json.dumps({
                "success": True,
                "final_path": str(destination),
                "metadata": final_metadata
            })
            
        except Exception as e:
            logger.error(f"Failed to save final design: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })


# Agent System
class InteriorDesignAgentSystem:
    def __init__(self):
        self.tools = {
            "enhance": PromptEnhancementTool(),
            "generate": ImageGenerationTool(),
            "feedback": FeedbackAnalysisTool(),
            "save": SaveFinalImageTool()
        }
        self.session_data = {
            "prompts": [],
            "images": [],
            "current_image": None,
            "current_prompt": None
        }
    
    def parse_tool_output(self, result: str) -> Dict:
        """Parse tool output from CrewAI response"""
        try:
            # First try to parse as JSON directly
            if isinstance(result, dict):
                return result
            
            if isinstance(result, str):
                # Look for JSON in the string
                json_pattern = r'\{[^{}]*\}'
                matches = re.findall(json_pattern, result, re.DOTALL)
                
                for match in reversed(matches):  # Start from the end
                    try:
                        parsed = json.loads(match)
                        if "success" in parsed or "filepath" in parsed:
                            return parsed
                    except:
                        continue
                
                # Try to find JSON in code blocks
                code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
                code_blocks = re.findall(code_block_pattern, result, re.DOTALL)
                
                for block in code_blocks:
                    try:
                        parsed = json.loads(block)
                        if "success" in parsed or "filepath" in parsed:
                            return parsed
                    except:
                        continue
            
            logger.error(f"Could not parse tool output: {result[:200]}...")
            return {"success": False, "error": "Failed to parse output"}
            
        except Exception as e:
            logger.error(f"Error parsing tool output: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_image_directly(self, prompt, negative_prompt="", seed=None):
        """Generate an image directly using the tool without CrewAI agent overhead"""
        # Call the tool directly
        result_json = self.tools["generate"]._run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed
        )
        
        # Parse the result
        result = json.loads(result_json)
        
        return result
    
    def generate_initial_design(self, user_prompt: str) -> Dict:
        """Generate the initial design from user prompt"""
        try:
            # Step 1: Enhance the prompt directly
            enhance_result_json = self.tools["enhance"]._run(
                original_prompt=user_prompt,
                style_preferences=""
            )
            enhance_result = json.loads(enhance_result_json)
            
            if not enhance_result.get("success"):
                # Fallback to original prompt if enhancement fails
                enhanced_prompt = user_prompt
                logger.warning(f"Prompt enhancement failed, using original: {enhance_result.get('error')}")
            else:
                enhanced_prompt = enhance_result.get("enhanced_prompt", user_prompt)
            
            # Step 2: Generate image directly
            generation_result = self.generate_image_directly(enhanced_prompt)
            
            if generation_result.get("success") and generation_result.get("filepath"):
                self.session_data["current_image"] = generation_result["filepath"]
                self.session_data["current_prompt"] = enhanced_prompt
                self.session_data["images"].append(generation_result["filepath"])
                self.session_data["prompts"].append(enhanced_prompt)
                
                return {
                    "success": True,
                    "image_path": generation_result["filepath"],
                    "prompt_used": enhanced_prompt,
                    "seed": generation_result.get("seed")
                }
            else:
                return {
                    "success": False,
                    "error": generation_result.get("error", "Unknown error in generation")
                }
                
        except Exception as e:
            logger.error(f"Initial generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def process_feedback(self, feedback: str) -> Dict:
        """Process user feedback and generate improved design"""
        if not self.session_data["current_image"]:
            return {"success": False, "error": "No current design to improve"}
        
        try:
            # Step 1: Analyze feedback directly
            feedback_result_json = self.tools["feedback"]._run(
                current_prompt=self.session_data["current_prompt"],
                current_image_path=self.session_data["current_image"],
                user_feedback=feedback,
                previous_prompts=self.session_data["prompts"]
            )
            feedback_result = json.loads(feedback_result_json)
            
            if not feedback_result.get("success"):
                # Fallback: simple concatenation
                improved_prompt = f"{self.session_data['current_prompt']}. {feedback}"
                logger.warning(f"Feedback analysis failed, using fallback: {feedback_result.get('error')}")
            else:
                improved_prompt = feedback_result.get("improved_prompt", self.session_data["current_prompt"])
            logger.info(f"User feedback: {feedback}")
            logger.info(f"Improved prompt: {improved_prompt}")
            # Step 2: Generate new image directly
            generation_result = self.generate_image_directly(improved_prompt)
            
            if generation_result.get("success") and generation_result.get("filepath"):
                self.session_data["current_image"] = generation_result["filepath"]
                self.session_data["current_prompt"] = improved_prompt
                self.session_data["images"].append(generation_result["filepath"])
                self.session_data["prompts"].append(improved_prompt)
                
                return {
                    "success": True,
                    "image_path": generation_result["filepath"],
                    "prompt_used": improved_prompt,
                    "iteration": len(self.session_data["images"])
                }
            else:
                return {
                    "success": False,
                    "error": generation_result.get("error", "Unknown error in generation")
                }
                
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def save_final_design(self, design_name: str, tags: List[str] = []) -> Dict:
        """Save the final approved design"""
        if not self.session_data["current_image"]:
            return {"success": False, "error": "No design to save"}
        
        try:
            # Use tool directly instead of CrewAI
            save_result_json = self.tools["save"]._run(
                source_path=self.session_data["current_image"],
                design_name=design_name,
                tags=tags
            )
            save_result = json.loads(save_result_json)
            
            if save_result.get("success"):
                return {
                    "success": True,
                    "final_path": save_result.get("final_path"),
                    "message": f"Design '{design_name}' saved successfully!"
                }
            else:
                return {
                    "success": False,
                    "error": save_result.get("error", "Failed to save design")
                }
                
        except Exception as e:
            logger.error(f"Save operation failed: {e}")
            return {"success": False, "error": str(e)}


# Gradio Interface
def create_gradio_interface():
    # Create system instance
    system = InteriorDesignAgentSystem()
    
    with gr.Blocks(title="Interior Design AI System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🏠 Interior Design AI System
        ### Create stunning interior designs with AI-powered generation and intelligent feedback
        
        **Workflow:**
        1. Describe your ideal interior design
        2. AI generates a photorealistic visualization
        3. Provide feedback to refine the design
        4. Save your perfect design
        """)
        
        # State management - FIXED: Use proper state
        session_state = gr.State(value={"system": system})
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                prompt_input = gr.Textbox(
                    label="Design Request",
                    placeholder="Describe your ideal interior... (e.g., 'Modern minimalist bedroom with warm lighting')",
                    lines=3
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🎨 Generate Design", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
                
                gr.Markdown("### 💭 Refine Your Design")
                feedback_input = gr.Textbox(
                    label="Your Feedback",
                    placeholder="What would you like to change? (e.g., 'Add more plants and warmer colors')",
                    lines=3
                )
                iterate_btn = gr.Button("🔄 Apply Changes", variant="secondary")
                
                gr.Markdown("### 💾 Save Final Design")
                with gr.Row():
                    design_name = gr.Textbox(
                        label="Design Name",
                        placeholder="my_dream_living_room",
                        scale=2
                    )
                    tags_input = gr.Textbox(
                        label="Tags (comma-separated)",
                        placeholder="modern, minimalist, bedroom",
                        scale=2
                    )
                save_btn = gr.Button("💾 Save Design", variant="primary")
                
                # Status section
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=2):
                # Output section
                output_image = gr.Image(
                    label="Generated Design",
                    height=600,
                    show_label=True,
                    show_download_button=True,
                    interactive=False
                )
                
                with gr.Accordion("📋 Design Details", open=False):
                    prompt_display = gr.Textbox(
                        label="Current Prompt",
                        interactive=False,
                        lines=3
                    )
                    iteration_display = gr.Number(
                        label="Iteration",
                        value=0,
                        interactive=False
                    )
        
        # Examples
        gr.Examples(
            examples=[
                ["Modern Scandinavian living room with large windows and natural materials"],
                ["Cozy bohemian bedroom with warm earth tones and lots of plants"],
                ["Minimalist Japanese-inspired bathroom with clean lines and zen atmosphere"],
                ["Industrial loft kitchen with exposed brick and stainless steel appliances"],
                ["Luxury Art Deco dining room with bold geometric patterns and gold accents"],
                ["Coastal cottage style bedroom with soft blues and whites, nautical elements"],
                ["Mid-century modern home office with walnut furniture and vintage accents"],
                ["Rustic farmhouse kitchen with reclaimed wood and vintage fixtures"]
            ],
            inputs=prompt_input
        )
        
        # FIXED: Event handlers with proper state management
        def generate_design(prompt, state):
            if not prompt:
                return None, "", 0, "❌ Please enter a design request", state
            
            # Get system from state
            current_system = state["system"]
            
            # Reset system for new design
            current_system.session_data = {
                "prompts": [],
                "images": [],
                "current_image": None,
                "current_prompt": None
            }
            
            try:
                result = current_system.generate_initial_design(prompt)
                
                if result.get("success"):
                    # Update state
                    state["system"] = current_system
                    
                    return (
                        result["image_path"],
                        result["prompt_used"],
                        1,
                        f"✅ Design generated successfully! (Seed: {result.get('seed', 'N/A')})",
                        state
                    )
                else:
                    return (
                        None,
                        "",
                        0,
                        f"❌ Error: {result.get('error', 'Unknown error')}",
                        state
                    )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return (
                    None,
                    "",
                    0,
                    f"❌ Error: {str(e)}",
                    state
                )
        
        def apply_feedback(feedback, state):
            if not feedback:
                return None, "", 0, "❌ Please provide feedback", state

            # Get system from state
            current_system = state["system"]
            
            if not current_system.session_data["current_image"]:
                return None, "", 0, "❌ Please generate an initial design first", state

            try:
                result = current_system.process_feedback(feedback)
                
                if result.get("success"):
                    # Update state
                    state["system"] = current_system
                    
                    return (
                        result["image_path"],
                        result["prompt_used"],
                        result["iteration"],
                        f"✅ Design updated! (Iteration {result['iteration']})",
                        state
                    )
                else:
                    return (
                        current_system.session_data["current_image"],
                        current_system.session_data["current_prompt"],
                        len(current_system.session_data["images"]),
                        f"❌ Error: {result.get('error', 'Unknown error')}",
                        state
                    )
            except Exception as e:
                logger.error(f"Feedback error: {e}")
                return (
                    current_system.session_data.get("current_image"),
                    current_system.session_data.get("current_prompt", ""),
                    len(current_system.session_data.get("images", [])),
                    f"❌ Error: {str(e)}",
                    state
                )

        def save_design(name, tags, state):
            if not name:
                return "❌ Please provide a design name", state
            
            # Get system from state
            current_system = state["system"]
            
            if not current_system.session_data["current_image"]:
                return "❌ No design to save", state

            # Parse tags
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
            
            try:
                result = current_system.save_final_design(name, tag_list)
                
                if result.get("success"):
                    return f"✅ {result['message']}\n📁 Saved to: {result['final_path']}", state
                else:
                    return f"❌ Error: {result.get('error', 'Failed to save')}", state
            except Exception as e:
                logger.error(f"Save error: {e}")
                return f"❌ Error: {str(e)}", state
        
        def clear_all():
            # Create new system instance
            new_system = InteriorDesignAgentSystem()
            new_state = {"system": new_system}
            return None, "", 0, "🗑️ Cleared", new_state, "", ""
        
        # FIXED: Connect events with proper state handling
        generate_btn.click(
            fn=generate_design,
            inputs=[prompt_input, session_state],
            outputs=[output_image, prompt_display, iteration_display, status_output, session_state]
        )
        
        iterate_btn.click(
            fn=apply_feedback,
            inputs=[feedback_input, session_state],
            outputs=[output_image, prompt_display, iteration_display, status_output, session_state]
        )

        save_btn.click(
            fn=save_design,
            inputs=[design_name, tags_input, session_state],
            outputs=[status_output, session_state]
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[output_image, prompt_display, iteration_display, status_output, session_state, feedback_input, prompt_input]
        )
    
    return interface


if __name__ == "__main__":
    # Load models on startup
    logger.info("Initializing Interior Design AI System...")
    
    # Preload models
    logger.info("Preloading models...")
    qwen_llm.load_model()
    
    # Launch interface with file serving - FIXED: Enable file serving
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[str(config.output_dir), str(config.final_dir)]  # CRITICAL: Allow access to image directories
    )