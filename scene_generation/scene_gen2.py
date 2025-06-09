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
from crewai.tools import BaseTool as CrewAIBaseTool
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

"""
2025-06-03 22:24:44,341 - __main__ - ERROR - Failed to extract valid result: {"tool_output": "```json
{
  "file_path": "generated_interior_image.jpg",
  "metadata": {
    "tool": "generate_interior_image",
    "arguments": {
      "prompt": "Minimalist Japanese-inspired bathroom with clean lines and simplicity, featuring a large window for natural light, a walk-in shower with frosted glass for privacy, a modern low-profile toilet, and a compact vanity with storage, accentuated with soft grey and light blue accents.",
      "negative_prompt": null,
      "seed": 42
    }
  }
}
```"}"""

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
    num_inference_steps: int = int(os.getenv("INFERENCE_STEPS", "25"))
    guidance_scale: float = float(os.getenv("GUIDANCE_SCALE", "7.5"))
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_8bit: bool = os.getenv("USE_8BIT", "false").lower() == "true"
    enable_cpu_offload: bool = os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "5"))
    
    # Directories
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./generated_images"))
    final_dir: Path = Path(os.getenv("FINAL_DIR", "./final_designs"))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", "./../hf_cache"))

config = SystemConfig()

# Ensure directories exist
config.output_dir.mkdir(parents=True, exist_ok=True)
config.final_dir.mkdir(parents=True, exist_ok=True)
config.cache_dir.mkdir(parents=True, exist_ok=True)


class QwenLLM(BaseLLM):
    """Qwen2.5-VL"""
    
    def __init__(self, model: str = "Qwen/Qwen2.5-VL-3B-Instruct", temperature: float = 0.7):
        super().__init__(model=model, temperature=temperature)
        self.tokenizer = None
        self.processor = None
        self._loaded = False
        
    def load_model(self):
        """Load the Qwen model"""
        if self._loaded:
            return
            
        logger.info("Loading Qwen2.5-VL model...")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.qwen_model_path,
                torch_dtype="auto",
                device_map="auto" if config.device == "cuda" else "cpu"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path)
            self.processor = AutoProcessor.from_pretrained(config.qwen_model_path)
            
            self._loaded = True
            logger.info("Qwen2.5-VL loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
            # Fallback to a simple text generation
            self._loaded = False
    
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text using Qwen2.5-VL"""
        if not self._loaded:
            self.load_model()
        
        try:
            if isinstance(messages, str):
                messages = [
                    {"role": "system", "content": "You are a helpful interior design assistant."},
                    {"role": "user", "content": messages}
                ]
            
            text_prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = self.tokenizer(
                text_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            else:
                response = response[len(text_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None
    
    @property
    def _llm_type(self) -> str:
        return "qwen_vl"


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
            # Configure quantization if needed
            kwargs = {
                "torch_dtype": torch.bfloat16,
                # "variant": "bf16",
                "use_safetensors": True,
                "cache_dir": str(config.cache_dir)
            }
            
            if config.use_8bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
            
            # Load model
            self.flux_pipe = FluxPipeline.from_pretrained(
                config.flux_model_path,
                **kwargs
            )
            
            # Move to device
            if config.device == "cuda":
                if config.enable_cpu_offload:
                    self.flux_pipe.enable_model_cpu_offload()
                else:
                    self.flux_pipe = self.flux_pipe.to(config.device)
                
                # Memory optimizations
                self.flux_pipe.enable_vae_slicing()
                self.flux_pipe.enable_attention_slicing()
            
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


# Global model manager
model_manager = ModelManager()
qwen_llm = QwenLLM()
qwen_llm.load_model()



class ImageGenerationInput(BaseModel):
    """Input schema for image generation"""
    prompt: str = Field(description="The text prompt for image generation")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class PromptRewriteInput(BaseModel):
    """Input schema for prompt rewriting"""
    original_prompt: str = Field(description="The original prompt")
    feedback: str = Field(description="User feedback for improvements")
    previous_attempts: Optional[List[str]] = Field(default=[], description="Previous prompt attempts")
    current_image_path: Optional[str] = Field(default=None, description="Path to current image")


class ImageGenerationTool(BaseTool):
    """Production-ready image generation tool using FLUX.1-dev"""
    name: str = "generate_interior_image"
    description: str = "Generate high-quality interior design images using FLUX.1-dev"
    args_schema: type[BaseModel] = ImageGenerationInput
    
    def _run(self, prompt: str, negative_prompt: str = "", seed: Optional[int] = None) -> Dict:
        """Generate image from prompt"""
        try:
            # Load model if needed
            pipe = model_manager.load_flux()
            
            # Set seed for reproducibility
            if seed is not None:
                generator = torch.Generator(device=config.device).manual_seed(seed)
            else:
                generator = None
            
            # Enhanced prompt for interior design
            enhanced_prompt = f"Professional interior design photograph: {prompt}. High quality, architectural photography, well-lit, 8k resolution"
            
            # Add default negative prompt for better quality
            full_negative_prompt = f"low quality, blurry, distorted, amateur, {negative_prompt}"
            
            logger.info(f"Generating image with prompt: {enhanced_prompt[:100]}...")
            
            # Generate image
            with torch.inference_mode():
                result = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=full_negative_prompt,
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
                "negative_prompt": full_negative_prompt,
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
            
            # Clear some memory
            model_manager.clear_memory()
            
            logger.info(f"Image generated successfully: {filepath}")
            
            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "prompt_used": enhanced_prompt,
                "seed": seed,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class PromptRewriteTool(BaseTool):
    """Production-ready prompt rewriting using Qwen2.5-VL"""
    name: str = "rewrite_prompt"
    description: str = "Intelligently rewrite prompts based on user feedback"
    args_schema: type[BaseModel] = PromptRewriteInput
    
    def _run(
        self, 
        original_prompt: str, 
        feedback: str, 
        previous_attempts: List[str] = [],
        current_image_path: Optional[str] = None
    ) -> str:
        """Rewrite prompt based on feedback"""
        try:
            # Load model components
            model, processor, tokenizer = model_manager.load_qwen()
            
            # Prepare the conversation
            system_prompt = """You are an expert interior designer specializing in creating detailed prompts for AI image generation. 
            Your task is to rewrite prompts based on user feedback while maintaining design coherence and adding rich details."""
            
            user_message = f"""Original prompt: {original_prompt}

User feedback: {feedback}

Previous attempts: {', '.join(previous_attempts[-3:]) if previous_attempts else 'None'}

Please rewrite the prompt to incorporate the user's feedback. Include:
1. Specific design style (e.g., Scandinavian, Industrial, Bohemian)
2. Color palette with specific colors
3. Key furniture pieces and their arrangement
4. Lighting details (natural/artificial, mood)
5. Materials and textures
6. Decorative elements and accessories
7. Spatial layout and composition
8. Atmosphere and mood

Make the prompt detailed but concise, focusing on visual elements."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Process with model
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # If current image is provided, we could analyze it (future enhancement)
            inputs = processor(
                text=[text_prompt], 
                return_tensors="pt"
            ).to(model.device)
            
            logger.info("Generating rewritten prompt...")
            
            # Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the rewritten prompt
            if "assistant" in response:
                rewritten_prompt = response.split("assistant")[-1].strip()
            else:
                # Fallback to last part after the user message
                rewritten_prompt = response.split(user_message)[-1].strip()
            
            # Clean up the prompt
            rewritten_prompt = rewritten_prompt.replace("\n", " ").strip()
            
            # Ensure it's not too long
            if len(rewritten_prompt) > 500:
                rewritten_prompt = rewritten_prompt[:497] + "..."
            
            logger.info(f"Prompt rewritten successfully: {rewritten_prompt[:100]}...")
            
            # Clear memory
            model_manager.clear_memory()
            
            return rewritten_prompt
            
        except Exception as e:
            logger.error(f"Prompt rewriting failed: {e}")
            # Fallback: simple concatenation
            return f"{original_prompt}, {feedback}"


class SaveImageTool(BaseTool):
    """Tool for saving final approved images"""
    name: str = "save_final_image"
    description: str = "Save the final approved image with metadata"
    
    def _run(self, source_path: str, design_name: str, tags: List[str] = []) -> Dict:
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
                "original_metadata": original_metadata,
                "file_info": {
                    "size_bytes": destination.stat().st_size,
                    "dimensions": f"{image.width}x{image.height}",
                    "format": image.format
                }
            }
            
            # Save metadata
            metadata_destination = destination.with_suffix('.json')
            with open(metadata_destination, 'w') as f:
                json.dump(final_metadata, f, indent=2)
            
            # Create thumbnail
            thumbnail_size = (256, 256)
            thumbnail = image.copy()
            thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            thumbnail_path = destination.with_stem(f"{destination.stem}_thumb")
            thumbnail.save(thumbnail_path, "PNG")
            
            logger.info(f"Design saved successfully: {destination}")
            
            return {
                "success": True,
                "final_path": str(destination),
                "thumbnail_path": str(thumbnail_path),
                "metadata": final_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to save final design: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Agent Definitions
class InteriorDesignAgents:
    def __init__(self):
        self.image_tool = ImageGenerationTool()
        self.rewrite_tool = PromptRewriteTool()
        self.save_tool = SaveImageTool()
    
    def prompt_analyst_agent(self) -> Agent:
        return Agent(
            role='Senior Interior Design Consultant',
            goal='Transform vague ideas into detailed, actionable design specifications',
            backstory="""You are a renowned interior designer with 20 years of experience 
            in residential and commercial spaces. You have an exceptional ability to understand 
            client needs and translate them into vivid, detailed design concepts. You're known 
            for your knowledge of design trends, color theory, and spatial psychology.""",
            verbose=True,
            allow_delegation=False,
            llm=qwen_llm  # Use local Qwen model
        )
    
    def image_generator_agent(self) -> Agent:
        return Agent(
            role='AI Visualization Specialist',
            goal='Create stunning, photorealistic interior design visualizations',
            backstory="""You are an expert in AI-powered design visualization, specializing 
            in FLUX.1-dev for architectural and interior rendering. You understand how to 
            craft prompts that result in magazine-quality interior photographs.""",
            tools=[self.image_tool],
            verbose=True,
            allow_delegation=False,
            llm=qwen_llm  # Use local Qwen model
        )
    
    def feedback_processor_agent(self) -> Agent:
        return Agent(
            role='Client Relations Specialist',
            goal='Interpret client feedback and refine design directions',
            backstory="""You excel at understanding subtle client preferences and translating 
            sometimes vague feedback into concrete design improvements. You have a talent for 
            maintaining design coherence while incorporating diverse client requests.""",
            tools=[self.rewrite_tool],
            verbose=True,
            allow_delegation=True,
            llm=qwen_llm  # Use local Qwen model
        )
    
    def quality_assurance_agent(self) -> Agent:
        return Agent(
            role='Design Quality Director',
            goal='Ensure exceptional quality and maintain design portfolio',
            backstory="""You are responsible for maintaining the highest standards in the 
            design portfolio. You have a keen eye for detail and ensure that only the best 
            designs are saved to the final collection.""",
            tools=[self.save_tool],
            verbose=True,
            allow_delegation=False,
            llm=qwen_llm  # Use local Qwen model
        )


# Task Definitions
class InteriorDesignTasks:
    def analyze_prompt_task(self, agent: Agent, initial_prompt: str) -> Task:
        return Task(
            description=f"""Analyze and enhance this interior design request:

'{initial_prompt}'

Provide a comprehensive design brief including:
1. **Design Style**: Identify the primary and secondary styles
2. **Color Palette**: Specify 3-5 colors with their roles (primary, accent, neutral)
3. **Space Layout**: Describe the spatial arrangement and flow
4. **Key Furniture**: List 5-7 essential furniture pieces with materials
5. **Lighting Plan**: Natural light sources and artificial lighting fixtures
6. **Materials & Textures**: Flooring, wall treatments, fabrics
7. **Decorative Elements**: Art, plants, accessories
8. **Mood & Atmosphere**: The feeling the space should evoke
9. **Enhanced Prompt**: A detailed 3-4 sentence prompt for image generation

Format your response as a structured brief.""",
            agent=agent,
            expected_output="Detailed design brief with enhanced prompt"
        )
    
    def generate_image_task(self, agent: Agent, seed: Optional[int] = None) -> Task:
        return Task(
            description=f"""Using the enhanced prompt from the design brief, generate a 
photorealistic interior design visualization. 

Requirements:
- Use the exact enhanced prompt provided
- Ensure high quality and photorealism
- Capture all specified design elements
- Maintain proper lighting and perspective
{"- Use seed: " + str(seed) if seed else "- Use random seed"}

IMPORTANT: Use the generate_interior_image tool to create the image and return THE EXACT TOOL OUTPUT without any additional formatting or description.""",
            agent=agent,
            expected_output="Generated image filepath with metadata"
        )
    
    def process_feedback_task(self, agent: Agent, feedback: str, current_image: str) -> Task:
        return Task(
            description=f"""Process this client feedback and create an improved design prompt:

Feedback: '{feedback}'
Current image: {current_image}

Analyze the feedback to:
1. Identify specific elements to change
2. Determine elements to preserve
3. Understand the client's design preferences
4. Create a refined prompt that addresses all concerns

Use the rewrite_prompt tool to generate an improved prompt that incorporates 
the feedback while maintaining design coherence.""",
            agent=agent,
            expected_output="Refined prompt incorporating client feedback"
        )
    
    def quality_check_task(self, agent: Agent, image_path: str, design_name: str) -> Task:
        return Task(
            description=f"""Perform final quality assessment and save the design:

Image: {image_path}
Design name: {design_name}

Quality checklist:
1. Technical quality (resolution, clarity, lighting)
2. Design coherence (style consistency, color harmony)
3. Completeness (all requested elements present)
4. Professional appeal (portfolio-worthy)

If approved, save the design with appropriate tags and metadata.
Tags should include: style, room type, color scheme, and key features.""",
            agent=agent,
            expected_output="Quality report and save confirmation"
        )

def direct_generate_image(prompt: str, negative_prompt: str = "", seed: Optional[int] = None) -> Dict:
    """Direct image generation without CrewAI for testing purposes"""
    tool = ImageGenerationTool()
    return tool._run(prompt, negative_prompt, seed)

# Main System Class
class InteriorDesignSystem:
    def __init__(self):
        self.agents = InteriorDesignAgents()
        self.tasks = InteriorDesignTasks()
        self.session_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "iterations": [],
            "current_prompt": None,
            "current_image": None,
            "original_request": None
        }
    
    def generate_initial_design(self, user_prompt: str, seed: Optional[int] = None) -> Dict:
        """Generate the initial design"""
        logger.info(f"Starting new design session: {self.session_data['session_id']}")
        
        try:
            logger.info("Attempting direct generation to verify tool...")
            image_tool = ImageGenerationTool()
            direct_result = image_tool._run(user_prompt, "", seed)
            if not direct_result.get("success"):
                logger.error(f"Direct image generation failed: {direct_result.get('error')}")
                return direct_result
            logger.info(f"Direct generation succeeded: {direct_result.get('filepath')}")
            
            self.session_data["original_request"] = user_prompt
            # Create crew
            crew = Crew(
                agents=[
                    self.agents.prompt_analyst_agent(),
                    self.agents.image_generator_agent()
                ],
                tasks=[
                    self.tasks.analyze_prompt_task(
                        self.agents.prompt_analyst_agent(), 
                        user_prompt
                    ),
                    self.tasks.generate_image_task(
                        self.agents.image_generator_agent(),
                        seed
                    )
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute
            result = crew.kickoff()
            
            # Parse result and extract tool output
            if isinstance(result, str):
                # Look for JSON-like structure with filepath
                json_patterns = [
                    r'\{.*?"filepath".*?\}',  # Look for JSON with filepath
                    r'\{.*?"success".*?:.*?true.*?\}',  # Look for successful JSON
                    r'\{.*?\}'  # Any JSON as fallback
                ]
                
                for pattern in json_patterns:
                    json_matches = re.findall(pattern, result, re.DOTALL)
                    if json_matches:
                        for match in json_matches:
                            try:
                                extracted = json.loads(match)
                                if extracted and "filepath" in extracted:
                                    result = extracted
                                    break
                            except json.JSONDecodeError:
                                continue
                        if isinstance(result, dict):
                            break
            
            # Parse result and update session
            if isinstance(result, dict) and result.get("filepath"):
                self.session_data["current_image"] = result.get("filepath")
                self.session_data["current_prompt"] = result.get("prompt_used")
                self.session_data["iterations"].append({
                    "type": "initial",
                    "prompt": user_prompt,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                return {
                        "success": True,
                        "filepath": result.get("filepath"),
                        "prompt_used": result.get("prompt_used", ""),
                        "seed": result.get("seed")
                 }
            else:
                logger.error(f"Failed to extract valid result: {result}")
                return {"success": False, "error": str(result)}
            
        except Exception as e:
            logger.error(f"Initial generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def iterate_on_feedback(self, feedback: str) -> Dict:
        """Process feedback and generate improved design"""
        if not self.session_data["current_image"]:
            return {"success": False, "error": "No current design to iterate on"}
        
        if len(self.session_data["iterations"]) >= config.max_iterations:
            return {"success": False, "error": f"Maximum iterations ({config.max_iterations}) reached"}
        
        try:
            # Create crew
            crew = Crew(
                agents=[
                    self.agents.feedback_processor_agent(),
                    self.agents.image_generator_agent()
                ],
                tasks=[
                    self.tasks.process_feedback_task(
                        self.agents.feedback_processor_agent(),
                        feedback,
                        self.session_data["current_image"]
                    ),
                    self.tasks.generate_image_task(
                        self.agents.image_generator_agent()
                    )
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute
            result = crew.kickoff()
            
            # Update session
            if isinstance(result, dict) and result.get("success"):
                self.session_data["current_image"] = result.get("filepath")
                self.session_data["current_prompt"] = result.get("prompt_used")
                self.session_data["iterations"].append({
                    "type": "feedback",
                    "feedback": feedback,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Feedback iteration failed: {e}")
            return {"success": False, "error": str(e)}
    
    def finalize_design(self, design_name: str) -> Dict:
        """Finalize and save the approved design"""
        if not self.session_data["current_image"]:
            return {"success": False, "error": "No design to finalize"}
        
        try:
            # Create crew
            crew = Crew(
                agents=[self.agents.quality_assurance_agent()],
                tasks=[
                    self.tasks.quality_check_task(
                        self.agents.quality_assurance_agent(),
                        self.session_data["current_image"],
                        design_name
                    )
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute
            result = crew.kickoff()
            
            # Save session data
            session_file = config.final_dir / f"session_{self.session_data['session_id']}.json"
            with open(session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"Design finalization failed: {e}")
            return {"success": False, "error": str(e)}


# Gradio Interface
def create_gradio_interface():
    system = InteriorDesignSystem()
    
    with gr.Blocks(title="Interior Design AI System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🏠 Interior Design AI System
        ### Create stunning interior designs with AI-powered generation and intelligent feedback
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                prompt_input = gr.Textbox(
                    label="Design Request",
                    placeholder="Describe your ideal interior... (e.g., 'Modern minimalist bedroom with warm lighting')",
                    lines=4
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🎨 Generate Design", variant="primary")
                    seed_input = gr.Number(label="Seed (optional)", value=-1)
                
                gr.Markdown("### Refine Your Design")
                feedback_input = gr.Textbox(
                    label="Your Feedback",
                    placeholder="What would you like to change? (e.g., 'Add more plants and warmer colors')",
                    lines=3
                )
                iterate_btn = gr.Button("🔄 Apply Changes", variant="secondary")
                
                gr.Markdown("### Save Final Design")
                with gr.Row():
                    design_name = gr.Textbox(
                        label="Design Name",
                        placeholder="my_dream_living_room"
                    )
                    save_btn = gr.Button("💾 Save Design", variant="primary")
            
            with gr.Column(scale=2):
                # Output section
                output_image = gr.Image(
                    label="Generated Design",
                    type="filepath",
                    height=600
                )
                
                with gr.Accordion("Design Details", open=False):
                    prompt_display = gr.Textbox(
                        label="Current Prompt",
                        interactive=False
                    )
                    iteration_count = gr.Number(
                        label="Iteration",
                        value=0,
                        interactive=False
                    )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
        
        # Hidden state
        state = gr.State()
        
        def generate_design(prompt, seed, state):
            if not prompt:
                return None, None, 0, "Please enter a design request", state
            
            # Initialize new system for new session
            system = InteriorDesignSystem()
            
            # Use seed if provided
            if seed == -1:
                seed = None
            else:
                seed = int(seed)
            
            result = system.generate_initial_design(prompt, seed)
            
            if isinstance(result, dict) and result.get("success"):
                return (
                    result.get("filepath"),
                    result.get("prompt_used"),
                    1,
                    "✅ Design generated successfully!",
                    system
                )
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                return None, None, 0, f"❌ Error: {error_msg}", state
        
        def iterate_design(feedback, state):
            if not feedback:
                return None, None, 0, "Please provide feedback", state
            
            if not state:
                return None, None, 0, "Please generate an initial design first", state
            
            result = state.iterate_on_feedback(feedback)
            
            if isinstance(result, dict) and result.get("success"):
                iterations = len(state.session_data["iterations"])
                return (
                    result.get("filepath"),
                    result.get("prompt_used"),
                    iterations,
                    f"✅ Design updated (iteration {iterations})",
                    state
                )
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                return None, None, 0, f"❌ Error: {error_msg}", state
        
        def save_design(name, state):
            if not name:
                return "Please provide a design name"
            
            if not state:
                return "No design to save"
            
            result = state.finalize_design(name)
            
            if isinstance(result, dict) and result.get("success"):
                return f"✅ Design saved successfully!\nLocation: {result.get('final_path')}"
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                return f"❌ Error: {error_msg}"
        
        # Connect events
        generate_btn.click(
            fn=generate_design,
            inputs=[prompt_input, seed_input, state],
            outputs=[output_image, prompt_display, iteration_count, status_output, state]
        )
        
        iterate_btn.click(
            fn=iterate_design,
            inputs=[feedback_input, state],
            outputs=[output_image, prompt_display, iteration_count, status_output, state]
        )
        
        save_btn.click(
            fn=save_design,
            inputs=[design_name, state],
            outputs=[status_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Modern Scandinavian living room with large windows and natural materials"],
                ["Cozy bohemian bedroom with warm earth tones and lots of plants"],
                ["Minimalist Japanese-inspired bathroom with clean lines and zen atmosphere"],
                ["Industrial loft kitchen with exposed brick and stainless steel appliances"],
                ["Luxury Art Deco dining room with bold geometric patterns and gold accents"]
            ],
            inputs=prompt_input
        )
    
    return interface


# CLI Interface
def cli_interface():
    """Command-line interface for the system"""
    system = InteriorDesignSystem()
    
    print("🏠 Interior Design AI System - CLI Mode")
    print("=" * 50)
    
    # Get initial prompt
    prompt = input("\nDescribe your ideal interior design:\n> ").strip()
    if not prompt:
        print("No input provided. Exiting.")
        return
    
    # Generate initial design
    print("\n🎨 Generating initial design...")
    result = system.generate_initial_design(prompt)
    
    if isinstance(result, dict) and result.get("success"):
        print(f"\n✅ Design generated: {result.get('filepath')}")
        print(f"Prompt used: {result.get('prompt_used', '')[:100]}...")
    else:
        print(f"\n❌ Generation failed: {result}")
        return
    
    # Feedback loop
    while len(system.session_data["iterations"]) < config.max_iterations:
        print(f"\n📊 Current iteration: {len(system.session_data['iterations'])}/{config.max_iterations}")
        print("\nOptions:")
        print("1. Provide feedback for improvements")
        print("2. Save current design")
        print("3. Exit without saving")
        
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == "1":
            feedback = input("\nWhat would you like to change?\n> ").strip()
            if feedback:
                print("\n🔄 Processing feedback...")
                result = system.iterate_on_feedback(feedback)
                if isinstance(result, dict) and result.get("success"):
                    print(f"\n✅ Design updated: {result.get('filepath')}")
                else:
                    print(f"\n❌ Update failed: {result}")
        
        elif choice == "2":
            name = input("\nName for your design: ").strip() or "my_design"
            print("\n💾 Saving design...")
            result = system.finalize_design(name)
            if isinstance(result, dict) and result.get("success"):
                print(f"\n✅ Design saved: {result.get('final_path')}")
            else:
                print(f"\n❌ Save failed: {result}")
            break
        
        elif choice == "3":
            print("\n👋 Exiting without saving.")
            break
        
        else:
            print("\n❌ Invalid choice.")
    
    if len(system.session_data["iterations"]) >= config.max_iterations:
        print(f"\n⚠️ Maximum iterations ({config.max_iterations}) reached.")
        save = input("Save current design? (y/n): ").strip().lower()
        if save == 'y':
            name = input("Design name: ").strip() or "my_design"
            result = system.finalize_design(name)
            print(f"\n{'✅ Saved' if result.get('success') else '❌ Save failed'}")


if __name__ == "__main__":
    import sys
    
    if "--cli" in sys.argv:
        cli_interface()
    else:
        # Web interface
        print("Starting web interface...")
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )