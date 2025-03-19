from dataclasses import dataclass, field
from typing import List, Optional
import os
import cv2
import torch
import json
import csv
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from enum import Enum, auto

DEFAULT_CONFIG = {
            "system": {
            "prompt": "You are a professional video analyst. Please provide an analysis of this video by covering each of these aspects in your answer. Use only one paragraph. DO NOT separate your answer into topics.",
            "use_default": False
        },
        "prompt": {
            "text": 
"""
1. **Main Content:**
    * What is the primary focus of the scene?
    * Who are the main characters visible?

2. **Object and Character Details:**
    * Don't refer to characters as 'individual', 'characters' and 'persons', instead always use their gender or refer to them with their gender.
    * Describe the appearance in detail
    * What notable objects are present?

3. **Actions and Movement:**
    * Describe ALL movements, no matter how subtle.
    * Specify the exact type of movement (walking, running, etc.).
    * Note the direction and speed of movements.

4. **Background Elements:**
    * Describe the setting and environment.
    * Note any environmental changes.

5. **Visual Style:**
    * Describe the lighting and color palette.
    * Note any special effects or visual treatments.
    * What is the overall style of the video? (e.g., realistic, animated, artistic, documentary)

6. **Camera Work:**
    * Describe EVERY camera angle change.
    * Note the distance from subjects (close-up, medium, wide shot).
    * Describe any camera movements (pan, tilt, zoom).

7. **Scene Transitions:**
    * How does each shot transition to the next?
    * Note any changes in perspective or viewing angle.

Please be extremely specific and detailed in your description. If you notice any movement or changes, describe them explicitly.""",
            "language": "en"
        },
    "model": {
        "name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "device": "cuda:0",                       # Device to run the model on
                                                # Options: "cuda", "cpu", "mps"
                                                # Use "cuda" for GPU acceleration
        "torch_dtype": "bfloat16",              # Model precision type
                                                # Options: "float32", "float16", "bfloat16"
                                                # bfloat16 offers good balance of precision/speed

        "max_new_tokens": 512,                  # Maximum length of generated text
                                                # Higher values allow longer responses
                                                # But increase memory usage and generation time
        "quantization": {
           "enabled": False,                   # Whether to use quantization
                                            # True = reduced memory usage but slightly lower quality
                                            # False = original model quality but more memory

           "bits": 8,                        # Quantization precision
                                            # Options: 4 or 8
                                            # 4 = more compression, lower quality
                                            # 8 = less compression, better quality

           "format": "nf4",                  # Quantization format. Not used. Only here for backward compatibility.
                                            # Common option: "nf4" for 4-bit normal float
                                            # Affects how weights are stored

           "double_quant": False,             # Whether to use double quantization. Can be used only with 4bit quantization. bits = 4.
                                            # True = additional memory savings
                                            # False = slightly better quality

           "quant_type": "proxy"            # Quantization algorithm type
                                           # "proxy" is recommended for stable performance can be used only with 8bit quantization. 
                                           # use nf4 or fp4 for 4bit
       },
       # Text generation parameters
       "generation": {
            "temperature": 0.3,               # Randomness in generation
                                            # Range: 0.0-1.0
                                            # Lower = more focused/deterministic
                                            # Higher = more creative/random

            "top_p": 0.9,                    # Nucleus sampling parameter
                                           # Range: 0.0-1.0
                                           # Controls diversity of generated text
                                           # Higher = more diverse outputs

            "top_k": 50,                     # Top-k sampling parameter
                                           # Limits vocabulary choices to top k tokens
                                           # Higher = more diverse word choices

            "repetition_penalty": 1.2,       # Penalty for repeating tokens
                                           # > 1.0 reduces repetition
                                           # Higher = stronger penalty

            "do_sample": True,               # Whether to use sampling.
                                           # True = use sampling strategies. num_beams should be 1.
                                           # False = greedy decoding

            "num_beams": 1,                   # Number of beams for beam search, should be 1 if do_sample is True. Can be used only if do_sample is False.
                                           # 1 = no beam search
                                           # Higher = more thorough but slower
            "seed": 424242
       },

       # Model loading options
       "loading": {
           "trust_remote_code": True,       # Whether to trust remote code
                                           # Required for some models
                                           # Security consideration for untrusted sources

           "use_flash_attention": True,     # Whether to use flash attention
                                           # Faster attention implementation
                                           # Requires compatible GPU

           "revision": None,                # Specific model revision to use
                                           # None = latest version
                                           # Can specify commit hash or tag

           "use_cache": True                # Whether to use KV cache
                                           # Speeds up generation
                                           # Uses more memory
       },

       # Memory management options
       "memory": {
           "max_memory": None,              # Maximum memory usage per device
                                           # None = no limit
                                           # Can specify in GB, e.g., {"cuda:0": "10GiB"}

           "offload_folder": None,          # Folder for CPU offloading
                                           # None = no offloading
                                           # Specify path to enable offloading

           "torch_compile": False,          # Whether to use torch.compile EXPERIMENTAL, MIGHT NOT WORK YET
                                           # Can speed up inference
                                           # But increases initial loading time

           "low_cpu_mem_usage": True        # Optimize for low CPU memory
                                           # True = lower CPU memory usage
                                           # May be slightly slower
       }
    },
    "process": {
        "process_type": "VIDEO",         # Can be "VIDEO", "IMAGE", or "BOTH"
        "fps": 8.0,                     # Frames per second for video processing
                                        # Controls how many frames to extract per second
                                        # Higher = more detailed analysis but more memory/processing time
                                        # Lower = faster processing but might miss details
                                        # Recommended range: 1-30
                                        # 8.0 is a good balance between detail and performance
                                        # Formula: frames_extracted = video_length_seconds * fps
                                        # E.g.: 10 second video at 8 fps = 80 frames
        "input_path": "/workspace/dataset",
        "output_dir": "/workspace/dataset",
        "output_format": "individual",         # Output format for captions
                                        # Options: 
                                        # - "csv": Saves all captions in a single CSV file
                                        # - "individual": Creates separate .txt files for each caption
                                        #                 using video filename (e.g., video1.mp4 -> video1.txt)
        "output_file": "video_captions.csv",
        "csv_delimiter": ","
    },
    "image": { 
        "min_pixels": 128 * 28 * 28,
        "max_pixels": 768 * 28 * 28
    },
    "video": {
        # Frame extraction parameters
        "min_frames": 4,           # Minimum number of frames to extract from video
                                  # Lower value = less memory but lower quality
                                  # Recommended range: 4-16

        "max_frames": 768,         # Maximum number of frames to extract from video
                                  # Higher value = better quality but more memory
                                  # Recommended range: 128-768, depending on GPU memory

        "frame_factor": 2,         # Rounding factor for number of frames
                                  # Usually doesn't need to be changed
                                  # Recommended value: 2

        # Resolution parameters
        "min_pixels": 128 * 28 * 28,  # Minimum number of pixels per frame
                                      # Lower = faster processing but lower quality
                                      # Recommended minimum: 128 * 28 * 28

        "max_pixels": 768 * 28 * 28,  # Maximum number of pixels per frame
                                      # Higher = better quality but more memory
                                      # Maximum depends on GPU memory

        # Optional explicit resizing BYPASSED AT THE MOMENT IN THE CODE SO USE MAX_PIXELS PARAMATERS TO ADJUST THE RESIZING
        #"resized_height": 112,    # Fixed frame height
                                  # None = automatic sizing
                                  # Value must be multiple of 28

        #"resized_width": 112,     # Fixed frame width
                                  # None = automatic sizing
                                  # Value must be multiple of 28

        # Video reading parameters
        "video_reader_backend": "torchvision",  # Video reader backend
                                               # Possible values: "torchvision" or "decord"
                                               # decord is generally faster

        "video_start": None,       # Video start time in seconds
                                  # None = from video start
                                  # E.g.: 1.5 = start from 1.5 seconds

        "video_end": None          # Video end time in seconds
                                  # None = until video end
                                  # E.g.: 10.0 = until 10 seconds
    }
}

# Constants
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov"}  # Changed to set for O(1) lookup
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}

@dataclass
class ImageConfig:
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 768 * 28 * 28

@dataclass
class VideoConfig:
    # Frame selection params
    min_frames: int = 4  # FPS_MIN_FRAMES from qwen
    max_frames: int = 768  # FPS_MAX_FRAMES from qwen
    frame_factor: int = 2  # FRAME_FACTOR from qwen
    
    # Resolution params
    min_pixels: int = 128 * 28 * 28  # VIDEO_MIN_PIXELS from qwen
    max_pixels: int = 768 * 28 * 28  # VIDEO_MAX_PIXELS from qwen
    resized_height: Optional[int] = None  # explicit height resize
    resized_width: Optional[int] = None   # explicit width resize
    
    # Video reading params
    video_reader_backend: str = "torchvision"  # or "decord"
    video_start: Optional[float] = None  # start time in seconds
    video_end: Optional[float] = None    # end time in seconds
    
    def validate(self):
        if self.video_reader_backend not in ["torchvision", "decord"]:
            raise ValueError("video_reader_backend must be either 'torchvision' or 'decord'")
        if self.min_frames < 1:
            raise ValueError("min_frames must be positive")
        if self.max_frames < self.min_frames:
            raise ValueError("max_frames must be greater than min_frames")

@dataclass
class QuantizationConfig:
    enabled: bool = True
    bits: int = 8
    format: str = "nf4"
    double_quant: bool = True
    quant_type: str = "proxy"

    def validate(self):
        if self.enabled and self.bits not in {4, 8}:
            raise ValueError("Quantization bits must be either 4 or 8")
        if self.enabled and self.bits == 4 and self.quant_type not in {"nf4", "fp4"}:
            raise ValueError("For 4-bit quantization, quant_type must be either 'nf4' or 'fp4'")
        if self.enabled and self.bits == 8 and self.quant_type != "proxy":
            raise ValueError("For 8-bit quantization, quant_type should be 'proxy'")

@dataclass
class SystemConfig:
    prompt: str = "You are a professional video analyst"
    use_default: bool = False

@dataclass
class PromptConfig:
    text: str = """Answer the following questions but don't use the word 'character', instead always use their gender or refer to them with their gender.
1. How many characters appear in the scene?
2. Can you describe all characters' appearance and clothing in detail?
3. To which direction the characters move? What is the type of their movement (walking, running, etc.)?
4. Is there physical interaction between the characters?
5. What notable objects are present?
6. Can you describe the setting and environment?"""
    language: str = "en"

@dataclass
class ProcessConfig:
    process_type: str = "VIDEO"  # Enum helyett string
    fps: float = 8.0
    input_path: Path = Path("/path/to/input")
    output_dir: Path = Path("/path/to/output")
    output_format: str = "csv"
    output_file: str = "captions.csv"
    csv_delimiter: str = "|"

    def validate(self):
        if self.output_format not in ["csv", "individual"]:
            raise ValueError("output_format must be either 'csv' or 'individual'")
        if self.process_type not in ["VIDEO", "IMAGE", "BOTH"]:  # Enum helyett string ellenőrzés
            raise ValueError("process_type must be either 'VIDEO', 'IMAGE' or 'BOTH'")
        
@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    seed: Optional[int] = None

@dataclass
class ModelLoadConfig:
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    revision: Optional[str] = None
    use_cache: bool = True

@dataclass
class MemoryConfig:
    max_memory: Optional[dict] = None
    offload_folder: Optional[str] = "./offload_folder"  # Adj meg egy tényleges mappát
    torch_compile: bool = False
    low_cpu_mem_usage: bool = True

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    loading: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config instance from dictionary."""
        model_dict = config_dict.get("model", {})
        if "quantization" in model_dict:
            model_dict["quantization"] = QuantizationConfig(**model_dict["quantization"])
        if "generation" in model_dict:
            model_dict["generation"] = GenerationConfig(**model_dict["generation"])
        if "loading" in model_dict:
            model_dict["loading"] = ModelLoadConfig(**model_dict["loading"])
        if "memory" in model_dict:
            model_dict["memory"] = MemoryConfig(**model_dict["memory"])
                
        process_dict = config_dict.get("process", {})
        if "input_path" in process_dict:
            process_dict["input_path"] = Path(process_dict["input_path"])
        if "output_dir" in process_dict:
            process_dict["output_dir"] = Path(process_dict["output_dir"])
        
        # processing video config
        video_dict = config_dict.get("video", {})
        # processing image config
        image_dict = config_dict.get("image", {})
                
        return cls(
            system=SystemConfig(**config_dict.get("system", {})),
            prompt=PromptConfig(**config_dict.get("prompt", {})),
            model=ModelConfig(**model_dict),
            process=ProcessConfig(**process_dict),
            image=ImageConfig(**image_dict),
            video=VideoConfig(**video_dict) 
        )

class VideoProcessor:
    """Handles video frame extraction and cleanup."""
    
    def __init__(self, fps: float):
        self.fps = fps
        
    def extract_frames(self, video_path: Path) -> List[Path]:
        """Extract frames from video at specified FPS."""
        video = cv2.VideoCapture(str(video_path))
        video_fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.fps)
        
        temp_dir = Path(f"temp_frames_{video_path.stem}")
        temp_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        frame_count = 0
        
        try:
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_path = temp_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
                    
                frame_count += 1
                
        finally:
            video.release()
            
        return frame_paths

    @staticmethod
    def cleanup_frames(frame_paths: List[Path]):
        """Clean up extracted frame files."""
        if not frame_paths:
            return
            
        temp_dir = frame_paths[0].parent
        for frame_path in frame_paths:
            frame_path.unlink(missing_ok=True)
        temp_dir.rmdir()

class MediaCaptioner:
    def __init__(self, config: Config):
        self.config = config
        self.video_processor = VideoProcessor(config.process.fps)
        if config.model.generation.seed is not None:
            torch.manual_seed(config.model.generation.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.model.generation.seed)
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model and processor with configuration."""
        try:
            # initialize processor first
            print(f"Initializing processor from {self.config.model.name}")
            self.processor = AutoProcessor.from_pretrained(
                self.config.model.name,
                trust_remote_code=self.config.model.loading.trust_remote_code
            )
            print("Processor initialized successfully")
            
            # initialize model
            quantization_config = None
            if self.config.model.quantization.enabled:
                self.config.model.quantization.validate()
                quantization_config = self._create_quantization_config()
                
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model.name,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                attn_implementation="flash_attention_2" if self.config.model.loading.use_flash_attention else None,
                device_map=self.config.model.device,
                quantization_config=quantization_config,
                trust_remote_code=self.config.model.loading.trust_remote_code,
                revision=self.config.model.loading.revision,
                use_cache=self.config.model.loading.use_cache,
                max_memory=self.config.model.memory.max_memory,
                offload_folder=self.config.model.memory.offload_folder,
                low_cpu_mem_usage=self.config.model.memory.low_cpu_mem_usage
            )

            if self.config.model.memory.torch_compile:
                self.model = torch.compile(self.model)
                
        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            print(f"Model name: {self.config.model.name}")
            print(f"Device: {self.config.model.device}")
            raise

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration."""
        quant = self.config.model.quantization
        
        # use quant_type for 8-bit
        if quant.enabled and quant.bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quant.enabled and quant.bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.model.torch_dtype),
                bnb_4bit_use_double_quant=quant.double_quant,
                bnb_4bit_quant_type=quant.quant_type
            )
        else:
            return None
        
    def _create_image_message(self, image_path: Path) -> List[dict]:
        """Create message list for image input."""
        messages = []
        
        if not self.config.system.use_default:
            messages.append({
                "role": "system",
                "content": self.config.system.prompt
            })
            
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": str(image_path),
                    "min_pixels": self.config.image.min_pixels,
                    "max_pixels": self.config.image.max_pixels
                },
                {"type": "text", "text": self.config.prompt.text}
            ]
        })
        
        return messages

    def process_video(self, video_path: Path) -> dict:
            """Process single video and generate caption."""
            frame_paths = self.video_processor.extract_frames(video_path)
            
            try:
                messages = []
                
                if not self.config.system.use_default:
                    messages.append({
                        "role": "system",
                        "content": self.config.system.prompt
                    })
                    
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config.prompt.text},
                        {
                            "video": [str(path) for path in frame_paths],
                            "total_pixels": self.config.video.max_pixels * len(frame_paths),
                            "min_pixels": self.config.video.min_pixels,
                            "max_pixels": self.config.video.max_pixels
                            #"resized_height": self.config.video.resized_height,
                            #"resized_width": self.config.video.resized_width
                        }
                    ]
                })
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Debug code to check frame dimensions after processing
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Debug video frame dimensions
#                if video_inputs:
#                    print("\n--- DEBUG: Video Frame Information ---")
#                    
#                    # Check the structure of video_inputs
#                    print(f"Type of video_inputs: {type(video_inputs)}")
#                    
#                    if isinstance(video_inputs, list) and len(video_inputs) > 0:
#                        first_video = video_inputs[0]
#                        print(f"Type of first video: {type(first_video)}")
#                        
#                        if isinstance(first_video, list):
#                            print(f"Number of frames in first video: {len(first_video)}")
#                            
#                            if len(first_video) > 0:
#                                # Get the first frame to check dimensions
#                                first_frame = first_video[0]
#                                
#                                if hasattr(first_frame, "width") and hasattr(first_frame, "height"):
#                                    # PIL Image
#                                    w, h = first_frame.width, first_frame.height
#                                    actual_pixels = w * h
#                                    print(f"First frame dimensions: {w}x{h}")
#                                    print(f"Actual pixels per frame: {actual_pixels}")
#                                    print(f"Configured max_pixels: {self.config.video.max_pixels}")
#                                    print(f"Ratio of actual to configured: {actual_pixels / self.config.video.max_pixels:.2f}x")
#                    
#                    print("--- End of Debug Info ---\n")
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    fps=self.config.process.fps,
                    padding=True,
                    return_tensors="pt"
                ).to(self.config.model.device)

                with torch.no_grad():
                    if self.config.model.generation.seed is not None:
                        torch.manual_seed(self.config.model.generation.seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(self.config.model.generation.seed)

                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.model.max_new_tokens,
                        temperature=self.config.model.generation.temperature,
                        top_p=self.config.model.generation.top_p,
                        top_k=self.config.model.generation.top_k,
                        repetition_penalty=self.config.model.generation.repetition_penalty,
                        do_sample=self.config.model.generation.do_sample,
                        num_beams=self.config.model.generation.num_beams,
                    )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    caption = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]

                    print(f"\nGenerated Caption for {video_path.name}:")
                    print("-" * 50)
                    print(caption)
                    print("-" * 50 + "\n")
                    
                    return {
                        "path": str(video_path),
                        "caption": caption,
                        "fps_used": self.config.process.fps,
                        "frame_count": len(frame_paths)
                    }
                    
            finally:
                self.video_processor.cleanup_frames(frame_paths)

    def process_image(self, image_path: Path) -> dict:
        """Process single image and generate caption."""
        try:
            messages = self._create_image_message(image_path)
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, _ = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.config.model.device)

            with torch.no_grad():
                if self.config.model.generation.seed is not None:
                    torch.manual_seed(self.config.model.generation.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(self.config.model.generation.seed)

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.model.max_new_tokens,
                    temperature=self.config.model.generation.temperature,
                    top_p=self.config.model.generation.top_p,
                    top_k=self.config.model.generation.top_k,
                    repetition_penalty=self.config.model.generation.repetition_penalty,
                    do_sample=self.config.model.generation.do_sample,
                    num_beams=self.config.model.generation.num_beams,
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                caption = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                print(f"\nGenerated Caption for {image_path.name}:")
                print("-" * 50)
                print(caption)
                print("-" * 50 + "\n")
                
                return {
                    "path": str(image_path),
                    "caption": caption
                }
                
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

def save_caption(result: dict, config: ProcessConfig, output_dir: Path):
    """Save a video's caption either to CSV or as TXT file.
    
    Args:
        result: Dictionary containing 'path' and 'caption' for a video
        config: ProcessConfig containing output settings
        output_dir: Directory where to save the output
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "csv":
        output_file = output_dir / config.output_file
        # Append mode for CSV to save captions as they're generated
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=config.csv_delimiter)
            # Write header if file is empty
            if output_file.stat().st_size == 0:
                writer.writerow(['path', 'caption'])
            writer.writerow([result['path'], result['caption']])
    
    else:  # individual files
        video_path = Path(result['path'])
        output_filename = video_path.stem + '.txt'
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['caption'])

def process_media(config: Config):
    """Process all media files in the specified directory."""
    input_path = Path(config.process.input_path)
    output_dir = Path(config.process.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear output file if using CSV format
    if config.process.output_format == "csv":
        output_file = output_dir / config.process.output_file
        output_file.unlink(missing_ok=True)
    
    captioner = MediaCaptioner(config)
    processed_count = 0
    
    def process_single_file(file_path: Path, is_video: bool):
        nonlocal processed_count
        try:
            rel_path = file_path.relative_to(input_path)
            print(f"\nProcessing {'video' if is_video else 'image'}: {rel_path}")
            
            if is_video:
                result = captioner.process_video(file_path)
            else:
                result = captioner.process_image(file_path)
                
            if result:
                result_dict = {
                    'path': str(rel_path),
                    'caption': result['caption']
                }
                save_caption(result_dict, config.process, output_dir)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if input_path.is_dir():
        # Define which file types to process based on config
        formats_to_process = set()
        if config.process.process_type in ["VIDEO", "BOTH"]:  # Enum helyett string
            formats_to_process.update(SUPPORTED_VIDEO_FORMATS)
        if config.process.process_type in ["IMAGE", "BOTH"]:  # Enum helyett string
            formats_to_process.update(SUPPORTED_IMAGE_FORMATS)
            
        media_files = [
            f for f in input_path.glob("**/*.*")
            if f.suffix.lower() in formats_to_process
        ]
        
        for media_file in tqdm(media_files, desc="Processing media files"):
            is_video = media_file.suffix.lower() in SUPPORTED_VIDEO_FORMATS
            process_single_file(media_file, is_video)
            
    else:
        suffix = input_path.suffix.lower()
        if suffix in SUPPORTED_VIDEO_FORMATS and config.process.process_type in ["VIDEO", "BOTH"]:
            process_single_file(input_path, True)
        elif suffix in SUPPORTED_IMAGE_FORMATS and config.process.process_type in ["IMAGE", "BOTH"]:
            process_single_file(input_path, False)
        else:
            print(f"Unsupported file format or process type: {input_path}")
            print(f"Supported video formats: {SUPPORTED_VIDEO_FORMATS}")
            print(f"Supported image formats: {SUPPORTED_IMAGE_FORMATS}")
    
    if processed_count > 0:
        save_location = config.process.output_file if config.process.output_format == "csv" else "individual txt files"
        print(f"\nResults saved to: {output_dir / save_location}")
        print(f"Total files processed: {processed_count}")
    else:
        print("No files were processed successfully.")

if __name__ == "__main__":
    config = Config.from_dict(DEFAULT_CONFIG)
    process_media(config)
