from dataclasses import dataclass, field
from typing import List, Optional, Set
import os
import cv2
import torch
import json
import csv
import tomli  # For reading TOML files
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from enum import Enum, auto
from huggingface_hub import login

# Constants
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov"}  # Changed to set for O(1) lookup
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}

@dataclass
class ImageConfig:
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 768 * 28 * 28
    resized_height: Optional[int] = None  # explicit height resize
    resized_width: Optional[int] = None   # explicit width resize
    
    def validate(self):
        if self.resized_height is not None or self.resized_width is not None:
            if self.resized_height is None or self.resized_width is None:
                raise ValueError("Both resized_height and resized_width must be set together")
            if self.resized_height % 28 != 0 or self.resized_width % 28 != 0:
                print("Warning: resized_height and resized_width should be multiples of 28. Values will be rounded.")

@dataclass
class VideoConfig:
    # Frame selection params
    min_frames: int = 4  
    max_frames: int = 768  
    frame_factor: int = 2  
    
    # Resolution params
    min_pixels: int = 128 * 28 * 28  
    max_pixels: int = 768 * 28 * 28  
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
        if self.resized_height is not None or self.resized_width is not None:
            if self.resized_height is None or self.resized_width is None:
                raise ValueError("Both resized_height and resized_width must be set together")
            if self.resized_height % 28 != 0 or self.resized_width % 28 != 0:
                print("Warning: resized_height and resized_width should be multiples of 28. Values will be rounded.")

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
    hf_token: Optional[str] = None

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
    
    @classmethod
    def from_toml(cls, toml_path: str) -> 'Config':
        """Create Config instance from TOML file."""
        try:
            with open(toml_path, "rb") as f:
                config_dict = tomli.load(f)
                
            # Handle empty string values in TOML (convert to None)
            config_dict = _process_empty_strings(config_dict)
                
            return cls.from_dict(config_dict)
        except Exception as e:
            print(f"Error loading TOML configuration: {str(e)}")
            raise

def _process_empty_strings(config_dict):
    """Process empty strings in TOML to None for the config system."""
    if isinstance(config_dict, dict):
        result = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and value == "":
                result[key] = None
            elif isinstance(value, dict):
                result[key] = _process_empty_strings(value)
            else:
                result[key] = value
        return result
    return config_dict

class VideoProcessor:
    """Handles video frame extraction and cleanup."""
    
    def __init__(self, fps: float):
        self.fps = fps
        
    def extract_frames(self, video_path: Path, max_frames: int = None) -> List[Path]:
        """Extract frames from video at specified FPS with an optional frame limit.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (if None, no limit)
        """
        video = cv2.VideoCapture(str(video_path))
        video_fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.fps)
        
        temp_dir = Path(f"temp_frames_{video_path.stem}")
        temp_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_path = temp_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                    
                    # Check if we've reached the max frames limit
                    if max_frames is not None and extracted_count >= max_frames:
                        print(f"Reached maximum frame limit ({max_frames}). Stopping extraction.")
                        break
                    
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

class CaptionLog:
    """Tracks files that have already been captioned."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.processed_files: Set[str] = set()
        self._load_log()
        
    def _load_log(self):
        """Load the processed files log if it exists."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.processed_files.add(line)
                print(f"Loaded {len(self.processed_files)} previously processed files from log.")
            except Exception as e:
                print(f"Error loading log file: {str(e)}")
        
    def is_processed(self, file_path: str) -> bool:
        """Check if a file has already been processed."""
        return file_path in self.processed_files
        
    def mark_processed(self, file_path: str):
        """Mark a file as processed and update the log."""
        if file_path not in self.processed_files:
            self.processed_files.add(file_path)
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{file_path}\n")
            except Exception as e:
                print(f"Error updating log file: {str(e)}")
    
    def get_processed_count(self) -> int:
        """Return the number of processed files."""
        return len(self.processed_files)

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
            # Check if we have a token and log in if needed
            if self.config.model.loading.hf_token:
                from huggingface_hub import login
                login(token=self.config.model.loading.hf_token)
                print("Logged in to Hugging Face Hub with provided token")

            # initialize processor first
            print(f"Initializing processor from {self.config.model.name}")
            self.processor = AutoProcessor.from_pretrained(
                self.config.model.name,
                trust_remote_code=self.config.model.loading.trust_remote_code,
                min_pixels=self.config.video.min_pixels,
                max_pixels=self.config.video.max_pixels,
                token=self.config.model.loading.hf_token,
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
                low_cpu_mem_usage=self.config.model.memory.low_cpu_mem_usage,
                token=self.config.model.loading.hf_token,
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
            
        image_content = {
            "type": "image", 
            "image": str(image_path),
            "min_pixels": self.config.image.min_pixels,
            "max_pixels": self.config.image.max_pixels
        }
        
        # Add resized dimensions if specified
        if hasattr(self.config.image, 'resized_height') and hasattr(self.config.image, 'resized_width'):
            if self.config.image.resized_height is not None and self.config.image.resized_width is not None:
                image_content["resized_height"] = self.config.image.resized_height
                image_content["resized_width"] = self.config.image.resized_width
            
        messages.append({
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": self.config.prompt.text}
            ]
        })
        
        return messages

    def process_video(self, video_path: Path) -> dict:
        """Process single video and generate caption."""
        frame_paths = self.video_processor.extract_frames(video_path, max_frames=self.config.video.max_frames)
        
        try:
            messages = []
            
            if not self.config.system.use_default:
                messages.append({
                    "role": "system",
                    "content": self.config.system.prompt
                })
                
            # Apply max_frames limit to frame_paths if needed
            if len(frame_paths) > self.config.video.max_frames:
                print(f"Limiting frames from {len(frame_paths)} to {self.config.video.max_frames}")
                frame_paths = frame_paths[:self.config.video.max_frames]
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": self.config.prompt.text},
                    {
                        "type": "video",
                        "video": [str(path) for path in frame_paths],
                        "min_pixels": self.config.video.min_pixels,
                        "max_pixels": self.config.video.max_pixels,
                        "fps": self.config.process.fps,
                        "max_frames": self.config.video.max_frames,
                        **({"resized_height": self.config.video.resized_height, 
                            "resized_width": self.config.video.resized_width} 
                           if self.config.video.resized_height is not None and 
                              self.config.video.resized_width is not None else {})
                    }
                ]
            })
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Debug code to check frame dimensions after processing
            image_inputs, video_inputs = process_vision_info(messages)
            
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

def save_caption(result: dict, config: ProcessConfig, output_dir: Path, caption_log: CaptionLog):
    """Save a video's caption either to CSV or as TXT file.
    
    Args:
        result: Dictionary containing 'path' and 'caption' for a video
        config: ProcessConfig containing output settings
        output_dir: Directory where to save the output
        caption_log: Log to track processed files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rel_path = result['path']
    
    # Mark the file as processed in the log
    caption_log.mark_processed(rel_path)

    if config.output_format == "csv":
        output_file = output_dir / config.output_file
        # Append mode for CSV to save captions as they're generated
        file_exists = output_file.exists() and output_file.stat().st_size > 0
        
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=config.csv_delimiter)  # This is correct, since config is ProcessConfig here
            # Write header if file is empty
            if not file_exists:
                writer.writerow(['path', 'caption'])
            writer.writerow([rel_path, result['caption']])
    
    else:  # individual files
        media_path = Path(result['path'])
        output_filename = media_path.stem + '.txt'
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['caption'])

def caption_exists(file_path: Path, config: ProcessConfig, output_dir: Path) -> bool:
    """Check if a caption already exists for this file.
    
    Args:
        file_path: Path to the media file
        config: ProcessConfig containing output settings
        output_dir: Directory where outputs are saved
        
    Returns:
        bool: True if caption exists, False otherwise
    """
    if config.output_format == "csv":
        output_file = output_dir / config.output_file
        if not output_file.exists():
            return False
        
        rel_path = str(file_path.relative_to(Path(config.input_path)))
        
        try:
            with open(output_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=config.csv_delimiter)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and row[0] == rel_path:
                        return True
        except Exception:
            return False
            
        return False
    else:  # individual files
        output_filename = file_path.stem + '.txt'
        output_path = output_dir / output_filename
        return output_path.exists()
    
def process_media(config: Config):
    """Process all media files in the specified directory, skipping already captioned files."""
    input_path = Path(config.process.input_path)
    output_dir = Path(config.process.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize caption log
    log_file = output_dir / "processed_files.log"
    caption_log = CaptionLog(log_file)
    
    # Create CSV file if it doesn't exist and we're using CSV format
    if config.process.output_format == "csv":
        output_file = output_dir / config.process.output_file
        if not output_file.exists():
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=config.process.csv_delimiter)
                writer.writerow(['path', 'caption'])
    
    captioner = MediaCaptioner(config)
    processed_count = 0
    skipped_count = 0
    
    def process_single_file(file_path: Path, is_video: bool):
        nonlocal processed_count, skipped_count
        try:
            rel_path = file_path.relative_to(input_path)
            rel_path_str = str(rel_path)
            
            # Check if already processed
            if caption_log.is_processed(rel_path_str):
                print(f"Skipping already processed {'video' if is_video else 'image'}: {rel_path}")
                skipped_count += 1
                return
            
            print(f"\nProcessing {'video' if is_video else 'image'}: {rel_path}")
            
            if is_video:
                result = captioner.process_video(file_path)
            else:
                result = captioner.process_image(file_path)
                
            if result:
                result_dict = {
                    'path': rel_path_str,
                    'caption': result['caption']
                }
                save_caption(result_dict, config.process, output_dir, caption_log)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if input_path.is_dir():
        # Define which file types to process based on config
        formats_to_process = set()
        if config.process.process_type in ["VIDEO", "BOTH"]:
            formats_to_process.update(SUPPORTED_VIDEO_FORMATS)
        if config.process.process_type in ["IMAGE", "BOTH"]:
            formats_to_process.update(SUPPORTED_IMAGE_FORMATS)
            
        media_files = [
            f for f in input_path.glob("**/*.*")
            if f.suffix.lower() in formats_to_process
        ]
        
        # Display progress with tqdm
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
    
    # Print summary
    print("\n===== Processing Summary =====")
    print(f"Files processed in this run: {processed_count}")
    print(f"Files skipped (already processed): {skipped_count}")
    print(f"Total processed files in log: {caption_log.get_processed_count()}")
    
    if processed_count > 0:
        save_location = config.process.output_file if config.process.output_format == "csv" else "individual txt files"
        print(f"\nResults saved to: {output_dir / save_location}")
    elif skipped_count == 0:
        print("No files were processed successfully.")

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Video/Image Captioning")
    parser.add_argument("--config", type=str, default="config.toml",
                        help="Path to TOML configuration file (default: config.toml)")
    parser.add_argument("--input", type=str, default=None,
                        help="Override input path from config")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration from TOML file
    try:
        config = Config.from_toml(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Override input/output paths if provided
    if args.input:
        config.process.input_path = Path(args.input)
        print(f"Input path set to: {config.process.input_path}")
    
    if args.output:
        config.process.output_dir = Path(args.output)
        print(f"Output directory set to: {config.process.output_dir}")
    
    # Process media
    process_media(config)

if __name__ == "__main__":
    main()