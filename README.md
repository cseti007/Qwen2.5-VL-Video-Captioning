Please note that I'm not a developer. I made these scripts with the help of LLMs so please forgive me if I cannot help you with all of your technical problems.

This repository contains two scripts that work together to generate and refine video captions using the Qwen2.5 model family:

**1. Video Captioning Script (Qwen2.5-vl-captioner.py):**

- Uses the Qwen2.5-VL (Vision-Language) model to generate detailed captions from video content
- Extracts frames at configurable rates for analysis
- Outputs captions in either CSV format or individual text files

**2. Caption Refinement Script (qwen2.5_caption_refinement.py):**

- Takes the output from the video captioning script and refines the generated captions
- Uses the Qwen2.5-7B-Instruct model to improve caption quality
- Processes CSV files in batches with progress saving


# Table of Contents

- [Qwen2.5-VL Video Captioning](#qwen25-vl-video-captioning)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
    - [Configuration Parameters](#configuration-parameters)
      - [System Configuration](#system-configuration)
      - [Model Configuration](#model-configuration)
      - [Quantization Configuration](#quantization-configuration)
      - [Generation Configuration](#generation-configuration)
      - [Process Configuration](#process-configuration)
      - [Video Processing Configuration](#video-processing-configuration)
    - [Example Configuration Modification](#example-configuration-modification)
  - [How to Run](#how-to-run)
  - [Supported Video Formats](#supported-video-formats)
- [Qwen2.5 Caption Refining](#qwen25-caption-refining)
  - [Prerequisites](#prerequisites-1)
  - [Configuration](#configuration-1)
- [Acknowledgments](#acknowledgments)

# Qwen2.5-VL Video Captioning
This project uses the Qwen2.5-VL model to generate detailed captions for videos. It can process both individual video files and entire directories, with support for customizable frame rates, resolution, and output formats.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 16GB of GPU memory for optimal performance
- FFmpeg installed on your system

## Installation

1. First, clone the repository and create a virtual environment:

```bash
# Clone this repository
git clone https://github.com/cseti007/Qwen2.5-VL-Video-Captioning
cd Qwen2.5-VL-Video-Captioning

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Clone Qwen2.5-VL repository
git clone https://github.com/QwenLM/Qwen2.5-VL
```

2. Install the required packages:

```bash
pip install qwen-vl-utils[decord]==0.0.8
pip install opencv-python tqdm
pip install git+https://github.com/huggingface/transformers accelerate

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision

# Install other dependencies
pip install bitsandbytes
pip install flash-attn --no-build-isolation
```

## Configuration

The configuration is defined in the `DEFAULT_CONFIG` dictionary within the `Qwen2.5-vl-captioner.py` file. You'll need to modify this section directly in the code to customize the behavior of the video captioning system.

### Configuration Parameters

#### System Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `system prompt` | "You are a professional video analyst..." | System-level prompt that defines the model's base behavior and response style. Can be any text directing the model's analysis approach. |
| `use_default` | `False` | If `True`, the model uses its default prompt. If `False`, uses the custom prompt specified above. |
| `prompt text` | `` | User prompt. Defines how the model should analyze videos and what aspects it should focus on.  |

#### Model Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `name` | "Qwen/Qwen2.5-VL-7B-Instruct" | Model identifier. Currently only supports the Qwen2.5-VL model family. Do not change unless using a compatible model. |
| `device` | "cuda" | Processing device for model execution. |
| `torch_dtype` | "bfloat16" | Model computation precision.  |
| `max_new_tokens` | 512 | Maximum length of generated text in tokens. Higher values allow longer responses but increase memory usage and generation time. Recommended range: 256-1024. |

#### Quantization Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `enabled` | `True` | Enables model quantization. If `True`, reduces memory usage but slightly decreases accuracy. If `False`, maintains full precision but requires more memory. |
| `bits` | 8 | Number of bits for quantization. Options: 4 (stronger compression, lower quality) or 8 (less compression, better quality). Only these two values are supported. |
| `format` | "nf4" | Quantization format. Only included for backward compatibility. "nf4" is the only supported value currently. |
| `double_quant` | `False` | Enables double quantization. Only usable with 4-bit quantization. If `True`, provides additional memory savings. If `False`, maintains better quality. |
| `quant_type` | "proxy" | Quantization algorithm type. Use "proxy" for 8-bit quantization (recommended), "nf4" or "fp4" for 4-bit quantization. |

#### Generation Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `temperature` | 0.3 | Controls randomness in text generation. Range: 0.0-1.0. Lower values = more focused/consistent outputs. Higher values = more creative/diverse outputs. |
| `top_p` | 0.9 | Nucleus sampling parameter. Range: 0.0-1.0. Controls diversity of outputs. Higher values = more diverse word choices. Recommended range: 0.1-0.9. |
| `top_k` | 50 | Number of highest probability tokens to consider for each generation step. Higher values = more diverse vocabulary. Recommended range: 20-100. |
| `repetition_penalty` | 1.2 | Penalizes repetition of tokens. Values > 1.0 reduce repetition. Higher values = stronger penalty. Recommended range: 1.0-1.5. |
| `do_sample` | `True` | Enables sampling-based generation. If `True`, uses temperature and top_p/top_k parameters. If `False`, uses deterministic generation. |
| `num_beams` | 1 | Number of beams for beam search. Only used when `do_sample=False`. More beams = more thorough search but slower generation. |
| `seed` | 424242 | Random seed for reproducibility. Same seed produces same outputs. Set to `None` for random results each time. |

#### Process Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `fps` | 8.0 | Frames per second extracted from video. Higher values = more detailed analysis but more memory/processing time. Recommended range: 1-30. Formula: total_frames = video_length * fps |
| `input_path` | "/path/to/input" | Full path to input video file or directory containing videos. Supported formats: .mp4, .avi, .mov. Can be single file or directory for batch processing. |
| `output_dir` | "/path/to/output" | Full path to output directory. Will be created if it doesn't exist. All output files will be saved here. |
| `output_format` | "csv" | Output format for captions. "csv" = all captions in single file with video paths, "individual" = separate .txt file for each video. |
| `output_file` | "video_captions.csv" | Name of the output CSV file when `output_format` is "csv". Ignored for "individual" format. |
| `csv_delimiter` | "\|" | Delimiter character for CSV output. |

#### Video Processing Configuration
Personally, I haven't tested adjusting these parameters, so I can't guarantee that it will work with anything other than the default values.
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `min_frames` | 4 | Minimum number of frames to extract from video. Lower value = less memory but might miss details. Recommended minimum: 4. |
| `max_frames` | 768 | Maximum number of frames to process. Higher value = better analysis but more memory usage. Recommended range: 128-768 based on GPU memory. |
| `frame_factor` | 2 | Frame count rounding factor. Usually doesn't need modification. Used for internal frame count optimization. |
| `min_pixels` | 128 * 28 * 28 | Minimum pixels per frame after resizing. Affects quality and memory usage. Must be multiple of 28*28. Lower = faster but lower quality. |
| `max_pixels` | 768 * 28 * 28 | Maximum pixels per frame after resizing. Must be multiple of 28*28. Higher = better quality but more memory usage. |
| `resized_height` | None | Fixed frame height after resizing. Must be multiple of 28. `None` = automatic sizing based on aspect ratio. |
| `resized_width` | None | Fixed frame width after resizing. Must be multiple of 28. `None` = automatic sizing based on aspect ratio. |
| `video_reader_backend` | "torchvision" | Video reading backend. Options: "torchvision" (more stable) or "decord" (generally faster). Choose based on your needs. |
| `video_start` | None | Start time in seconds for video processing. `None` = start from beginning. Use for processing specific segments. |
| `video_end` | None | End time in seconds for video processing. `None` = process until end. Use for processing specific segments. |

### Example Configuration Modification

To modify the configuration, open `Qwen2.5-vl-captioner.py` and locate the `DEFAULT_CONFIG` dictionary. Here's an example of how to modify it:

```python
DEFAULT_CONFIG = {
    "system": {
        "prompt": "You are a professional video analyst...",
        "use_default": False
    },
    "model": {
        "name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "device": "cuda",  # Change to "cpu" if no GPU available
        "torch_dtype": "bfloat16",
        "max_new_tokens": 512,
        "quantization": {
            "enabled": True,
            "bits": 8,
            "format": "nf4",
            "double_quant": False,
            "quant_type": "proxy"
        }
    },
    "process": {
        "fps": 8.0,  # Adjust based on your needs
        "input_path": "/your/video/path",  # Set your input path
        "output_dir": "/your/output/path",  # Set your output path
        "output_format": "csv",
        "output_file": "video_captions.csv",
        "csv_delimiter": "|"
    }
}
```
## How to Run

1. Prepare Your Environment
```
# Activate your virtual environment first
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```
2. Configure Settings
Open Qwen2.5-vl-captioner.py and modify the DEFAULT_CONFIG dictionary:
```
DEFAULT_CONFIG = {
    "process": {
        "input_path": "/path/to/your/videos",  # Input video or directory
        "output_dir": "/path/to/output",       # Output directory
        "output_format": "csv",                # "csv" or "individual"
        "fps": 8.0                            # Adjust if needed
    },
    # ... other configurations
}
```
3. Run the Script
```
python Qwen2.5-vl-captioner.py

```
## Supported Video Formats

- .mp4
- .avi
- .mov

# Qwen2.5 Caption Refining
The other script is a caption refinement tool that processes automatically generated video captions. 
**NOTE:** This script works only with CSV files at the moment
## Prerequisites
Addition packages should be installed
```
pip install pandas
```
## Configuration
In this script, configuration is handled through the Config dataclass at the top of the file. Here's how to configure it:
1. File Configuration
```
INPUT_CSV: str = "/path/to/your/input/video_captions.csv"  # Input file path
OUTPUT_CSV: str = "/path/to/your/output/video_captions_refined.csv"  # Output file path
```
2. Define Column Names
```
INPUT_COLUMN: str = "caption"  # Name of the column containing original captions
OUTPUT_COLUMN: str = "qwen"    # Name of the column where refined captions will be saved
```
3. Setup the model
```
MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"  # Model to use
MAX_TOKENS: int = 200  # Maximum length of generated text
BATCH_SIZE: int = 1    # How many captions to process at once
```
4. Configure Memory Optimization
```
USE_QUANTIZATION: bool = True  # Enable model quantization to save memory
QUANTIZATION_BITS: int = 8     # Use 8 or 4 bits (8 is better quality, 4 saves more memory)
```
6. Other Parameters
```
TEMPERATURE: float = 0.7  # Higher = more creative, lower = more focused
TOP_P: float = 0.9       # Controls diversity of outputs
```
7. Configure System Prompt
```
SYSTEM_PROMPT = """You are an AI prompt engineer..."""
```
8. 
## Acknowledgments

This project uses the [Qwen2.5-VL model](https://github.com/QwenLM/Qwen2.5-VL) developed by Alibaba Cloud.

```
@misc{Qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}


@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```
