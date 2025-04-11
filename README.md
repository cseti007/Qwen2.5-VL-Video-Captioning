Please note that I'm not a developer. I made these scripts with the help of LLMs so please forgive me if I cannot help you with all of your technical problems.

# Qwen2.5-VL Video and Image Captioner
This project utilizes the Qwen2.5-VL large vision-language model to analyze videos and images and generate detailed captions. The project also includes a caption refinement tool that uses Qwen2.5 LLM to enhance the initially generated captions.

**Updates**:
- **11.04.2025:** TOML based configuration, bug fixes, added hugginface authentication
- **19.03.2025:** Image captions possibility is added


## Table of Contents

- [Qwen2.5-VL Video and Image Captioner](#qwen25-vl-video-and-image-captioner)
  - [Table of Contents](#table-of-contents)
  - [Components](#components)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [How to Run](#how-to-run)
  - [Supported Video Formats](#supported-video-formats)
- [Qwen2.5 Caption Refining](#qwen25-caption-refining)
  - [Prerequisites](#prerequisites-1)
  - [Configuration](#configuration-1)
  - [Acknowledgments](#acknowledgments)

## Components
The project consists of two main components:

1. **Video/Image Captioner (qwen_video_captioner.py):** Processes videos and images to generate initial captions using the Qwen2.5-VL multimodal model.
2. **Caption Refinement Tool (qwen2.5_caption_refinement.py):** Enhances the generated captions using the Qwen2.5 language model, ensuring they follow specific formatting and style guidelines.

## Prerequisites

- Python 3.9 or higher
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
pip install opencv-python tqdm tomli pandas
pip install git+https://github.com/huggingface/transformers accelerate

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision

# Install other dependencies
pip install bitsandbytes wheel
pip install flash-attn --no-build-isolation
```

## Configuration

The captioner uses a TOML configuration file (```captioning-config.toml```) for all settings. See the comments in the file for detailed explanations of all parameters.

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
Open ```captioning-config.toml``` and modify the Dconfiguration.

3. Run the Script
```
python Qwen2.5-vl-captioner_v3.py --config captioning-config.toml

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
