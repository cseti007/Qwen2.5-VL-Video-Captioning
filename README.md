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
# Clone the repository
git clone [your-repo-url]
cd [repo-name]

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

2. Install the required packages:

```bash
# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers
pip install opencv-python
pip install tqdm
pip install bitsandbytes
pip install accelerate
```

## Configuration

The configuration is defined in the `DEFAULT_CONFIG` dictionary within the `Qwen2.5-vl-captioner3.py` file. You'll need to modify this section directly in the code to customize the behavior of the video captioning system.

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

To modify the configuration, open `Qwen2.5-vl-captioner3.py` and locate the `DEFAULT_CONFIG` dictionary. Here's an example of how to modify it:

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

## Supported Video Formats

- .mp4
- .avi
- .mov

## Acknowledgments

This project uses the Qwen2.5-VL model developed by Alibaba Cloud.
