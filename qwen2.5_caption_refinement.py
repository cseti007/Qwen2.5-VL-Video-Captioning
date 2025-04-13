import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import tomli
import os
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass
class Config:
    # Processing mode
    process_mode: str = "both"  # Options: "csv", "txt", "both"
    
    # File paths
    input_csv: Optional[str] = None
    output_csv: Optional[str] = None
    
    # Directory paths for text files
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Column names
    input_column: str = "input"
    output_column: str = "output"
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 200
    
    # Quantization settings
    use_quantization: bool = True
    quantization_bits: int = 8
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 1
    
    # Prompts
    system_prompt: str = ""
    user_prompt: str = "Could you enhance and refine the following text while maintaining its core meaning:\n\n{text}\n\nPlease limit the response to {max_tokens} tokens."
    
    @classmethod
    def from_toml(cls, toml_path: str) -> 'Config':
        """Create Config instance from TOML file."""
        try:
            with open(toml_path, "rb") as f:
                config_dict = tomli.load(f)
            
            # Initialize with default values
            config = cls()
            
            # Processing mode
            if "processing" in config_dict and "mode" in config_dict["processing"]:
                mode = config_dict["processing"]["mode"].lower()
                if mode in ["csv", "txt", "both"]:
                    config.process_mode = mode
                else:
                    print(f"Warning: Invalid processing mode '{mode}'. Using default 'both'")
            
            # File paths (CSV)
            if "files" in config_dict:
                if "input_csv" in config_dict["files"]:
                    config.input_csv = config_dict["files"]["input_csv"]
                if "output_csv" in config_dict["files"]:
                    config.output_csv = config_dict["files"]["output_csv"]
            
            # Directory paths (TXT)
            if "directories" in config_dict:
                if "input_dir" in config_dict["directories"]:
                    config.input_dir = config_dict["directories"]["input_dir"]
                if "output_dir" in config_dict["directories"]:
                    config.output_dir = config_dict["directories"]["output_dir"]
            
            # Column names
            if "columns" in config_dict:
                if "input_column" in config_dict["columns"]:
                    config.input_column = config_dict["columns"]["input_column"].strip('"')
                if "output_column" in config_dict["columns"]:
                    config.output_column = config_dict["columns"]["output_column"].strip('"')
            
            # Model configuration
            if "model" in config_dict:
                if "name" in config_dict["model"]:
                    config.model_name = config_dict["model"]["name"]
                if "max_tokens" in config_dict["model"]:
                    config.max_tokens = config_dict["model"]["max_tokens"]
                
                # Quantization settings
                if "quantization" in config_dict["model"]:
                    if "enabled" in config_dict["model"]["quantization"]:
                        config.use_quantization = config_dict["model"]["quantization"]["enabled"]
                    if "bits" in config_dict["model"]["quantization"]:
                        config.quantization_bits = config_dict["model"]["quantization"]["bits"]
            
            # Generation settings
            if "generation" in config_dict:
                if "temperature" in config_dict["generation"]:
                    config.temperature = config_dict["generation"]["temperature"]
                if "top_p" in config_dict["generation"]:
                    config.top_p = config_dict["generation"]["top_p"]
                if "batch_size" in config_dict["generation"]:
                    config.batch_size = config_dict["generation"]["batch_size"]
            
            # Prompts
            if "prompts" in config_dict:
                if "system_prompt" in config_dict["prompts"]:
                    config.system_prompt = config_dict["prompts"]["system_prompt"]
                if "user_prompt" in config_dict["prompts"]:
                    config.user_prompt = config_dict["prompts"]["user_prompt"]
            
            return config
            
        except Exception as e:
            print(f"Error loading TOML configuration: {str(e)}")
            raise

class Qwen25Refiner:
    def __init__(self, config: Config):
        self.config = config
        
        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        
        # Add quantization settings if enabled
        if config.use_quantization:
            if config.quantization_bits not in [4, 8]:
                raise ValueError("quantization_bits must be either 4 or 8")
            
            quantization_key = f"load_in_{config.quantization_bits}bit"
            model_kwargs[quantization_key] = True
        
        print(f"Loading model with settings: {model_kwargs}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def refine_text(self, text: str) -> str:
        # Format the user prompt with the actual text and token limit
        formatted_user_prompt = self.config.user_prompt.format(
            text=text, 
            max_tokens=self.config.max_tokens
        )
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Prepare inputs
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            # Extract generated response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response.strip()

        except Exception as e:
            print(f"Error refining text: {str(e)}")
            return ""

def process_csv_captions(config: Config, refiner: Qwen25Refiner):
    """Process captions from a CSV file."""
    try:
        if not config.input_csv or not config.output_csv:
            print("CSV processing skipped: input_csv or output_csv not specified")
            return False
            
        # Create output directory if it doesn't exist
        output_path = Path(config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input CSV
        print(f"Reading captions from {config.input_csv}")
        df = pd.read_csv(config.input_csv)
        
        # Check if input column exists
        if config.input_column not in df.columns:
            raise ValueError(f"Input column '{config.input_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        # Create output column if it doesn't exist
        if config.output_column not in df.columns:
            df[config.output_column] = ""
            print(f"Created new column '{config.output_column}' for refined captions")
        
        # Process rows in batches
        total_rows = len(df)
        print(f"Found {total_rows} rows to process")
        
        for i in range(0, total_rows, config.batch_size):
            batch_end = min(i + config.batch_size, total_rows)
            print(f"\nProcessing rows {i+1} to {batch_end} of {total_rows}")
            
            for idx in range(i, batch_end):
                if pd.isna(df.loc[idx, config.input_column]):
                    print(f"Skipping row {idx+1}: Empty input text")
                    continue
                    
                if df.loc[idx, config.output_column] != "":
                    print(f"Skipping row {idx+1}: Already processed")
                    continue
                
                print(f"Processing row {idx+1}...")
                input_text = str(df.loc[idx, config.input_column])
                refined_text = refiner.refine_text(input_text)
                df.loc[idx, config.output_column] = refined_text
            
            # Save progress after each batch
            df.to_csv(config.output_csv, index=False)
            print(f"Progress saved to {config.output_csv}")
            
        print("\nCSV processing complete!")
        return True
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return False

def process_txt_files(config: Config, refiner: Qwen25Refiner):
    """Process text files from a directory."""
    try:
        if not config.input_dir or not config.output_dir:
            print("TXT processing skipped: input_dir or output_dir not specified")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # List all .txt files in the input directory
        txt_files = [f for f in os.listdir(config.input_dir) if f.endswith('.txt')]
        total_files = len(txt_files)
        print(f"Found {total_files} .txt files to process.")
        
        if total_files == 0:
            print("No .txt files found in the input directory.")
            return False
        
        for idx, filename in enumerate(txt_files):
            input_path = os.path.join(config.input_dir, filename)
            output_path = os.path.join(config.output_dir, filename)
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                print(f"Skipping file {filename}: Output already exists")
                continue
                
            try:
                with open(input_path, "r", encoding="utf-8") as file:
                    input_text = file.read().strip()
                
                if not input_text:
                    print(f"Skipping file {filename}: Empty content")
                    continue
                
                print(f"\nProcessing file {idx+1}/{total_files}: {filename}...")
                refined_text = refiner.refine_text(input_text)
                
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(refined_text)
                    
                print(f"Refined text saved to {output_path}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        print("\nTXT file processing complete!")
        return True
        
    except Exception as e:
        print(f"Error processing TXT files: {str(e)}")
        return False

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Qwen2.5 Caption Refinement Tool")
    parser.add_argument("--config", type=str, default="refinement_config.toml",
                        help="Path to TOML configuration file (default: refinement_config.toml)")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Override input CSV path from config")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Override output CSV path from config")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory path for text files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory path for text files")
    parser.add_argument("--process-mode", type=str, choices=["csv", "txt", "both"], default=None,
                        help="Override processing mode: 'csv', 'txt', or 'both'")
    
    args = parser.parse_args()
    
    # Load configuration from TOML file
    try:
        config = Config.from_toml(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration settings")
        config = Config()
    
    # Override paths if provided via command line
    if args.input_csv:
        config.input_csv = args.input_csv
        print(f"Input CSV path set to: {config.input_csv}")
    
    if args.output_csv:
        config.output_csv = args.output_csv
        print(f"Output CSV path set to: {config.output_csv}")
    
    if args.input_dir:
        config.input_dir = args.input_dir
        print(f"Input directory path set to: {config.input_dir}")
    
    if args.output_dir:
        config.output_dir = args.output_dir
        print(f"Output directory path set to: {config.output_dir}")
    
    # Override processing mode if provided via command line
    if args.process_mode:
        config.process_mode = args.process_mode
        print(f"Processing mode set to: {config.process_mode}")
    
    print(f"Using processing mode: {config.process_mode}")
    
    # Initialize the model
    refiner = Qwen25Refiner(config)
    
    # Process based on selected mode
    csv_processed = False
    txt_processed = False
    
    if config.process_mode in ["csv", "both"]:
        csv_processed = process_csv_captions(config, refiner)
    
    if config.process_mode in ["txt", "both"]:
        txt_processed = process_txt_files(config, refiner)
    
    if not csv_processed and not txt_processed:
        print("\nNo processing was completed. Please check your configuration and input paths.")
    else:
        print("\nAll processing complete!")

if __name__ == "__main__":
    main()