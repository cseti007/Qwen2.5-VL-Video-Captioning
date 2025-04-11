import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import tomli
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class Config:
    # File paths
    input_csv: str
    output_csv: str
    
    # Column names
    input_column: str
    output_column: str
    
    # Model configuration
    model_name: str
    max_tokens: int
    
    # Quantization settings
    use_quantization: bool
    quantization_bits: int
    
    # Generation settings
    temperature: float
    top_p: float
    batch_size: int
    
    # Prompts
    system_prompt: str
    user_prompt: str
    
    @classmethod
    def from_toml(cls, toml_path: str) -> 'Config':
        """Create Config instance from TOML file."""
        try:
            with open(toml_path, "rb") as f:
                config_dict = tomli.load(f)
            
            # Extract values from the config dictionary
            return cls(
                # File paths
                input_csv=config_dict["files"]["input_csv"],
                output_csv=config_dict["files"]["output_csv"],
                
                # Column names
                input_column=config_dict["columns"]["input_column"].strip('"'),
                output_column=config_dict["columns"]["output_column"].strip('"'),
                
                # Model configuration
                model_name=config_dict["model"]["name"],
                max_tokens=config_dict["model"]["max_tokens"],
                
                # Quantization settings
                use_quantization=config_dict["model"]["quantization"]["enabled"],
                quantization_bits=config_dict["model"]["quantization"]["bits"],
                
                # Generation settings
                temperature=config_dict["generation"]["temperature"],
                top_p=config_dict["generation"]["top_p"],
                batch_size=config_dict["generation"]["batch_size"],
                
                # Prompts
                system_prompt=config_dict["prompts"]["system_prompt"],
                user_prompt=config_dict["prompts"]["user_prompt"]
            )
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

def process_captions(config: Config):
    """Process captions based on the provided configuration."""
    # Initialize the model
    refiner = Qwen25Refiner(config)
    
    try:
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
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        raise
    
    finally:
        print("\nProcessing complete!")

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Qwen2.5 Caption Refinement Tool")
    parser.add_argument("--config", type=str, default="refinement_config.toml",
                        help="Path to TOML configuration file (default: refinement_config.toml)")
    parser.add_argument("--input", type=str, default=None,
                        help="Override input CSV path from config")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output CSV path from config")
    
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
        config.input_csv = args.input
        print(f"Input CSV path set to: {config.input_csv}")
    
    if args.output:
        config.output_csv = args.output
        print(f"Output CSV path set to: {config.output_csv}")
    
    # Process captions
    process_captions(config)

if __name__ == "__main__":
    main()