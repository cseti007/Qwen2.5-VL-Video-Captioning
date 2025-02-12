import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # File paths
    INPUT_CSV: str = "/home/cseti/Data/Datasets/videos/Arcane/Cut_Original/best_of/jinx/33x16x1360x768/training/test/1/video_captions.csv"
    OUTPUT_CSV: str = "/home/cseti/Data/Datasets/videos/Arcane/Cut_Original/best_of/jinx/33x16x1360x768/training/test/1/video_captions_refined.csv"
    
    # Column names
    INPUT_COLUMN: str = "caption"
    OUTPUT_COLUMN: str = "qwen"
    
    # Model configuration
    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    MAX_TOKENS: int = 200
    BATCH_SIZE: int = 1
    
    # Quantization settings
    USE_QUANTIZATION: bool = True  # Set to True to enable quantization
    QUANTIZATION_BITS: int = 8      # Can be 4 or 8
    
    # Generation settings
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

# Prompt templates
SYSTEM_PROMPT = """You are an AI prompt engineer tasked with helping me modifying a list of automatically generated prompts.

Keep the original text but only do the following modifications:
- you responses should just be the prompt
- Always start each prompt with "Dustify disintegration effect."
- Write continuously, don't use multiple paragraphs, make the text form one coherent whole
- do not mention your task or the text itself
- Don't refer to characters as 'characters' and 'persons', instead always use their gender or refer to them with their gender
- remove references to video such as "the video begins" or "the video features" etc., but keep those sentences meaningful
- mention the clothing details of the characters
- use only declarative sentences"""

class Qwen25Refiner:
    def __init__(self, config: Config):
        self.config = config
        
        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        
        # Add quantization settings if enabled
        if config.USE_QUANTIZATION:
            if config.QUANTIZATION_BITS not in [4, 8]:
                raise ValueError("QUANTIZATION_BITS must be either 4 or 8")
            
            quantization_key = f"load_in_{config.QUANTIZATION_BITS}bit"
            model_kwargs[quantization_key] = True
        
        print(f"Loading model with settings: {model_kwargs}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def refine_text(self, text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Could you enhance and refine the following text while maintaining its core meaning:\n\n{text}\n\nPlease limit the response to {self.config.MAX_TOKENS} tokens."}
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
                max_new_tokens=self.config.MAX_TOKENS,
                do_sample=True,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P
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

def main():
    # Initialize configuration
    config = Config(
        USE_QUANTIZATION=True,  # Enable or disable quantization here
        QUANTIZATION_BITS=8     # Choose 4 or 8 bits
    )
    
    # Initialize the model
    refiner = Qwen25Refiner(config)
    
    try:
        # Read input CSV
        df = pd.read_csv(config.INPUT_CSV)
        
        # Create output column if it doesn't exist
        if config.OUTPUT_COLUMN not in df.columns:
            df[config.OUTPUT_COLUMN] = ""
        
        # Process rows in batches
        total_rows = len(df)
        for i in range(0, total_rows, config.BATCH_SIZE):
            batch_end = min(i + config.BATCH_SIZE, total_rows)
            print(f"\nProcessing rows {i+1} to {batch_end} of {total_rows}")
            
            for idx in range(i, batch_end):
                if pd.isna(df.loc[idx, config.INPUT_COLUMN]):
                    print(f"Skipping row {idx+1}: Empty input text")
                    continue
                    
                if df.loc[idx, config.OUTPUT_COLUMN] != "":
                    print(f"Skipping row {idx+1}: Already processed")
                    continue
                
                print(f"Processing row {idx+1}...")
                input_text = str(df.loc[idx, config.INPUT_COLUMN])
                refined_text = refiner.refine_text(input_text)
                df.loc[idx, config.OUTPUT_COLUMN] = refined_text
            
            # Save progress after each batch
            df.to_csv(config.OUTPUT_CSV, index=False)
            print(f"Progress saved to {config.OUTPUT_CSV}")
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
    
    finally:
        print("\nProcessing complete!")

if __name__ == "__main__":
    main()