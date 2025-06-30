import os
import torch
import yaml
import argparse
from pathlib import Path
from huggingface_hub import HfApi
from composer import Trainer
from composer.models import HuggingFaceModel
from src.flex_bert import create_flex_bert_mlm

def parse_args():
    parser = argparse.ArgumentParser(description='Create a random init Composer model and upload to HF')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the training config YAML file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/random_init',
                        help='Directory to save the model checkpoints')
    parser.add_argument('--repo_id', type=str, default='PLACEHOLDER',
                        help='HuggingFace repository ID to upload the model')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token for private repos')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Creating model with config from {args.config_path}")
    
    model_config = config['model']['model_config']
    
    valid_attention_types = ['base', 'parallel', 'rope', 'rope_parallel']
    if 'attention_layer' in model_config and model_config['attention_layer'] not in valid_attention_types:
        print(f"Warning: Invalid attention_layer '{model_config['attention_layer']}', falling back to 'rope'")
        model_config['attention_layer'] = 'rope'
    
    try:
        model = create_flex_bert_mlm(
            pretrained_model_name=config['model']['pretrained_model_name'],
            tokenizer_name=config['tokenizer_name'],
            model_config=model_config
        )
        print("HF model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Attempting with simplified config...")
        
        for key in list(model_config.keys()):
            if key not in ['vocab_size', 'hidden_size', 'num_hidden_layers', 
                           'num_attention_heads', 'attention_layer', 'padding']:
                model_config.pop(key, None)
        
        model_config['attention_layer'] = 'rope'
        model_config['padding'] = 'unpadded'
        
        model = create_flex_bert_mlm(
            pretrained_model_name=config['model']['pretrained_model_name'],
            tokenizer_name=config['tokenizer_name'],
            model_config=model_config
        )
        print("HF model created with simplified config.")


    composer_model = HuggingFaceModel(
        model=model,
        tokenizer=None, 
        use_logits=True
    )
    print("Composer model created.")

    checkpoint_path = os.path.join(args.output_dir, "latest-rank0.pt")

    trainer = Trainer(
        model=composer_model,
        max_duration="1ba",
        device="cpu" 
    )

    print(f"Saving Composer checkpoint to {checkpoint_path}...")
    trainer.save_checkpoint(checkpoint_path)
    
    config_path = os.path.join(args.output_dir, f"{Path(args.output_dir).name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Config saved at: {config_path}")
    
    if args.token:
        print(f"Uploading to HuggingFace repo: {args.repo_id}")
        api = HfApi(token=args.token)

        try:
            api.repo_info(repo_id=args.repo_id)
            print(f"Repository {args.repo_id} already exists")
        except Exception:
            print(f"Creating new repository: {args.repo_id}")
            api.create_repo(
                repo_id=args.repo_id,
                private=True,
                repo_type="model",
                exist_ok=True
            )
            print(f"Repository {args.repo_id} created successfully")
        
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"{Path(args.output_dir).name}/latest-rank0.pt",
            repo_id=args.repo_id,
            token=args.token
        )
        
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=f"{Path(args.output_dir).name}/{Path(args.output_dir).name}.yaml",
            repo_id=args.repo_id,
            token=args.token
        )
        
        print("Upload complete!")
    else:
        print("No HuggingFace token provided. Skipping upload.")

if __name__ == "__main__":
    main()
