import os
import argparse
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm
import numpy as np
import gc
import pandas as pd
from transformers import EvalPrediction
import evaluate
import json
from torch.optim import AdamW

# Cargar métricas de Hugging Face
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


def compute_metrics(eval_pred: EvalPrediction, tokenizer, save_dir):
    predictions, references = eval_pred.predictions, eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim == 3:  
        predictions = predictions.argmax(axis=-1)

    if all(isinstance(pred, str) for pred in predictions):
        decoded_preds = predictions
        decoded_labels=references
    else:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(references, skip_special_tokens=True)

    bleu_1 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels], max_order=1)["bleu"]
    bleu_2 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels], max_order=2)["bleu"]
    rouge_L = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]

    metrics = {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "rouge_L": rouge_L,
        "meteor": meteor_score,
    }

    file_path = os.path.join(save_dir, 'predictions.txt')
    with open(file_path, 'w') as file:
        file.write("Metrics:\n")
        file.write(f"{metrics}\n")

        for label, pred in zip(decoded_labels, decoded_preds):
            file.write(f"Label: {label}\n")
            file.write(f"Pred: {pred}\n")
            file.write("\n")

    print(f"Resuls saved: {file_path}")

    return metrics

class CustomDataset(TorchDataset):
    def __init__(self, dataset, processor, tokenizer):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = Image.open(example['image']).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        encoding = self.tokenizer(example['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=16)
        return {
            'pixel_values': pixel_values,
            'input_ids': encoding.input_ids.squeeze()
        }

def load_data(data_dir, split):
    """
    Loads the image paths and associated text from a CSV file.
    
    Args:
        data_dir (str): Path to the directory where the images are stored.
        split (str): The dataset split to load ('train', 'valid', or 'test').
    
    Returns:
        img_paths (list): List of image file paths.
        text_data (list): List of image descriptions.
    
    Raises:
        ValueError: If the value of 'split' is not 'train', 'valid', or 'test'.
    """
    if split not in ['train', 'valid', 'test', 'train_prueba', 'valid_prueba', 'test_prueba']:
        raise ValueError("The 'split' argument must be 'train', 'valid', or 'test'.")
    
    image_extensions = {'.jpeg', '.jpg', '.png', '.bmp', '.webp'}

    csv_path = os.path.join(data_dir, f"{split}.csv")
    images_dir = os.path.join(data_dir, f"{split}")

    df = pd.read_csv(csv_path)

    image_text_map = dict(zip(df['Image_Name'], df['Title']))
    img_paths, text_data = [], []
    image_files = [f for f in os.listdir(images_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    with tqdm(total=len(image_files), desc="Loading data") as pbar:
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            image_name = image_file.split('.')[0]
            if image_name in image_text_map:
                img_paths.append(image_path)
                text_data.append(image_text_map[image_name])
            pbar.update(1)
    
    return img_paths, text_data

def preprocess_data(example, processor, tokenizer):
    try:
        image = Image.open(example['image']).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze()
        encoding = tokenizer(example['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=16)
        return {
            'pixel_values': pixel_values,
            'input_ids': encoding.input_ids.squeeze()
        }
    except Exception as e:
        print(f"Problem with image file {example['image']}: {e}")
        return None

def collate_fn(batch):
    batch = [b for b in batch if b]  # Remove empty elements
    if not batch:
        return {}
    
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    input_ids = torch.stack([b['input_ids'] for b in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': input_ids  # Use 'labels' for the text inputs
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if 'num_items_in_batch' in inputs:
            inputs.pop('num_items_in_batch')

        loss = super().compute_loss(model, inputs)

        if return_outputs:
            outputs = model(**inputs)
            return loss, outputs
        return loss

    def create_optimizer(self):
        encoder_lr = 5e-5
        decoder_lr = 1e-4

        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            elif "decoder" in name:
                decoder_params.append(param)

        optimizer_grouped_parameters = [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=encoder_lr)
        return self.optimizer

def freeze_encoder_decoder(model, freeze_option, freeze_encoder_layers=0, freeze_decoder_layers=0):
    if freeze_option == "freeze_all":
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("All parameters are frozen.")

    elif freeze_option == "freeze_none":
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.decoder.parameters():
            param.requires_grad = True
        print("No parameters are frozen.")

    elif freeze_option == "freeze_embeddings":
        for param in model.encoder.embeddings.parameters():
            param.requires_grad = False
        for param in model.decoder.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.decoder.transformer.wpe.parameters():
            param.requires_grad = False
        print("Embeddings are frozen.")

    elif freeze_option == "freeze_layers":
        num_encoder_layers = len(model.encoder.encoder.layer)
        num_decoder_layers = len(model.decoder.transformer.h)

        for i in range(min(freeze_encoder_layers, num_encoder_layers)):
            for param in model.encoder.encoder.layer[i].parameters():
                param.requires_grad = False

        for i in range(min(freeze_decoder_layers, num_decoder_layers)):
            for param in model.decoder.transformer.h[i].parameters():
                param.requires_grad = False

        print(f"Frozen the first {freeze_encoder_layers} encoder layers and {freeze_decoder_layers} decoder layers.")
    
    return model

def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())  
    trainable_encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}\n")
    print(f"Encoder (ViT): {encoder_params} total params, {trainable_encoder_params} trainable")

    trainable_patch_embeddings_params = sum(p.numel() for p in model.encoder.embeddings.patch_embeddings.parameters() if p.requires_grad)
    total_patch_embeddings_params = sum(p.numel() for p in model.encoder.embeddings.patch_embeddings.parameters())
    print(f"Patch Embeddings (trainable params/total): {trainable_patch_embeddings_params} / {total_patch_embeddings_params}")

    for i, layer in enumerate(model.encoder.encoder.layer):
        trainable_layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total_layer_params = sum(p.numel() for p in layer.parameters())
        print(f"Layer {i} (trainable params/total): {trainable_layer_params} / {total_layer_params}")
    trainable_pooler_params = sum(p.numel() for p in model.encoder.pooler.parameters() if p.requires_grad)
    total_pooler_params = sum(p.numel() for p in model.encoder.pooler.parameters())
    print(f"Pooler (trainable params/total): {trainable_pooler_params} / {total_pooler_params}")
    
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    trainable_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    trainable_decoder_wte_params = sum(p.numel() for p in model.decoder.transformer.wte.parameters() if p.requires_grad)    
    total_decoder_wte_params = sum(p.numel() for p in model.decoder.transformer.wte.parameters())
    print(f"Embedding wte (trainable params/total): {trainable_decoder_wte_params} / {total_decoder_wte_params}")
    trainable_decoder_wpe_params = sum(p.numel() for p in model.decoder.transformer.wpe.parameters() if p.requires_grad)    
    total_decoder_wpe_params = sum(p.numel() for p in model.decoder.transformer.wpe.parameters())
    print(f"Embedding wpe (trainable params/total): {trainable_decoder_wpe_params} / {total_decoder_wpe_params}")

    print(f"Encoder (ViT): {encoder_params} total params, {trainable_encoder_params} trainable")
    print(f"Decoder (GPT-2): {decoder_params} total params, {trainable_decoder_params} trainable")
    for i, layer in enumerate(model.decoder.transformer.h):
        trainable_layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total_layer_params = sum(p.numel() for p in layer.parameters())
        print(f"Capa {i} (trainable params/total): {trainable_layer_params} / {total_layer_params}")

def main(path_to_model, data_dir, save_dir, subgroups_count, training_args, freeze_option, freeze_encoder_layers, freeze_decoder_layers):
    img_paths_train, text_data_train = load_data(data_dir=data_dir, split="train")
    img_paths_val, text_data_val = load_data(data_dir=data_dir, split="valid")

    if len(img_paths_train) != len(text_data_train):
        raise ValueError("The number of images and texts does not match.")

    dataset_train = Dataset.from_dict({'image': img_paths_train, 'text': text_data_train})
    del img_paths_train, text_data_train
    gc.collect()

    dataset_val = Dataset.from_dict({'image': img_paths_val, 'text': text_data_val})
    del img_paths_val, text_data_val
    gc.collect()

    processor = ViTImageProcessor.from_pretrained(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = VisionEncoderDecoderModel.from_pretrained(path_to_model).to("cuda")

    train_subgroups = np.array_split(dataset_train, subgroups_count)  # Split for optimization

    model = freeze_encoder_decoder(model, freeze_option, freeze_encoder_layers, freeze_decoder_layers)
    print_model_summary(model)

    for i, subgroup in enumerate(train_subgroups):
        print(f"Training on subgroup {i + 1}/{len(train_subgroups)}")

        custom_dataset_train = CustomDataset(subgroup, processor, tokenizer)
        custom_dataset_val = CustomDataset(dataset_val, processor, tokenizer)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=custom_dataset_train,
            eval_dataset=custom_dataset_val,
            data_collator=collate_fn,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, save_dir),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5,early_stopping_threshold=0.001)]  
        )

        trainer.train()
        torch.cuda.empty_cache()
        print(trainer.state.log_history)
        print(f"Best model saved: {trainer.state.best_model_checkpoint}")
    
    save_model = os.path.join(save_dir, "model")
    model.save_pretrained(save_model)
    save_tok = os.path.join(save_dir, "tokenizer")
    tokenizer.save_pretrained(save_tok)
    save_proc = os.path.join(save_dir, "processor")
    processor.save_pretrained(save_proc)
    save_vit = os.path.join(save_dir, "vit")
    vit = model.encoder
    vit.save_pretrained(save_vit)

    print("\nEvaluating on test set...")
    img_paths_test, text_data_test = load_data(data_dir=data_dir, split="test")
    dataset_test = Dataset.from_dict({'image': img_paths_test, 'text': text_data_test})
    test_dataset = CustomDataset(dataset_test, processor, tokenizer)
    pred_result_test = trainer.predict(test_dataset)
    print(pred_result_test.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT2-ViT model on a custom dataset.")
    parser.add_argument("--path_to_model", type=str, default='nlpconnect/vit-gpt2-image-captioning', help="Path to the pretrained GPT2-ViT model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory that will contain the final model file")

    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the results")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--subgroups_count", type=int, default=4, help="Number of subgroups to split the dataset for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    parser.add_argument("--logging_dir", type=str, default='./logs', help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps")
    parser.add_argument("--remove_unused_columns", type=bool, default=False, help="Remove unused columns")

    parser.add_argument("--freeze_option", type=str, choices=["freeze_all", "freeze_none", "freeze_embeddings", "freeze_layers"], default="freeze_none", help="Option to freeze layers of the model (encoder/decoder).")
    parser.add_argument("--freeze_encoder_layers", type=int, default=0, help="Number of encoder layers to freeze (valid only if --freeze_option is 'freeze_layers').")
    parser.add_argument("--freeze_decoder_layers", type=int, default=0, help="Number of decoder layers to freeze (valid only if --freeze_option is 'freeze_layers').")

    args = parser.parse_args()

    name_wandb=os.path.basename(args.save_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        logging_strategy="epoch",
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        save_strategy="epoch",
        remove_unused_columns=args.remove_unused_columns,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",  
        run_name=name_wandb,
        #do_predict=True
        #warmup_ratio=0.2
    )

    os.makedirs(args.save_dir,exist_ok=True)
    main(args.path_to_model, args.data_dir, args.save_dir, args.subgroups_count, training_args, args.freeze_option, args.freeze_encoder_layers, args.freeze_decoder_layers)