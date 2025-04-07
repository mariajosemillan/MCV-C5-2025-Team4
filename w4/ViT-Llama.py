import torch
from transformers import (
    ViTModel, 
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset as HFDataset
import evaluate
import pandas as pd
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader
import os
from PIL import Image
import csv
from huggingface_hub import login
from transformers import EvalPrediction
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchvision import transforms
from tqdm import tqdm
import wandb

login(token="TOKEN")  # needs replacement

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
}
DATASET_PATH = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/food_dataset_split"

def load_custom_dataset(split):
    image_dir = os.path.join(DATASET_PATH, f"{split}")
    csv_path = os.path.join(DATASET_PATH, f"{split}.csv")
    
    data = []
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = os.path.join(image_dir, row['Image_Name'] + '.jpg')
            caption = row['Title']
            data.append({"image_path": image_path, "caption": caption})
    return data

def create_hf_dataset(split):
    data = load_custom_dataset(split)
    return HFDataset.from_dict({
        "image_path": [d["image_path"] for d in data],
        "caption": [d["caption"] for d in data]
    })

class ImageCaptioningDataset(TorchDataset):
    def __init__(self, dataset, tokenizer, max_len=64, transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_path"]
        caption = item["caption"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        encoded = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
    
def collate_fn(batch):
    batch = [b for b in batch if b]  # Elimina los elementos vacíos
    if not batch:
        return {}
    
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone(),  # Los labels son los mismos que los inputs, ya que la pérdida es autoregresiva
    }

def get_lora_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config).to(DEVICE), tokenizer

class ViTLlamaForCaptioning(torch.nn.Module):
    def __init__(self, vit, llama):
        super().__init__()
        self.vit = vit
        self.llama = llama
        self.proj = torch.nn.Linear(vit.config.hidden_size, llama.config.hidden_size, bias=False).to(DEVICE)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        with torch.no_grad():
            vit_features = self.vit(pixel_values).last_hidden_state  # (B, N_img, D_vit)

        projected_features = self.proj(vit_features)  

        if input_ids is not None:
            text_embeds = self.llama.get_input_embeddings()(input_ids)  
            inputs_embeds = torch.cat([projected_features, text_embeds], dim=1)  

            img_attention = torch.ones(projected_features.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
            extended_attention_mask = torch.cat([img_attention, attention_mask], dim=1)  

            if labels is not None:
                ignore_labels = torch.full(projected_features.shape[:2], -100, dtype=labels.dtype, device=labels.device)
                extended_labels = torch.cat([ignore_labels, labels], dim=1)  
            else:
                extended_labels = None

            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels,
                use_cache=False
            )
            return CausalLMOutputWithPast(
                    loss=outputs.loss,
                    logits=outputs.logits,
                    past_key_values=None
                )

        return self.llama(inputs_embeds=projected_features)

def compute_metrics(predictions, references, tokenizer):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(references, skip_special_tokens=True)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

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

    return metrics

def train_and_evaluate(model_name, vit_encoder):
    lora_alpha = 8
    lora_r = 4
    num_epochs = 1  
    wandb.init(project="image-captioning", entity="c5mcv04", name=f"{MODELS[model_name]}_checkpoint_epochs{num_epochs}_alpha{lora_alpha}_r{lora_r}")

    peft_model, tokenizer = get_lora_model(MODELS[model_name])
    model = ViTLlamaForCaptioning(vit_encoder, peft_model).to(DEVICE)

    train_data = create_hf_dataset("train")
    valid_data = create_hf_dataset("valid")
    
    train_dataset = ImageCaptioningDataset(train_data, tokenizer, 64)
    eval_dataset = ImageCaptioningDataset(valid_data, tokenizer, 64)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_epoch = 0
    checkpoints_dir = f"/ghome/c5mcv04/MCV-C5-2025-Team4/w4/jobs/llama3.2_3B_nodenuevo/checkpoint_epochs{num_epochs}_alpha{lora_alpha}_r{lora_r}"

    os.makedirs(checkpoints_dir, exist_ok=True)
    best_ckpt_dir = os.path.join(checkpoints_dir, "best_checkpoint")
    last_ckpt_dir = os.path.join(checkpoints_dir, "last_checkpoint")

    os.makedirs(best_ckpt_dir, exist_ok=True)
    os.makedirs(last_ckpt_dir, exist_ok=True)

    for epoch in  tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0
        for batch in tqdm(eval_dataloader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.no_grad():
                outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_val_loss += outputs.loss.item()

            # Compute metrics
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
            all_preds.extend(pred_ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(eval_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

        # Compute metrics
        metrics = compute_metrics(all_preds, all_labels, tokenizer)
        print(f"Epoch {epoch + 1}/{num_epochs} - Evaluation metrics: {metrics}")
        wandb.log({
            "bleu_1": metrics["bleu_1"],
            "bleu_2": metrics["bleu_2"],
            "rouge_L": metrics["rouge_L"],
            "meteor": metrics["meteor"],
            "epoch": epoch + 1
        })

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            print(f"New best model found at epoch {epoch + 1}, saving to {best_ckpt_dir} ...")
            peft_model.save_pretrained(best_ckpt_dir)
            torch.save(model.state_dict(), os.path.join(best_ckpt_dir, 'model_weights.pth'))

    print("Saving the final model (last checkpoint) ...")
    os.makedirs(last_ckpt_dir, exist_ok=True)
    peft_model.save_pretrained(last_ckpt_dir)
    torch.save(model.state_dict(), os.path.join(last_ckpt_dir, 'model_weights.pth'))

    print("Training complete!")
    wandb.log({"final_model": last_ckpt_dir})
    wandb.finish()

    # --- Inference on test set ---
    print("Running inference on test set...")
    test_data = create_hf_dataset("test")
    test_dataset = ImageCaptioningDataset(test_data, tokenizer, 64)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(test_dataloader, desc="Testing", leave=False):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        all_preds.extend(pred_ids.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_metrics = compute_metrics(all_preds, all_labels, tokenizer)
    print(f"Test set metrics: {test_metrics}")

    print("==== Predicciones vs Ground Truth ====")
    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)
    for pred_caption, real_caption in zip(decoded_preds, decoded_labels):
        print("------------------------------")
        print(f"GR:   {real_caption}")
        print(f"Pred: {pred_caption}")
        print("------------------------------")


if __name__ == '__main__':
    vit_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
    vit_encoder.eval()
    for param in vit_encoder.parameters():
        param.requires_grad = False

    results = {}
    for model_name in MODELS:
        train_and_evaluate(model_name, vit_encoder)
