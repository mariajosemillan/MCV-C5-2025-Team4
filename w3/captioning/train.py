import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from config import *
from dataset import FoodDataset
from model import CaptioningModel
from early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from char_mapping import tokenizer_
from utils import get_next_experiment_folder

def train_test(dataloader, optimizer, model, criterion, mode="train",teacher=False):
    model.train() if mode == "train" else model.eval()
    total_loss, total = 0, 0

    for imgs, captions in dataloader:
        imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)
        optimizer.zero_grad()
        if teacher:
            outputs = model(imgs,target=captions)
        else:
            outputs=model(imgs,target=None)
        loss = criterion(outputs, captions)

        if mode == "train":
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return total_loss / total

if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    partitions = np.load(DATA_PARTITIONS, allow_pickle=True).item()
    data = pd.read_csv(CLEANED_DATA)

    for enc, dec, epochs, lr, opt, batch_size in zip(ENCODER, DECODER, EPOCHS, LEARNING_RATE, OPTIMIZER, BATCH_SIZE):
        if USE_WORD_MAPPING:
            mapping = "word"
        elif USE_CHAR_MAPPING:
            mapping = "char"
        elif USE_WORDPIECE_MAPPING:
            mapping = "wordpiece"

        NAME_MODEL=f"epoch{epochs}_lr{lr}_optimizer{opt}_encoder{enc}_decoder{dec}_{mapping}_teacher{TEACHER}"
        output_dir = get_next_experiment_folder(MODEL_SAVE_DIR, name=f"{NAME_MODEL}")
        os.makedirs(output_dir)

        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=os.path.basename(output_dir))
        
        train_dataset = FoodDataset(data, partitions["train"], tokenizer=tokenizer_)
        val_dataset = FoodDataset(data, partitions["valid"], tokenizer=tokenizer_)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = CaptioningModel(encoder=enc, decoder=dec).to(DEVICE)
        #model.load_state_dict(torch.load("/ghome/c5mcv04/MCV-C5-2025-Team4/w3/models/_epoch30_lr0.001_optimizerSGD_encoderresnet-18_decodergru_char1/best_model.pth"))
        criterion = nn.CrossEntropyLoss()
        if opt == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        elif opt == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif opt == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

        early_stopping = EarlyStopping(save_path=f"{output_dir}/best_model.pth", delta=0.001)

        for epoch in range(epochs):
            train_loss = train_test(train_loader, optimizer, model, criterion, mode="train",teacher=TEACHER)
            val_loss = train_test(val_loader, optimizer, model, criterion, mode="test",teacher=False)

            wandb.log({"Train_loss": train_loss, "Val_loss": val_loss}, step=epoch + 1)

            if early_stopping(val_loss, model):
                break

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")
        
        wandb.finish()
