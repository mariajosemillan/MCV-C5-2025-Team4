import numpy as np
import random
from transformers import ResNetModel
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import torch.optim as optim
from torchvision.transforms import v2
import torch
import pandas as pd
import evaluate
import os
from dataset import FoodDataset
from model import CaptioningModel
from torch.utils.data import DataLoader
from config import *
from char_mapping import tokenizer_, word2idx, char2idx, idx2word, idx2char
DEVICE = 'cuda'
TEXT_MAX_LEN=201
    
img_path = DATASET_DIR
data = pd.read_csv(CLEANED_DATA)
partitions = np.load(DATA_PARTITIONS, allow_pickle=True).item()

all_text = "".join(data["Title"].astype(str).values)
unique= sorted(set(all_text))
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

test_loader = FoodDataset(data=data,partition=partitions['test'], tokenizer=tokenizer_)
batch_size=1
test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, num_workers=4)
cont=0
for enc, dec, epochs, lr, opt, batch_size in zip(ENCODER, DECODER, EPOCHS, LEARNING_RATE, OPTIMIZER, BATCH_SIZE):
    model = CaptioningModel(encoder=enc, decoder=dec).to(DEVICE)
    model.load_state_dict(torch.load("/ghome/c5mcv04/MCV-C5-2025-Team4/w3/models/_epoch30_lr0.001_optimizerAdamW_encoderresnet-50_decodergru_wordpiece_teacherTrue1/best_model.pth"))
    list_captions=[]
    list_pred_caption=[]
    with torch.no_grad():
        for imgs, captions in test_dataloader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs,target=None)  # batch, vocab_size, seq_len
            predicted_indices = torch.argmax(outputs, dim=1)  # batch, seq_len

            for i in range(predicted_indices.shape[0]):
                pred_caption_idx = predicted_indices[i].cpu().numpy().tolist()
                true_caption_idx = captions[i].cpu().numpy().tolist()

                if USE_CHAR_MAPPING:
                    pred_caption = ''.join([idx2char[idx] for idx in pred_caption_idx if idx not in [char2idx['<PAD>'], char2idx['<EOS>'], char2idx['<SOS>']]]).strip()
                    true_caption = ''.join([idx2char[idx] for idx in true_caption_idx if idx not in [char2idx['<PAD>'], char2idx['<EOS>'], char2idx['<SOS>']]]).strip()

                elif USE_WORD_MAPPING:
                    pred_caption = ' '.join([idx2word[idx] for idx in pred_caption_idx if idx not in [word2idx['<PAD>'], word2idx['<EOS>'], word2idx['<SOS>']]]).strip()
                    true_caption = ' '.join([idx2word[idx] for idx in true_caption_idx if idx not in [word2idx['<PAD>'], word2idx['<EOS>'], word2idx['<SOS>']]]).strip()

                elif USE_WORDPIECE_MAPPING:
                    pred_caption = tokenizer_.convert_ids_to_tokens(pred_caption_idx, skip_special_tokens=True)
                    pred_caption= tokenizer_.convert_tokens_to_string(pred_caption)
                    # pred_caption = ' '.join(pred_caption).replace("##", "").strip()
                    true_caption = tokenizer_.convert_ids_to_tokens(true_caption_idx, skip_special_tokens=True)
                    true_caption= tokenizer_.convert_tokens_to_string(true_caption)
                    
                    # true_caption = ' '.join(true_caption).replace("##", "").strip()
                print(f"Sample {cont}:")
                cont=cont+1
                print(f"Ground Truth: {true_caption}")
                print(f"Predicted: {pred_caption}")
                print("-" * 50)
                list_pred_caption.append(pred_caption)
                list_captions.append(true_caption)
                
    # print(list_pred_caption)
    # print(list_captions)
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    res_b_1 = bleu.compute(predictions=list_pred_caption, references=[[cap] for cap in list_captions], max_order=1)
    res_b_2 = bleu.compute(predictions=list_pred_caption, references=[[cap] for cap in list_captions], max_order=2)
    res_b_3 = bleu.compute(predictions=list_pred_caption, references=[[cap] for cap in list_captions], max_order=3)
    res_b_4 = bleu.compute(predictions=list_pred_caption, references=[[cap] for cap in list_captions], max_order=4)
    res_r = rouge.compute(predictions=list_pred_caption, references=[[cap] for cap in list_captions])
    res_m = meteor.compute(predictions=list_pred_caption, references=[[cap] for cap in list_captions])

    print("BLEU_1:", res_b_1)
    print("BLEU_2:", res_b_2)
    print("BLEU_3:", res_b_3)
    print("BLEU_4:", res_b_4)
    print("ROUGE:", res_r)
    print("METEOR:", res_m)

