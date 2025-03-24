import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
from config import TEXT_MAX_LEN, DATASET_DIR, USE_WORD_MAPPING, USE_CHAR_MAPPING, USE_WORDPIECE_MAPPING
from char_mapping import char2idx, word2idx,tokenize

class FoodDataset(Dataset):
    def __init__(self, data, partition, tokenizer=None):
        self.data = data
        self.partition = partition
        self.tokenizer = tokenizer

        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        path, caption = self.partition[idx]
        img_path = os.path.join(DATASET_DIR, path)
        img = Image.open(img_path).convert("RGB")
        img = self.img_proc(img)

        if USE_WORD_MAPPING:
            #cap_list = caption.split()
            cap_list=tokenize(caption)
        elif USE_CHAR_MAPPING:
            cap_list = list(caption)
        elif USE_WORDPIECE_MAPPING:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for WordPiece mapping but not provided.")
            cap_list = self.tokenizer.tokenize(caption)  # Tokenize using WordPiece
        else:
            raise ValueError("At least one mapping method (Word, Char, or WordPiece) must be enabled.")
        
        final_list = ["<SOS>"] + cap_list + ["<EOS>"]
        gap = TEXT_MAX_LEN - len(final_list)
        final_list.extend(["<PAD>"] * gap)


        if USE_WORD_MAPPING:
            cap_idx = [word2idx.get(word, word2idx["<UNK>"]) for word in final_list]
        elif USE_CHAR_MAPPING:
            cap_idx = [char2idx.get(char, char2idx["<UNK>"]) for char in final_list]
        elif USE_WORDPIECE_MAPPING:
            cap_idx = self.tokenizer.convert_tokens_to_ids(final_list)  # Convert WordPiece tokens to IDs
        
        captions = torch.tensor(cap_idx, dtype=torch.long)

        return img, captions
