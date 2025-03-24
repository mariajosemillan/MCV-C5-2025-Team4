import numpy as np
import random
from transformers import ResNetModel, AutoTokenizer
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import pandas as pd
import evaluate
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb

wandb.login(key='72429990e8cd3ab9daf1dea018b3a069a69dd3c0')
wandb.init(project="C5")

DEVICE = 'cuda'
img_path = '/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/foodDataset/Food_Images'
cap_path = '/ghome/c5mcv04/MCV-C5-2025-Team4/w3/laila/f_cleaned.csv'
data = pd.read_csv(cap_path)
partitions = np.load('/ghome/c5mcv04/MCV-C5-2025-Team4/w3/food_data.npy', allow_pickle=True).item()

# Tokenization Options
USE_WORD_MAPPING = True
USE_CHAR_MAPPING = False
USE_WORDPIECE_MAPPING = False

# Character-based tokenization setup
if USE_CHAR_MAPPING:
    chars = ['<SOS>', '<EOS>', '<PAD>','®', '’','/','é','\n','+', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '–', '-', '_','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','é','á','ó','í','ú','ñ','ä','ü','ï','ë','ö','à','%', 'è' ,'ì', 'ò', 'ù','“',"â", "ê", "î", "ô", "û", "ã", "õ", "ā", "ē", "ī", "ō", "ū", "ă", "ĕ", "ĭ", "ŏ", "ŭ", "ǎ", "ě", "ǐ", "ǒ", "ǔ", "ą", "ę", "į", "ǫ", "ų", "ø", "å",'ç', 'Á', 'É', 'Í', 'Ó', 'Ú', 'À', 'È', 'Ì', 'Ò', 'Ù', 'Â', 'Ê', 'Î', 'Ô', 'Û', 'Ä', 'Ë', 'Ï', 'Ö', 'Ü', 'Ã', 'Õ', 'Ñ']
    NUM_TOKENS = len(chars)
    idx2char = {k: v for k, v in enumerate(chars)}
    char2idx = {v: k for k, v in enumerate(chars)}

# Word-based tokenization setup
if USE_WORD_MAPPING:
    all_words = []
    for title in data["Title"].astype(str).values:
        words = title.lower().split()  
        all_words.extend(words)
    unique_words = sorted(set(all_words))
    special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    words = special_tokens + unique_words
    NUM_TOKENS = len(words)
    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for idx, word in enumerate(words)}

# WordPiece tokenization setup
if USE_WORDPIECE_MAPPING:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = ["[SOS]", "[EOS]", "[PAD]", "[UNK]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    NUM_TOKENS = tokenizer.vocab_size + len(special_tokens)

# Constants
TEXT_MAX_LEN_CHAR = 115  # Maximum number of characters for character-based tokenization
TEXT_MAX_LEN_WORD = 50   # Maximum number of words for word-based tokenization
TEXT_MAX_LEN_WORDPIECE = 50  # Maximum number of subword tokens for WordPiece tokenization

if USE_WORD_MAPPING:
    TEXT_MAX_LEN = TEXT_MAX_LEN_WORD
elif USE_CHAR_MAPPING:
    TEXT_MAX_LEN = TEXT_MAX_LEN_CHAR
elif USE_WORDPIECE_MAPPING:
    TEXT_MAX_LEN = TEXT_MAX_LEN_WORDPIECE
def token_indices_to_text(indices):
    """
    Convert token indices to text based on the tokenization method.
    """
    if USE_WORD_MAPPING:
        # Convert word indices to words using idx2word
        tokens = [idx2word.get(idx, "<UNK>") for idx in indices]
        text = " ".join(tokens)  # Join words with spaces
    elif USE_CHAR_MAPPING:
        # Convert character indices to characters using idx2char
        tokens = [idx2char.get(idx, " ") for idx in indices]
        text = "".join(tokens)  # Join characters without spaces
    elif USE_WORDPIECE_MAPPING:
        # Convert WordPiece indices to tokens using tokenizer
        tokens = tokenizer.convert_ids_to_tokens(indices)
        text = tokenizer.convert_tokens_to_string(tokens)  # Join tokens into a sentence
    else:
        raise ValueError("No tokenization method selected!")

    # Remove padding tokens and special tokens
    if USE_WORD_MAPPING or USE_CHAR_MAPPING:
        text = text.replace("<PAD>", "").replace("<SOS>", "").replace("<EOS>", "").strip()
    elif USE_WORDPIECE_MAPPING:
        text = text.replace("[PAD]", "").replace("[SOS]", "").replace("[EOS]", "").strip()

    return text
class Data(Dataset):
    def __init__(self, data, partition, img_path):
        self.data = data
        self.max_len = TEXT_MAX_LEN
        self.partition = partition
        self.img_path = img_path
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        # Image processing
        img_filename = self.data.iloc[idx]['Image_Name']
        caption = self.data.iloc[idx]['Title']
        img = Image.open(f'{self.img_path}/{img_filename}.jpg').convert('RGB')
        img = self.img_proc(img)

        # Caption processing
        if USE_WORD_MAPPING:
            # Convert the caption into a list of words, lowercased
            words = caption.lower().split()
            words = ["<SOS>"] + words + ["<EOS>"]
            gap = self.max_len - len(words)
            words.extend(["<PAD>"] * gap)
            cap_idx = [word2idx.get(word, word2idx["<UNK>"]) for word in words]
            
            min_idx, max_idx = min(cap_idx), max(cap_idx)
            if max_idx >= NUM_TOKENS or min_idx < 0:
                print(f"Caption: {caption}")
                print(f"Token indices: {cap_idx}")
                raise ValueError(f"Invalid token indices: min={min_idx}, max={max_idx}")
            
        elif USE_CHAR_MAPPING:

            cap_list = list(caption)
            final_list = ["<SOS>"] + cap_list + ["<EOS>"]
            gap = self.max_len - len(final_list)
            final_list.extend(["<PAD>"] * gap)
            cap_idx = [char2idx.get(char, char2idx[" "]) for char in final_list]

        elif USE_WORDPIECE_MAPPING:
    
            tokens = tokenizer.tokenize(caption)
            tokens = ["[SOS]"] + tokens + ["[EOS]"]
            gap = self.max_len - len(tokens)
            tokens.extend(["[PAD]"] * gap)
            
            cap_idx = tokenizer.convert_tokens_to_ids(tokens)
            
            # Debug: Check for out-of-bounds WordPiece indices
            min_idx, max_idx = min(cap_idx), max(cap_idx)
            if max_idx >= NUM_TOKENS or min_idx < 0:
                print(f"Caption: {caption}")
                print(f"Tokenized tokens: {tokens}")
                print(f"Token indices: {cap_idx}")
                raise ValueError(f"Invalid token indices: min={min_idx}, max={max_idx}")
            
        assert len(cap_idx) == self.max_len, f"Caption length mismatch: {len(cap_idx)} != {self.max_len}"
        # Debug: Check for invalid indices
        if max(cap_idx) >= NUM_TOKENS or min(cap_idx) < 0:
            raise ValueError(f"Invalid token indices: min={min(cap_idx)}, max={max(cap_idx)}")

        cap_tensor = torch.tensor(cap_idx, dtype=torch.long)
        return img, cap_tensor
    
torch.cuda.empty_cache()
class Model(nn.Module):
    def __init__(self, num_layers=1, nhead=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(512, NUM_TOKENS)
        self.embed = nn.Embedding(NUM_TOKENS, 512)
        # Learnable positional encodings
        self.positional_encoding = nn.Embedding(TEXT_MAX_LEN, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) 
        if USE_WORD_MAPPING or USE_CHAR_MAPPING:
            start_token = word2idx["<SOS>"] if USE_WORD_MAPPING else char2idx["<SOS>"]
        elif USE_WORDPIECE_MAPPING:
            start_token = tokenizer.convert_tokens_to_ids(["[SOS]"])[0]
        start = torch.tensor(start_token).to(DEVICE)
        start_embed = self.embed(start) 
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) 
        positions = torch.arange(0, TEXT_MAX_LEN).unsqueeze(0).expand(batch_size, -1).to(DEVICE)
        pos_encodings = self.positional_encoding(positions)
        inp = start_embeds + pos_encodings[:, 0:1, :]  
        inp = inp.permute(1, 0, 2)
        memory = feat.expand(TEXT_MAX_LEN, -1, -1) 
        memory = memory.permute(1, 0, 2)
        outputs = []
        for t in range(TEXT_MAX_LEN):

                # Transformer decoder
                out = self.transformer_decoder(inp, memory)  # Shape: (batch_size, 1, 512)
                char_logits = self.proj(out[:, -1, :].unsqueeze(0))  # Shape: (batch_size, 1, num_char)
                outputs.append(char_logits)
                
                # the predicted character (greedy decoding)
                _, predicted_token = torch.max(char_logits, dim=-1)  
                
                if USE_WORD_MAPPING or USE_CHAR_MAPPING:
                    eos_token = word2idx["<EOS>"] if USE_WORD_MAPPING else char2idx["<EOS>"]
                elif USE_WORDPIECE_MAPPING:
                    eos_token = tokenizer.convert_tokens_to_ids(["[EOS]"])[0]
                
                if (predicted_token == eos_token).all():
                    for _ in range(t + 1, TEXT_MAX_LEN):
                        outputs.append(torch.full((batch_size, NUM_TOKENS), eos_token).to(DEVICE))
                    break
                # Embed the predicted character for the next step
                next_embed = self.embed(predicted_token.squeeze(-1)) 
                
                if t + 1 < TEXT_MAX_LEN:
                    next_inp = next_embed + pos_encodings[:, t + 1:t + 2, :]
                else:
                    next_inp = next_embed
                    next_inp = next_inp.permute(1, 0, 2)
                
                inp = torch.cat((inp, next_inp), dim=1)

        for i in range(len(outputs)):
            if outputs[i].dim() == 2:  # If the tensor has shape [seq_len, num_features]
                outputs[i] = outputs[i].unsqueeze(0)  
        res = torch.stack(outputs,dim=1) # batch, seq, num_char
        if len(res.shape) == 4:
            res = res.squeeze(0)
        res = res.permute(1,2,0) # batch, 80, seq
        return res


bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
def compute_metrics(preds, targets):
    preds = [str(pred) for pred in preds]
    targets = [[str(target)] for target in targets]  
    bleu_score = bleu.compute(predictions=preds, references=targets)['bleu']
    meteor_score = meteor.compute(predictions=preds, references=targets)['meteor']
    rouge_score = rouge.compute(predictions=preds, references=targets)['rougeL']

    return {
        'bleu': bleu_score,
        'meteor': meteor_score,
        'rouge': rouge_score
    }


def train_one_epoch(model, optimizer, crit, dataloader):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (img, caption) in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        img = img.to(DEVICE)
        caption = torch.tensor(caption).to(DEVICE)

        pred = model(img)
        loss = crit(pred, caption)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"batch_train_loss": loss.item()})
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch: {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / num_batches
    return avg_loss


def eval_epoch(model, crit, dataloader):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (img, caption) in enumerate(dataloader):
            img = img.to(DEVICE)
            caption = torch.tensor(caption).to(DEVICE)

            pred = model(img)
            loss = crit(pred, caption)
            total_loss += loss.item()
            _, predicted = torch.max(pred, dim=1)  # Greedy decoding
            for i in range(predicted.shape[0]): 
                # Ground truth text
                gt_indices = caption[i].cpu().numpy()  
                gt_text = token_indices_to_text(gt_indices)  

                # Predicted text
                pred_indices = predicted[i].cpu().numpy() 
                pred_text = token_indices_to_text(pred_indices)  

                # ground truth and predicted text
                print(f"Sample {i + 1}:")
                print(f"Ground Truth: {gt_text}")
                print(f"Predicted: {pred_text}")
                print("-" * 50)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(caption.cpu().numpy())

    avg_loss = total_loss / num_batches
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics

def train(EPOCHS):
    # Create datasets
    data_train = Data(data, partitions['train'], img_path=img_path)
    data_valid = Data(data, partitions['valid'], img_path=img_path)  
    data_test = Data(data, partitions['test'], img_path=img_path)

    # Create dataloaders
    dataloader_train = DataLoader(data_train, batch_size=8, shuffle=True)
    dataloader_valid = DataLoader(data_valid, batch_size=8, shuffle=False)
    dataloader_test = DataLoader(data_test, batch_size=8, shuffle=False)

    # Initialize model, optimizer, and loss
    model = Model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)
    if USE_WORD_MAPPING:
        pad_token_index = word2idx["<PAD>"]
    elif USE_CHAR_MAPPING:
        pad_token_index = char2idx["<PAD>"]
    elif USE_WORDPIECE_MAPPING:
        pad_token_index = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]

    crit = nn.CrossEntropyLoss(ignore_index=pad_token_index)  # Ignore padding tokens

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
     
        train_loss = train_one_epoch(model, optimizer, crit, dataloader_train)
        print(f'Train Epoch: {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}')
        wandb.log({"train_loss": train_loss})

        # Evaluate on validation set
        valid_loss, valid_metrics = eval_epoch(model, crit, dataloader_valid)
        print(f'Valid Epoch: {epoch + 1}/{EPOCHS}, Loss: {valid_loss:.4f}, Metrics: {valid_metrics}')
        wandb.log({"valid_loss": valid_loss, "bleu": valid_metrics["bleu"], "meteor": valid_metrics["meteor"], "rouge": valid_metrics["rouge"]})
        torch.save(model.state_dict(), f"checkpoint_30epoch_word{epoch + 1}.pth")
        wandb.save(f"checkpoint_30epoch_word{epoch + 1}.pth")

    test_loss, test_metrics = eval_epoch(model, crit, dataloader_test)
    print(f'Test Loss: {test_loss:.4f}, Metrics: {test_metrics}')
    wandb.log({"test_loss": test_loss, "bleu": test_metrics["bleu"], "meteor": test_metrics["meteor"], "rouge": test_metrics["rouge"]})

EPOCHS = 30
train(EPOCHS)



