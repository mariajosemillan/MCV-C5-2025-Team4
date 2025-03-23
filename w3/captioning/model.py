import torch
import torch.nn as nn
from transformers import ResNetModel
from config import DEVICE, TEXT_MAX_LEN, USE_WORD_MAPPING, USE_CHAR_MAPPING, USE_WORDPIECE_MAPPING
from char_mapping import NUM_TOKENS, char2idx, word2idx, tokenizer_

class CaptioningModel(nn.Module):
    def __init__(self, encoder="resnet-18", decoder="gru"):
        super().__init__()
        if encoder == "resnet-18":
            self.encoder = ResNetModel.from_pretrained("microsoft/resnet-18").to(DEVICE)
        elif encoder == "resnet-50":
            self.encoder = ResNetModel.from_pretrained("microsoft/resnet-50").to(DEVICE)
            self.hidden_proj = nn.Linear(2048, 512)  # Proyección a 512
        else:
            raise ValueError("Invalid encoder")
        
        self.proj = nn.Linear(512, NUM_TOKENS)
        self.embed = nn.Embedding(NUM_TOKENS, 512)
        
        if decoder == "gru":
            self.decoder = nn.GRU(512, 512, num_layers=1, batch_first=True)
        elif decoder == "lstm":
            self.decoder = nn.LSTM(512, 512, num_layers=1, dropout=0.3, batch_first=True)
        else:
            raise ValueError("Invalid decoder")

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.encoder(img).pooler_output  # Salida del encoder
        feat = feat.view(batch_size, 1, -1)  

        if hasattr(self, "hidden_proj"):  
            feat = self.hidden_proj(feat)

        hidden = (feat.permute(1, 0, 2), torch.zeros_like(feat.permute(1, 0, 2))) if isinstance(self.decoder, nn.LSTM) else feat.permute(1, 0, 2)

        start_token = "<SOS>"

        if USE_WORD_MAPPING:
            start_idx = word2idx.get(start_token, word2idx["<UNK>"])
        elif USE_CHAR_MAPPING:
            start_idx = char2idx.get(start_token, char2idx["<UNK>"])
        elif USE_WORDPIECE_MAPPING:
            start_idx = tokenizer_.convert_tokens_to_ids(start_token)
        
        inp = self.embed(torch.full((batch_size, 1), start_idx, dtype=torch.long, device=DEVICE))

        outputs = []
        for _ in range(TEXT_MAX_LEN):
            out, hidden = self.decoder(inp, hidden)
            outputs.append(out)
            if USE_WORDPIECE_MAPPING:
                inp = self.embed(torch.argmax(self.proj(out), dim=-1))
            else:
                inp = out  

        res = torch.cat(outputs, dim=1)
        return self.proj(res).permute(0, 2, 1)
