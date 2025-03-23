import pandas as pd
from config import CLEANED_DATA, USE_WORD_MAPPING, USE_CHAR_MAPPING, USE_WORDPIECE_MAPPING, TOKENIZER_MODEL
from collections import Counter
import re
from transformers import AutoTokenizer

# Cargar datos
data = pd.read_csv(CLEANED_DATA)

tokenizer_ = None

# Función para tokenizar por palabras
def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

word2idx = None  
char2idx = None  
# Obtener todas las palabras o caracteres del texto según la configuración
if USE_WORD_MAPPING:
    all_words = []
    for title in data["Title"].astype(str).values:
        all_words.extend(tokenize(title))

    # Obtener las palabras únicas
    unique_words = sorted(set(all_words))

    # Definir tokens especiales
    special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    words = special_tokens + unique_words
    NUM_TOKENS = len(words)
    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for idx, word in enumerate(words)}
    print(f"Ejemplo de mapeo de palabras: {list(word2idx.items())[:10]}")

elif USE_CHAR_MAPPING:
    all_text = "".join(data["Title"].astype(str).values)
    unique_chars = sorted(set(all_text))

    # Definir tokens especiales
    special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    chars = special_tokens + unique_chars
    NUM_TOKENS = len(chars)
    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for idx, char in enumerate(chars)}
    print(f"Ejemplo de mapeo de caracteres: {list(char2idx.items())[:10]}")

elif USE_WORDPIECE_MAPPING:
    # Inicializar el tokenizador de WordPiece
    tokenizer_ = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    # Agregar tokens especiales
    special_tokens = ["[SOS]", "[EOS]", "[PAD]", "[UNK]"]
    tokenizer_.add_special_tokens({"additional_special_tokens": special_tokens})

    # Tokenizar todo el texto para previsualizar cómo se divide
    corpus = data["Title"].astype(str).tolist()  # Convierte la columna a una lista de strings
    tokenized_text = [tokenizer_.tokenize(text) for text in corpus]

    NUM_TOKENS = tokenizer_.vocab_size

    print(f"Ejemplo de tokens WordPiece: {tokenized_text[:10]}")