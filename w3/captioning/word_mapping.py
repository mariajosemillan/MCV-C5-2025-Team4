import pandas as pd
from config import CLEANED_DATA
from collections import Counter
import re

# Cargar los datos
data = pd.read_csv(CLEANED_DATA)

# Función para tokenizar por palabras, eliminando caracteres no deseados
def tokenize(text):
    text = text.lower()  # Convertir todo a minúsculas
    words = re.findall(r'\b\w+\b', text)  # Tokenización por palabras
    return words

# Obtener todas las palabras del texto
all_words = []
for title in data["Title"].astype(str).values:
    all_words.extend(tokenize(title))

# Contar la frecuencia de las palabras
word_freq = Counter(all_words)

# Obtener las palabras únicas
unique_words = sorted(set(all_words))

print(unique_words)

# Definir tokens especiales
special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]

# Crear mapeo de palabras a índices
words = special_tokens + unique_words
NUM_WORDS = len(words)
word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for idx, word in enumerate(words)}

# Si necesitas ver cómo quedó el mapeo
print(f"Ejemplo de mapeo de palabras a índices: {list(word2idx.items())[:10]}")