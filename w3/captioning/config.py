import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Device configuration
DEVICE = "cuda"
TEXT_MAX_LEN = 201
BATCH_SIZE = [32, 32, 32, 32, 32, 32, 32, 32]
EPOCHS = [30, 30, 30, 30, 30, 30, 30, 30]
LEARNING_RATE = [1e-3, 1e-4, 1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-3]
OPTIMIZER=["AdamW", "AdamW", "Adam", "SGD", "AdamW", "AdamW", "Adam", "SGD"]
ENCODER=["resnet-18", "resnet-18", "resnet-18", "resnet-18", "resnet-18", "resnet-18", "resnet-18", "resnet-18"]
DECODER=["gru", "gru", "gru", "gru", "lstm", "lstm", "lstm", "lstm"]

USE_WORD_MAPPING = True
USE_CHAR_MAPPING = False 
USE_WORDPIECE_MAPPING = False
TOKENIZER_MODEL = "bert-base-uncased" if USE_WORDPIECE_MAPPING else ""

# Paths
DATASET_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/foodDataset/Food_Images"
CAPTION_FILE = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/foodDataset/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
DATA_PARTITIONS = "/ghome/c5mcv04/MCV-C5-2025-Team4/w3/captioning/food_data.npy"
CLEANED_DATA = "/ghome/c5mcv04/MCV-C5-2025-Team4/w3/data/f_cleaned.csv"
MODEL_SAVE_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/w3/models"

# WANDB Configuration
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
