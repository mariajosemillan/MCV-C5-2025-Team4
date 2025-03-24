# MCV-C5-2025-Team4

## Session 3: Image Captioning

### Setup and Installation
To get started with the project, install the required dependencies:

```
pip install -r requirements.txt
```

### Data
We used the Food Ingredients and Recipes Dataset, which consists of 13,582 images of various dishes.

Expected data structure:
```
dataset
└───foodDataset
    └───Food_Images
    |   └───img1.jpg
    |   └───img2.jpg
    |   └───...
    └───Food Ingredients and Recipe Dataset with Image Name Mapping.csv
```

### Repository Structure

The repository contains the following main scripts:

- split_data.ipynb: This notebook handles dataset splitting into training (80%), validation (10%), and test (10%) sets.
- config.py: Defines global parameters and configurations for the project, including file paths, model hyperparameters, batch sizes, and training options.
- char_mapping.py: Handles text mapping at different levels: character, word, and wordpiece. It provides bidirectional mappings between text and indices, facilitating text-to-numeric conversions for model training.
- dataset.py: Defines the FoodDataset class for loading and processing images and their corresponding textual descriptions.
- model.py: Defines the CaptioningModel, combining a ResNet encoder and a GRU/LSTM decoder for image captioning, with support for word, character, and WordPiece mappings.
- train.py: Trains and evaluates the CaptioningModel using specified hyperparameters, optimizers, and data partitions, with support for early stopping and logging to WandB.
- eval.py: Evaluates the performance of a pre-trained CaptioningModel on a test dataset, calculating various metrics such as BLEU, ROUGE, and METEOR for image captioning quality.
- transformer.py: This script implements a transformer-based image captioning model, utilizing a ResNet-18 backbone for image feature extraction and a Transformer decoder to generate captions, with support for different tokenization strategies and evaluation metrics like BLEU, METEOR, and ROUGE.
- early_stopping.py: Implements early stopping to halt training when the validation loss stops improving, saving the best model weights based on a specified patience and delta.
- utils.py: Includes utility function for file handling.
- [Project_presentation.pdf](Project_presentation.pdf): slides with main tasks and results. 
