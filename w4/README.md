# MCV-C5-2025-Team4

## Session 4: Image Captioning

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

- ViT-GPT2.py: Script for image captioning using custom food datasets. It allows flexible freezing of encoder/decoder layers to experiment with different training configurations and test implementation.
- ViT-Llama.py: This script trains an image captioning model using ViT and [Llama-3.2-1B and 3B](https://huggingface.co/meta-llama/Llama-3.2-1B) with LoRA, leveraging PEFT for efficient fine-tuning. It preprocesses a custom dataset, computes evaluation metrics (BLEU, ROUGE, METEOR), and logs results to Weights & Biases. The trained model is saved after each epoch and evaluated on a test set.
- multimodal.py: This script loads a pre-trained [Qwen2.5-VL-7B-Instruct model from Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) to generate descriptive titles from images using multimodal processing. It extracts images from a JSON file, generates titles using the model, and saves the results in a CSV file.
- MLLMutils.py: This script scans a specified folder for images and generates a JSON file containing image paths and text prompts for generating captions. It structures the data to be used by a Qwen model for captioning tasks.
- [Project_Presentation.pdf](Project_Presentation.pdf): slides with main tasks and results.
- [Project_report[second-draft].pdf](https://overleaf.cvc.uab.cat/read/gqthgcccwyjq#24c72e). 
