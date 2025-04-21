import torch
import time
import json
import argparse
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    BitsAndBytesConfig,
    SD3Transformer2DModel,
)
from transformers import T5EncoderModel, AutoModelForCausalLM, AutoTokenizer

# # Ingredients appearing in 0.42% to 1% of dishes - Cleaned List - 87 items :
FOOD_ITEMS = ['pea', 'cheesecake', 'pear', 'peach', 'glaze', 'raspberry', 'spinach', 'meatballs', 'nut', 'buttermilk', 'strawberry', 'chops', 'apples', 'burgers', 'egg', 'celery', 'ricotta', 'cauliflower', 'avocado', 'spaghetti', 'gravy', 'crab', 'brussels', 'feta', 'asparagus', 'herb', 'pancakes', 'peppers', 'sandwiches', 'sausage', 'dip', 'curry', 'horseradish', 'ham', 'mango', 'pizza', 'maple', 'carrot', 'hazelnut', 'ribs', 'rib', 'miso', 'goat', 'olives', 'vegetables', 'broccoli', 'duck', 'prosciutto', 'fish', 'peas', 'chipotle', 'sesame', 'zucchini', 'spice', 'kale', 'syrup', 'chili', 'walnut', 'eggplant', 'tenderloin', 'pineapple', 'sage', 'banana', 'cucumber', 'peanut', 'toast', 'arugula', 'cheddar', 'stew', 'tacos', 'vanilla', 'noodles', 'mushrooms', 'pesto', 'steaks', 'milk', 'beet', 'blueberry', 'butternut', 'basil', 'parmesan', 'thyme', 'fruit', 'pecan', 'pistachio', 'cabbage']

OUTPUT_DIR = "food_train_images"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize LLM for prompt generation (using a small, efficient model)
#PROMPT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
prompt_tokenizer = None
prompt_model = None

def load_prompt_model():
    """Load the LLM for prompt generation"""
    global prompt_tokenizer, prompt_model
    if prompt_tokenizer is None:
        print("Loading prompt generation model...")
        prompt_tokenizer = AutoTokenizer.from_pretrained(PROMPT_MODEL)
        prompt_model = AutoModelForCausalLM.from_pretrained(
            PROMPT_MODEL,
            device_map="auto",
            torch_dtype=torch.float16
        )
    return prompt_tokenizer, prompt_model

def generate_llm_prompt(food_item: str, style: str = "photorealistic") -> str:
    """Generate optimized prompts for recipe image captioning training data"""
    tokenizer, model = load_prompt_model()
    
    system_prompt = (
        "You have to create prompts for AI-generated food images that will be used "
        "to train a food captioning model. Generate a single, detailed prompt "
        "that would produce an image suitable for illustrating only one dish using "
        f"the ingredient: {food_item}. The ingredient must be explicitly used in the preparation of the dish — Do not show the ingredient alone or as the sole focus —"
        "The image should be clear, well-composed,and show the ingredient in a way that would help someone recognize it "
        "in a dish context. Include details about:\n"
        "- Food styling (how it's prepared/cooked)\n"
        "- Presentation (plating, arrangement)\n"
        "The background is secondary, minimal and non-distracting, the most important and central part of the image have to be the food"
        "Keep the prompt under 25 words and focused on culinary accuracy. "
    )
    # system_prompt = (
    # "You are generating prompts for realistic food images using Stable Diffusion. "
    # "Generate a short, vivid, and descriptive prompt that would produce an image of a single, "
    # "well-composed dish featuring the ingredient: {food_item}. "
    # "The image should clearly present the dish in a way that visually highlights the ingredient. "
    # "Describe:\n"
    # "- How the food is styled or cooked\n"
    # "- How it's arranged or plated\n"
    # "- What kind of surface or background it's on (natural but minimal)\n"
    # "Avoid generic words like 'delicious' or 'tasty'. Focus on visual elements. "
    # "Do not include camera settings or lighting — that will be added later. "
    # "Keep the prompt under 30 words. Use elegant, sensory language suited for image generation."
    # )
    # system_prompt = (
    # "You have to generate concise image prompts for a text-to-image model like Stable Diffusion, that need short, descriptive visual phrases separated by commas — do not write full sentences."
    # "Your goal is to write a visually description of a realistic food photo of one full prepeared dish image including, using and "
    # "featuring the ingredient: {food_item}. "
    # "The dish havt to be a food with more ingredients that only: {food_item}."
    # "The focus must be entirely on the food itself — its preparation, texture, shape, and arrangement. "
    # "No background description the principal part of the image have to be the food."
    # "Don't ask questions, write narratively, or include extra commentary. "
    # "Just write one vivid, visual short description under 20 words. This will be followed by rendering keywords, so don't include terms like 'lighting', 'photorealistic', 'background' etc."
    # )
    # system_prompt = (
    # "You are generating concise image prompts for a text-to-image model like Stable Diffusion. "
    # "Your task is to write a single, vivid sentence that clearly describes a realistic image of one plated dish "
    # "that inlcude the ingredient: {food_item}. The focus must be entirely on the food itself — its preparation, texture, shape, and arrangement. "
    # "Use short, descriptive visual phrases separated by commas — do not write full sentences."
    # "Mention the plate or surface only if it enhances the presentation, and keep any background minimal and non-distracting. "
    # "Avoid emotional or subjective words like 'delicious' or 'beautiful'. "
    # "Do not write narratively or ask questions — this is not a caption. Just describe the visual content of the image in under 15 words. "
    # "Lighting, camera quality, or styling terms will be added separately — do not include them."
    # )
    # system_prompt = (
    # "You are generating detailed image prompts for a text-to-image model like Stable Diffusion. "
    # "Generate a single, vivid prompt describing one plated dish that visually highlights the ingredient: {food_item}. "
    # "Use short, descriptive visual phrases separated by commas — do not write full sentences. "
    # "Focus on the food’s appearance, preparation, texture, and arrangement. Mention the plate or surface only to enhance presentation. "
    # "Avoid describing the background unless absolutely necessary, and never include unrelated items like drinks, cutlery, or people. "
    # "Do not include emotions, smells, or storytelling. Just describe what's visible in the image. "
    # "Do not include camera terms or lighting — that will be added later. Limit the entire prompt to under 30 words."
    # )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a prompt featuring: {food_item}"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.6,  # Lower temperature for more factual results
        do_sample=True
    )
    
    prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt = prompt.split("[/INST]")[-1].strip()
    
    # Post-process to ensure culinary relevance
    culinary_checks = [
        "fresh", "prepared", "cooked", "ingredient", "recipe",
        "chopped", "sliced", "bowl", "plate", "serving"
    ]
    if not any(word in prompt.lower() for word in culinary_checks):
        prompt = f" {food_item} prepared for cooking, " + prompt
    
    return prompt

def get_pipeline(model_id: str, sch=True):
    model_id_lower = model_id.lower()

    if "3.5" in model_id_lower or "stable-diffusion-3" in model_id_lower:
        if "large-turbo" in model_id_lower:
            print("→ Using Stable Diffusion 3.5 Large Turbo pipeline (4-bit quantized)")

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16
            )

            t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                transformer=model_nf4,
                text_encoder_3=t5_nf4,
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
        else:
            print("→ Using standard Stable Diffusion 3.5 pipeline")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )
    
    elif "xl" in model_id_lower:
        print("→ Using Stable Diffusion XL pipeline")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        if sch:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    else:
        print("→ Using classic Stable Diffusion pipeline")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        if sch:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_food_prompt(ingredient: str, use_llm: bool = True, style: str = "photorealistic") -> str:
    """Generate prompts either with hardcoded templates or LLM"""
    if not use_llm:
        # Fallback to hardcoded prompts
        prompt_map = {
            "cake": f"Professional photo of {ingredient}, whole uncut, studio lighting, 8K",
            "pie": f"Slice of {ingredient} on plate, food magazine style",
            "cream": f"Close-up of fresh {ingredient}, texture detail, soft lighting",
            "cheese": f"Cheese board with {ingredient}, rustic presentation",
            "butter": f"Blocks of {ingredient} on marble surface",
            "pork": f"Raw {ingredient} on cutting board, natural light",
            "soup": f"Bowl of {ingredient} with steam, restaurant presentation",
            "default": f"High-quality {ingredient}, isolated on white background"
        }
        return prompt_map.get(ingredient, prompt_map["default"])
    else:
        # Use LLM to generate creative prompt
        prompt = generate_llm_prompt(ingredient, style) + ' ,soft natural lighting, high detail, photorealistic, 8k resolution, only one dish'
        return prompt

def generate_images():
    base_config = {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0", 
        "scheduler": False,
        "num_inference_steps": 80,
        "guidance_scale": 8,
        "negative_prompt": "deformed, unappetizing, incorrect texture, malformed , blurry, low quality, cartoon, CGI, oversaturated, unrealistic proportions,text, watermark, signature, logo, caption, labels, subtitles, handwriting, letters, numbers, ascii, typographic elements",
        "use_llm_prompts": True,
        "style": "photorealistic"  # Options: photorealistic, artistic, cinematic, etc.
    }

    print(f"\nGenerating {len(FOOD_ITEMS)} food images...")
    
    for food_item in FOOD_ITEMS:
        for i in range(1, 16):
            output_path = f"{OUTPUT_DIR}/{food_item}_photorealistic_{i}.png"
            prompt = generate_food_prompt(
                food_item, 
                use_llm=base_config["use_llm_prompts"],
                style=base_config["style"]
            )
            
            print(f"\nGenerating: {food_item}")
            # if "<|assistant|>" in prompt:
            #     prompt = prompt.split("<|assistant|>")[-1].strip()
            if "assistant" in prompt:
                prompt = prompt.split("assistant")[-1].strip()
            print(f"Passed Prompt: {prompt}")
            print("-"*50)
            
            start_time = time.time()
            generate_image(
                model_id=base_config["model_id"],
                prompt=prompt,
                negative_prompt=base_config["negative_prompt"],
                output_path=output_path,
                sch=base_config["scheduler"],
                num_inference_steps=base_config["num_inference_steps"],
                guidance_scale=base_config["guidance_scale"]
            )
            elapsed = time.time() - start_time
            print(f"Generated in {elapsed:.2f}s → {output_path}")

def generate_image(model_id: str, prompt: str, negative_prompt: str, output_path: str, sch=True, num_inference_steps=40, guidance_scale=4.5):
    pipe = get_pipeline(model_id, sch)

    if isinstance(pipe, StableDiffusion3Pipeline) or isinstance(pipe, StableDiffusionXLPipeline):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    else:
        pipe.enable_attention_slicing()
        image = pipe(prompt).images[0]

    image.save(output_path)

if __name__ == "__main__":
    generate_images()
    print(f"\nAll images saved to {OUTPUT_DIR}/")