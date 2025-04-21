import torch
import time
import json
import argparse
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    BitsAndBytesConfig,
    SD3Transformer2DModel,
)
from transformers import T5EncoderModel

def get_pipeline(model_id: str, sch=True):
    """Devuelve el pipeline correcto según el tipo de modelo."""
    model_id_lower = model_id.lower()

    if "3.5" in model_id_lower or "stable-diffusion-3" in model_id_lower:
        if "large-turbo" in model_id_lower:
            print("→ Usando Stable Diffusion 3.5 Large Turbo pipeline (cuantizado 4 bits)")

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

            # if sch:
            #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            # else:
            #     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 

            return pipe
        else:
            print("→ Usando Stable Diffusion 3.5 pipeline normal")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )
        # if sch:
        #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # else:
        #     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    
    elif "xl" in model_id_lower:
        print("→ Usando Stable Diffusion XL pipeline")
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
        print("→ Usando Stable Diffusion clásico pipeline")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        if sch:
            print("Setting DPM")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            print("Setting DDIM")
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe.to("cuda")

def generate_image(model_id: str, prompt: str, negative_prompt: str = "", output_path: str = "output.png", sch=True, num_inference_steps=40, guidance_scale=4.5):
    pipe = get_pipeline(model_id, sch)

    # Configuración por modelo
    if isinstance(pipe, StableDiffusion3Pipeline) or isinstance(pipe, StableDiffusionXLPipeline):
        print("setting pipeline")
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
    print(f"✅ Imagen generada y guardada en: {output_path}")

def load_config(config_path):
    """Carga la configuración desde un archivo JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generador de imágenes con Stable Diffusion.")
    parser.add_argument('--config', type=str, required=True, help="Ruta al archivo de configuración (config.json).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Extraer los parámetros de la configuración
    model_id = config['model_id']
    prompt = config['prompt']
    negative_prompt = config.get('negative_prompt', "")
    output_path = config['output_path']
    scheduler = config['scheduler']
    num_inference_steps = config.get('num_inference_steps', 40)
    guidance_scale = config.get('guidance_scale', 4.5)

    # Generar la imagen usando los parámetros de configuración
    start_time = time.time()
    generate_image(model_id=model_id, prompt=prompt, negative_prompt=negative_prompt, output_path=output_path, sch=scheduler, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Tiempo de generación: {elapsed_time:.2f} segundos")

