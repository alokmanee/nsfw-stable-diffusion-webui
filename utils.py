import os
import random
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline

def load_pipeline(model_id: str = "Lykon/dreamshaper-8", use_safety=False):
    """Carga el pipeline con configuraciones NSFW"""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None if not use_safety else None
    )
    pipe = pipe.to("cuda")
    pipe.safety_checker = None
    pipe.enable_attention_slicing()  # Ahorra VRAM
    return pipe

def save_image(image, prompt: str):
    """Guarda la imagen generada en outputs/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(x for x in prompt[:50] if x.isalnum() or x in " -_")
    filename = f"{timestamp}_{safe_prompt[:30]}.png"
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path

def get_random_prompt():
    """Devuelve prompts cachondos aleatorios"""
    prompts = [
        "beautiful girl, massive breasts, wearing tiny micro bikini, seductive pose, bedroom, detailed skin, 8k",
        "futanari, huge cock, gym mirror selfie, sweat, muscular, ahegao",
        "thicc milf, apron only, bent over kitchen counter, big ass, realistic",
        "tentacle monster fucking cute anime girl, bondage, ahegao, cum inflation",
        "lesbian couple, scissoring, oiled bodies, passionate kiss, detailed pussy",
    ]
    return random.choice(prompts)

def get_default_negative_prompt():
    return "blurry, low quality, deformed, ugly, bad anatomy, extra limbs, child, loli, shota, text, watermark"
