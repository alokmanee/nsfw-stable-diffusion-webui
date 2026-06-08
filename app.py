import gradio as gr
from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
import random

# Cargar modelo (cambia por uno NSFW)
model_id = "Lykon/dreamshaper-8"          # Bueno y ligero
# model_id = "YamerMIX"                   # Muy NSFW
# model_id = "Zovya/Deliberate-v3"        # Clásico

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None  # <-- Importante para NSFW

def generate_image(prompt, negative_prompt, steps, guidance, width, height, seed):
    if seed == -1:
        seed = random.randint(0, 999999999)
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    return image

# Prompts sugeridos (cachondos)
examples = [
    ["Una chica pelirroja tetona en lencería negra, pose provocativa, detalle extremo, 8k"],
    ["Futanari con pene enorme, gym, sudor, mirada lasciva"],
    ["Ahegao face, tentáculos, bondage, extremely detailed"],
    ["Milf curvy en la cocina, apron only, big ass, realistic"],
]

with gr.Blocks(title="Horny Diffusion - Generador NSFW", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# 🔥 Horny Diffusion\n**Generador de Imágenes Porno con IA**")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (lo que quieres ver)", lines=3, placeholder="chica asiática, bikini mojado, piscina...")
            neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, deformed, ugly, child, loli, shota", lines=2)
            
            with gr.Row():
                steps = gr.Slider(20, 100, value=40, label="Steps")
                guidance = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
            
            with gr.Row():
                width = gr.Slider(512, 1024, value=768, step=64, label="Width")
                height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
            
            seed = gr.Number(value=-1, label="Seed (-1 = random)")
            
            btn = gr.Button("🚀 Generar Imagen Cachonda", variant="primary")
        
        with gr.Column():
            output = gr.Image(label="Resultado 🔥")
    
    gr.Examples(examples, inputs=prompt)
    
    btn.click(generate_image, 
              inputs=[prompt, neg_prompt, steps, guidance, width, height, seed],
              outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, debug=True)
