import gradio as gr
import torch
from utils import load_pipeline, save_image, get_random_prompt, get_default_negative_prompt

# Cargar el modelo una sola vez
pipe = load_pipeline(model_id="Lykon/dreamshaper-8")   # Cambia aquí por tu favorito NSFW

def generate(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int
):
    if not prompt.strip():
        prompt = get_random_prompt()
    
    if seed == -1:
        seed = torch.randint(0, 999999999, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    
    path = save_image(image, prompt)
    return image, path

with gr.Blocks(title="🔥 Horny Diffusion", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# 🔥 Horny Diffusion\n**Tu generador de porno con IA**")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="📝 Prompt (describe lo que quieres ver)",
                lines=4,
                placeholder="chica pelirroja tetona, lencería transparente, pose provocativa...",
                value=get_random_prompt()
            )
            negative = gr.Textbox(
                label="🚫 Negative Prompt",
                value=get_default_negative_prompt(),
                lines=2
            )
            
            with gr.Row():
                steps = gr.Slider(20, 80, value=35, label="Steps")
                guidance = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
            
            with gr.Row():
                width = gr.Slider(512, 1152, value=768, step=64, label="Width")
                height = gr.Slider(512, 1152, value=1024, step=64, label="Height")
            
            seed = gr.Number(-1, label="Seed (-1 = aleatorio)")
            
            btn = gr.Button("🚀 GENERAR IMAGEN CACHONDA", variant="primary", size="large")
        
        with gr.Column(scale=2):
            output_img = gr.Image(label="Resultado 🔥")
            output_path = gr.Textbox(label="📁 Imagen guardada en:", interactive=False)
    
    gr.Markdown("### 💡 Tips: Usa `<lora:tu_lora:0.8>` en el prompt si tienes LoRAs en la carpeta `loras/`")
    
    btn.click(
        generate,
        inputs=[prompt, negative, steps, guidance, width, height, seed],
        outputs=[output_img, output_path]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
