import os
import gradio as gr
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Konfiguration af JoyCaptioning model
model_name = "nlpconnect/vit-gpt2-image-captioning"  # Eksempelmodel, udskift med JoyCaptioning model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Indlæs modellen og processoren
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_caption(image):
    """
    Genererer en beskrivelse af et billede ved hjælp af JoyCaptioning modellen.
    
    Args:
        image (PIL.Image): Billedet der skal beskrives.
    
    Returns:
        str: Beskrivelsen af billedet.
    """
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    # Forbehandling af billedet
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generer caption
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return caption

def process_images(images):
    """
    Behandler en liste af billeder: genererer beskrivelser og gemmer både billede og beskrivelse.
    
    Args:
        images (list of files): Listen af uploadede billedfiler.
    
    Returns:
        str: Opsummering af de behandlede billeder.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for img in images:
        try:
            # Åbn billedet
            image = Image.open(img)
            
            # Generer beskrivelse
            caption = generate_caption(image)
            
            # Forbered filnavne
            base_name = os.path.splitext(os.path.basename(img.name))[0]
            image_extension = os.path.splitext(os.path.basename(img.name))[1]
            image_filename = f"{base_name}{image_extension}"
            caption_filename = f"{base_name}.txt"
            
            # Gem billedet i output mappen
            image_save_path = os.path.join(output_dir, image_filename)
            image.save(image_save_path)
            
            # Gem beskrivelsen i en tekstfil
            caption_save_path = os.path.join(output_dir, caption_filename)
            with open(caption_save_path, "w", encoding="utf-8") as f:
                f.write(caption)
            
            results.append(f"**{image_filename}**: {caption}")
        except Exception as e:
            results.append(f"**{img.name}**: Fejl - {str(e)}")
    
    return "\n".join(results)

# Definér Gradio interfacet
iface = gr.Interface(
    fn=process_images,
    inputs=gr.inputs.Image(type="file", label="Upload Billeder", multiple=True),
    outputs="markdown",
    title="Billedbeskrivelse med JoyCaptioning",
    description="Upload flere billeder, og få beskrivelser genereret ved hjælp af JoyCaptioning. Billeder og beskrivelser gemmes i 'output' mappen."
)

if __name__ == "__main__":
    iface.launch()
