import streamlit as st
import torch
import torchaudio
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import io # Per salvare l'audio in memoria per Streamlit

st.set_page_config(layout="wide")

st.title("Image Captioning and Soundscape Generation")

# Funzione per caricare i modelli e metterli in cache
@st.cache_resource
def load_models():
    # Imposta il dispositivo su "cpu" come da requisiti per lo Space
    device = "cpu"
    st.write(f"Utilizzo del dispositivo: {device}")

    # Caricamento del modello ViT-GPT2 per la captioning dell'immagine
    st.write("Caricamento del modello ViT-GPT2 per la captioning dell'immagine...")
    try:
        vit_gpt2_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir="/app/hf_cache")
        vit_gpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir="/app/hf_cache")
        vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir="/app/hf_cache").to(device)
        st.write("Modello ViT-GPT2 caricato.")
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello ViT-GPT2: {e}")
        st.stop() # Ferma l'app se il modello essenziale non carica

    # Caricamento del modello Stable Audio Open Small per la generazione del soundscape
    st.write("Caricamento del modello Stable Audio Open Small per la generazione del soundscape...")
    try:
        # Carica il modello Stable Audio usando stable_audio_tools
        stable_audio_model, stable_audio_config = get_pretrained_model("stabilityai/stable-audio-open-small", cache_dir="/app/hf_cache")
        stable_audio_model = stable_audio_model.to(device)
        st.write("Modello Stable Audio Open Small caricato.")
        return vit_gpt2_feature_extractor, vit_gpt2_model, vit_gpt2_tokenizer, stable_audio_model, stable_audio_config
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello Stable Audio Open Small: {e}")
        st.stop() # Ferma l'app se il modello essenziale non carica


# Carica i modelli all'avvio dell'app
vit_gpt2_feature_extractor, vit_gpt2_model, vit_gpt2_tokenizer, stable_audio_model, stable_audio_config = load_models()

# Funzione per generare la caption dell'immagine
def generate_caption(image_pil):
    pixel_values = vit_gpt2_feature_extractor(images=image_pil, return_tensors="pt").pixel_values
    output_ids = vit_gpt2_model.generate(pixel_values, max_new_tokens=16)
    caption = vit_gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Funzione per generare il soundscape
def generate_soundscape(prompt_text):
    sample_size = stable_audio_config["sample_size"]
    sample_rate = stable_audio_config["sample_rate"]
    
    # Assicurati che il modello sia sulla CPU per la generazione
    device = "cpu"
    
    conditioning = [{
      "prompt": prompt_text,
    }]

    # Genera audio
    with st.spinner("Generazione audio in corso... (potrebbe richiedere un po' di tempo)"):
        output = generate_diffusion_cond(
          stable_audio_model,
          conditioning=conditioning,
          sample_size=sample_size,
          device=device,
          steps=100, # Numero di step di diffusione (puoi renderlo configurabile)
          cfg_scale=7, # Scala di classifer-free guidance
          sigma_min=0.03,
          sigma_max=500,
          sampler_type="dpmpp-3m-sde" # Tipo di sampler
        )

    # Riorganizza il batch audio in una singola sequenza
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, converti in int16, e prepara per la riproduzione
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Salva l'audio in un buffer di memoria per Streamlit
    buffer = io.BytesIO()
    torchaudio.save(buffer, output, sample_rate, format="wav")
    return buffer.getvalue(), sample_rate

# Streamlit UI
uploaded_file = st.file_uploader("Carica un'immagine per la captioning:", type=["png", "jpg", "jpeg"])

caption = ""
if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Immagine caricata.", use_column_width=True)

    with st.spinner("Generazione della caption..."):
        caption = generate_caption(image)
        st.success(f"Caption generata: **{caption}**")

# Campo di input per il prompt del soundscape
st.header("Generazione Soundscape")
soundscape_prompt_input = st.text_input(
    "Inserisci un prompt per il soundscape (es. 'A gentle rain with thunder and distant birds'):",
    value=caption if caption else "A natural outdoor soundscape" # Pre-popola con la caption se disponibile
)

if st.button("Genera Soundscape Audio"):
    if soundscape_prompt_input:
        audio_bytes, sr = generate_soundscape(soundscape_prompt_input)
        st.audio(audio_bytes, format='audio/wav', sample_rate=sr)
    else:
        st.warning("Per favore, inserisci un prompt per generare il soundscape.")

st.info("Nota: La generazione del soundscape può richiedere un po' di tempo a seconda della complessità del prompt e delle risorse disponibili.")
