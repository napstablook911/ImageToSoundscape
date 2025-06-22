import streamlit as st
import io
from PIL import Image
import soundfile as sf
import librosa
import numpy as np
import torch # Importa torch
import sys
sys.setrecursionlimit(2000) # Aumentiamo il limite di ricorsione

# --- Configurazione del Dispositivo ---
# Questo rileva automaticamente se MPS (GPU Apple Silicon) è disponibile
# Per ora, useremo la CPU come fallback se MPS è problematico per Stable Audio
device = "mps" if torch.backends.mps.is_available() else "cpu"
# ******************** MODIFICA QUI: Forza device = "cpu" ********************
# Per superare i problemi di Stable Audio su MPS con float16/float32
# FORZA LA CPU PER TUTTI I MODELLI, per semplicità.
# Se la caption genera velocemente, potremmo tornare indietro e mettere il modello vit_gpt2 su MPS
device = "cpu"
# **************************************************************************
st.write(f"Utilizzo del dispositivo: {device}")


# --- 1. Caricamento dei Modelli AI (spostati qui, fuori dalle funzioni Streamlit) ---
@st.cache_resource
def load_models():
    # Caricamento del modello per la captioning (ViT-GPT2)
    from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModelForVision2Seq
    st.write("Caricamento del modello ViT-GPT2 per la captioning dell'immagine...")

    vit_gpt2_feature_extractor = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_gpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Questo modello andrà sulla CPU
    vit_gpt2_model = AutoModelForVision2Seq.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)

    st.write("Modello ViT-GPT2 caricato.")

    # Caricamento del modello Text-to-Audio (Stable Audio Open - 1.0)
    from diffusers import DiffusionPipeline
    st.write("Caricamento del modello Stable Audio Open - 1.0 per la generazione del soundscape...")
    # ******************** MODIFICA QUI ********************
    # Assicurati che non ci sia torch_dtype=torch.float16 e che vada sulla CPU
    stable_audio_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", force_download=True).to(device) 
    # ******************************************************
    st.write("Modello Stable Audio Open 1.0 caricato.")

    return vit_gpt2_feature_extractor, vit_gpt2_model, vit_gpt2_tokenizer, stable_audio_pipeline

# Carica i modelli all'avvio dell'app
vit_gpt2_feature_extractor, vit_gpt2_model, vit_gpt2_tokenizer, stable_audio_pipeline = load_models()


# --- 2. Funzioni della Pipeline ---
def generate_image_caption(image_pil):
    pixel_values = vit_gpt2_feature_extractor(images=image_pil.convert("RGB"), return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device) # Sposta input su CPU
    
    # Token di inizio per GPT-2, assicurandosi che sia su CPU
    # Ottieni il decoder_start_token_id dal modello o dal tokenizer
    if hasattr(vit_gpt2_model.config, "decoder_start_token_id"):
        decoder_start_token_id = vit_gpt2_model.config.decoder_start_token_id
    else:
        if vit_gpt2_tokenizer.pad_token_id is not None:
            decoder_start_token_id = vit_gpt2_tokenizer.pad_token_id
        else:
            decoder_start_token_id = 50256 # Default GPT-2 EOS token

    # Crea un input_ids iniziale con il decoder_start_token_id e spostalo su CPU
    input_ids = torch.ones((1, 1), device=device, dtype=torch.long) * decoder_start_token_id


    output_ids = vit_gpt2_model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        max_length=50,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    caption = vit_gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


def generate_soundscape_from_caption(caption: str, duration_seconds: int = 10):
    st.write(f"Generazione soundscape per: '{caption}' (durata: {duration_seconds}s)")
    with st.spinner("Generazione audio in corso..."):
        try:
            # Assicurati che il modello sia già su CPU dal caricamento
            audio_output = stable_audio_pipeline(
                prompt=caption,
                audio_end_in_s=duration_seconds 
            ).audios

            audio_data = audio_output[0].cpu().numpy() 
            sample_rate = stable_audio_pipeline.sample_rate

            audio_data = audio_data.astype(np.float32)
            audio_data = librosa.util.normalize(audio_data)

            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            buffer.seek(0)
            return buffer.getvalue(), sample_rate

        except Exception as e:
            st.error(f"Errore durante la generazione dell'audio: {e}")
            return None, None


# --- 3. Interfaccia Streamlit ---
st.title("Generatore di Paesaggi Sonori da Immagini")
st.write("Carica un'immagine e otterrai una descrizione testuale e un paesaggio sonoro generato!")

uploaded_file = st.file_uploader("Scegli un'immagine...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Immagine Caricata.', use_column_width=True)
    st.write("")

    audio_duration = st.slider("Durata audio (secondi):", 5, 30, 10, key="audio_duration_slider")


    if st.button("Genera Paesaggio Sonoro"):
        st.subheader("Processo in Corso...")

        # PASSO 1: Genera la caption
        st.write("Generazione della caption...")
        caption = generate_image_caption(input_image)
        st.write(f"Caption generata: **{caption}**")

        # PASSO 2: Genera il soundscape
        st.write("Generazione del paesaggio sonoro...")
        audio_data_bytes, sample_rate = generate_soundscape_from_caption(caption, duration_seconds=audio_duration)

        if audio_data_bytes is not None:
            st.subheader("Paesaggio Sonoro Generato")
            st.audio(audio_data_bytes, format='audio/wav', sample_rate=sample_rate)

            st.download_button(
                label="Scarica Audio WAV",
                data=audio_data_bytes,
                file_name="paesaggio_sonoro_generato.wav",
                mime="audio/wav"
            )
        else:
            st.error("La generazione del paesaggio sonoro è fallita.")