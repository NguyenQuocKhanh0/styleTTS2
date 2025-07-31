"""
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/base_model.pth -O Models/Finetune/base_model.pth
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/config.yaml -O Configs/config.yaml

uv sync --extra demo
uv run Demo/infer.py
"""
import soundfile as sf
import sys
import torch
import numpy as np
from pathlib import Path
import phonemizer


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from inference import StyleTTS2


def phonemize(text, lang):
    if sys.platform in ['win32', 'darwin']:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        import espeakng_loader
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    phonemizer_instance = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True,  with_stress=True, language_switch='remove-flags')
    return phonemizer_instance.phonemize([text])[0]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #setup GPU
    config_path = str(Path("Configs") / "config.yaml")
    models_path = str(Path('Models') / 'Finetune/base_model.pth')
    model = StyleTTS2(config_path, models_path).eval().to(device)
    
    text = 'Nearly 300 scholars currently working in the United States have applied for positions at Aix Marseille University in France.'
    phonemes = phonemize(text=text, lang="en-us")
    speed = 1
    denoise = 0.2
    avg_style = True
    stabilize = True

    speaker = {
        "path": 'Demo/Audio/1_heart.wav',
        "speed": speed
    }

    with torch.no_grad():
        styles = model.get_styles(speaker, denoise, avg_style)
        r = model.generate(phonemes, styles, stabilize, 18)
        r = r / np.max(np.abs(r))  # Normalize

    sr = 24000
    sf.write("audio.wav", r, sr)
    print('Created audio.wav')
    
main()