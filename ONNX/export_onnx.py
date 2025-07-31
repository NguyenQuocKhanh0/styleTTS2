"""
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/base_model.pth -O Models/base_model.pth
wget https://huggingface.co/dangtr0408/StyleTTS2-lite/resolve/main/Models/config.yaml -O Configs/config.yaml

uv sync --extra onnx --extra demo
uv run ONNX/export_onnx.py
"""
import torch
from inference_onnx import StyleTTS2, Preprocess
import sys
import phonemizer
from pathlib import Path

if sys.platform in ['win32', 'darwin']:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    import espeakng_loader
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

def get_phoneme(text, lang):
    try:
        my_phonemizer = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True,  with_stress=True, language_switch='remove-flags')
        return my_phonemizer.phonemize([text])[0]
    except Exception as e:
        print(e)

#Model path
config_path = str(Path("Configs") / "config.yaml")
models_path = str(Path('Models') / 'Finetune/base_model.pth')

#Inputs
speaker = {
    "path": "Demo/Audio/1_heart.wav",             #Ref audio path
    "speed": torch.tensor(1.0, dtype=torch.float32),#Speaking speed
}
text = 'Nearly 300 scholars currently working in the United States have applied for positions at Aix Marseille University in France.'

def export():
    with torch.no_grad():
        model = StyleTTS2(config_path, models_path)
        preprocess = Preprocess(config_path, models_path)

        phonemes = get_phoneme(text=text, lang="en-us")
        tokens, mel, speed = preprocess.preprocess_input(phonemes, speaker)
        style              = preprocess.get_style(mel)
        model_inputs = {
            "tokens": tokens,
            "style": style,
            "speed": speed,
        }
        input_names = ['tokens', 'style', 'speed']
        torch.onnx.export(model, kwargs=model_inputs,
                                f='model.onnx', 
                                dynamo=False, export_params=True, report=False, verify=True,
                                input_names=input_names,
                                output_names=["output_wav"],
                                training=torch.onnx.TrainingMode.EVAL,
                                opset_version=19,
                                dynamic_axes={
                                    'tokens':{0: "num_token"},
                                    'output_wav':{0: "wav_length"}
                                }
                        )

    print('Onnx export complete! ✅')

if __name__ == "__main__":
    export()