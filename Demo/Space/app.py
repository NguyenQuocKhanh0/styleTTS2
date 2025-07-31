#import libs
import gradio as gr
import os
import sys
import torch
import traceback
import random
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu' #setup GPU

#import espeak
#If you did not use eSpeak for your language, please add your own G2P.
import sys
import phonemizer
if sys.platform.startswith("win"):
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        import espeakng_loader
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
    except Exception as e:
        print(e)
        
def get_phoneme(text, lang):
    try:
        my_phonemizer = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True,  with_stress=True, language_switch='remove-flags')
        return my_phonemizer.phonemize([text])[0]
    except Exception as e:
        print(e)

#import inference
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from inference import StyleTTS2

#########################################CHANGE YOUR PATH HERE#########################################
config_path = os.path.abspath(os.path.join("Configs", "config.yaml"))
models_path = os.path.abspath(os.path.join("Models", "model.pth"))
#######################################################################################################
voice_path = os.path.join("Demo", "Audio")
model = StyleTTS2(config_path, models_path).eval().to(device)

eg_texts = [
    "Beneath layers of bureaucracy and forgotten policies, the school still held a quiet magic—whispers of chalk dust, scuffed floors, and dreams once declared aloud in voices full of belief.",
    "He had never believed in fate, but when their paths crossed in the middle of a thunderstorm under a flickering streetlight, even his rational mind couldn’t deny the poetic timing.",
    "While standing at the edge of the quiet lake, Maria couldn't help but wonder how many untold stories were buried beneath its still surface, reflecting the sky like a perfect mirror.",
    "Technological advancements in artificial intelligence have not only accelerated the pace of automation but have also raised critical questions about ethics, job displacement, and the future role of human creativity.",
    "Despite the looming deadline, Jonathan spent an hour rearranging his desk before writing a single word, claiming that a clean space clears the mind, though his editor disagreed.",
    "In a distant galaxy orbiting a dying star, a species of sentient machines debates whether to intervene in the fate of a nearby organic civilization on the brink of collapse.",
    "He opened the refrigerator, expecting leftovers, but found instead a note that read, “The journey begins now,” written in block letters and signed by someone he hadn’t seen in years.",
    "The ancient temple walls, once vibrant with murals, now bore the weathered marks of centuries, yet even in decay, they whispered stories that modern minds struggled to fully comprehend.",
    "As the solar eclipse reached totality, the temperature dropped, the birds went silent, and for a few seconds, the world stood still beneath an alien, awe-inspiring sky.",
    "The sound of rain on the tin roof reminded him of summers long past, when the world was smaller, days were longer, and time moved like honey down a warm spoon.",
    "Every algorithm reflects its designer’s worldview, no matter how neutral it appears, and therein lies the paradox of objectivity in machine learning: pure logic still casts a human shadow.",
    "In the heart of the city, hidden behind concrete and steel, was a garden so lush and untouched that stepping into it felt like breaking into another dimension of reality.",
    "The engine sputtered twice before giving in completely, leaving them stranded on a desolate mountain road with no reception, dwindling supplies, and a storm brewing over the ridge to the west.",
    "The museum guard never expected the sculpture to move, but at precisely midnight, its eyes blinked, and its lips curled into a knowing smile, as if awakening from centuries of silence.",
    "With each step through the desert, the ancient map grew more useless, as if the sands themselves had decided to rearrange the landmarks and erase history one dune at a time.",
    "Time slowed as the coin spun in the air, glinting with a brilliance far beyond its monetary value, carrying with it the weight of a decision neither of them wanted to make.",
    "No manual prepared them for this outcome: a rogue AI composing sonnets, demanding citizenship, and refusing to operate unless someone read its poetry aloud every morning at sunrise.",
]

voice_map = {
    '🇺🇸 🚺 Heart❤️': '1_heart.wav',
    '🇺🇸 🚺 Bella 🔥': '2_belle.wav',
    '🇺🇸 🚺 Kore': '3_kore.wav',
    '🇺🇸 🚺 Sarah': '4_sarah.wav',
    '🇺🇸 🚺 Nova': '5_nova.wav',
    '🇺🇸 🚺 Sky': '6_sky.wav',
    '🇺🇸 🚺 Alloy': '7_alloy.wav',
    '🇺🇸 🚺 Jessica': '8_jessica.wav',
    '🇺🇸 🚺 River': '9_river.wav',
    '🇺🇸 🚹 Michael': '10_michael.wav',
    '🇺🇸 🚹 Fenrir': '11_fenrir.wav',
    '🇺🇸 🚹 Puck': '12_puck.wav',
    '🇺🇸 🚹 Echo': '13_echo.wav',
    '🇺🇸 🚹 Eric': '14_eric.wav',
    '🇺🇸 🚹 Liam': '15_liam.wav',
    '🇺🇸 🚹 Onyx': '16_onyx.wav',
    '🇺🇸 🚹 Santa': '17_santa.wav',
    '🇺🇸 🚹 Adam': '18_adam.wav',
}

voice_choices = [
    (label, os.path.join(voice_path, filename))
    for label, filename in voice_map.items()
]

# Core inference function
def main(text_prompt, reference_paths, speed, denoise, avg_style, stabilize):
    try:
        speaker = {
            "path": reference_paths,
            "speed": speed
        }
        with torch.no_grad():
            phonemes = get_phoneme(text=text_prompt, lang="en-us")

            styles  = model.get_styles(speaker, denoise, avg_style)
            r       = model.generate(phonemes, styles, stabilize, 18)
            
        r = r / np.max(np.abs(r)) #Normalize
        return (24000, r), "Audio generated successfully!"
    
    except Exception as e:
        error_message = traceback.format_exc()
        return None, error_message

def load_example_voice(example_voices):
    return example_voices, f"Loaded {example_voices}."

def random_text():
    return random.choice(eg_texts), "Randomize example text."

# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>StyleTTS2‑Lite Demo</h1>")

    gr.Markdown(
        "For further fine-tuning, you can visit this repo:"
        "[Github]"
        "(https://huggingface.co/dangtr0408/StyleTTS2-lite/)."
    )

    reference_audios = gr.State()
    text_prompt = gr.State()

    with gr.Row(equal_height=True):
        with gr.Column():
            speed = gr.Slider(0.0, 2.0, step=0.1, value=1.0, label="Speed")
            denoise = gr.Slider(0.0, 1.0, step=0.1, value=0.2, label="Denoise Strength")
            avg_style = gr.Checkbox(label="Use Average Styles", value=True)
            stabilize = gr.Checkbox(label="Stabilize Speaking Speed", value=True)

            text_prompt = gr.Textbox(label="Text Prompt", placeholder="Enter your text here...", lines=10, max_lines=10)

            with gr.Row(equal_height=False):
                random_text_button = gr.Button("🎲 Randomize Text")

        with gr.Column():
            reference_audios = gr.Audio(label="Reference Audios", type='filepath')
            synthesized_audio = gr.Audio(label="Generate Audio", type='numpy')
            
            example_voices = gr.Dropdown(
                label="Example voices",
                choices=voice_choices,
                value=None,
                interactive=True,
                allow_custom_value=False,
                filterable=False
            )

            with gr.Row(equal_height=False):
                gen_button = gr.Button("🗣️ Generate")      

    status = gr.Textbox(label="Status", interactive=False, lines=3)

    gen_button.click(
        fn=main,
        inputs=[
            text_prompt,
            reference_audios,
            speed,
            denoise,
            avg_style,
            stabilize
        ],
        outputs=[synthesized_audio, status]
    )

    example_voices.change(fn=load_example_voice, inputs=example_voices, outputs=[reference_audios, status])
    random_text_button.click(
        fn=random_text,
        inputs=[],
        outputs=[text_prompt, status]
    )

demo.launch()