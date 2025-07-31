"""
Export voicepack with:
    import numpy as np
    with open('heart.bin', 'wb') as fp:    
        np.save(fp, style.numpy())
uv run ONNX/with_voicepacks.py
"""
import onnxruntime
import soundfile as sf
import numpy as np

def tokenize(phonemes: str):
    pad = "$"
    punctuation = ';:,.!?¡¿—…"«»“” '
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    extend = "" # Add more if needed
    
    table = {}
    for i, c in enumerate(pad + punctuation + letters + letters_ipa + extend):
        table[c] = i
    
    tokens = []
    for c in phonemes:
        if c in table:
            tokens.append(table[c])
    # Pad with silence
    tokens = [0] + tokens + [0]
    return tokens
        
def main():
    sess = onnxruntime.InferenceSession("model.onnx")
    phonemes = "nˌɪɹli θɹˈiːhˈʌndɹɪd skˈɑːlɚz kˈɜːɹəntli wˈɜːkɪŋ ɪnðə juːnˈaɪɾᵻd stˈeɪts hæv ɐplˈaɪd fɔːɹ pəzˈɪʃənz æɾ ˈeɪks mɑːɹsˈeɪ jˌuːnɪvˈɜːsᵻɾi ɪn fɹˈæns."
    tokens = tokenize(phonemes)    
    sess = onnxruntime.InferenceSession("model.onnx")
    with open('heart.bin', 'rb') as fp:        
        style = np.load(fp)
        
    wav = sess.run(None, {
        "tokens": tokens,
        "style": style,
        "speed": np.array(1.0, dtype=np.float32)
    })[0]

    sf.write('audio.wav', wav, 24000)
    print('Created audio.wav')
    
main()