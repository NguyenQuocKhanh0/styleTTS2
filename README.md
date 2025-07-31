# StyleTTS2-lite

## Base Model
_If you have a better checkpoint, your contribution would be greatly appreciated!_

**English**:
- English checkpoint (LibriTTS 100,000 steps): [Download (MIT)](https://huggingface.co/dangtr0408/StyleTTS2-lite/tree/main/Models)  
- LibriTTS dataset: [Download (CC BY 4.0)](https://huggingface.co/datasets/dangtr0408/LibriTTS-clean-460/tree/main) 
- Demo: [StyleTTS2-lite-space](https://huggingface.co/spaces/dangtr0408/StyleTTS2-lite-space)

**Vietnamese**:
- Vietnamese checkpoint (viVoice 120,000 steps): [Download (CC BY NC 4.0 SA)](https://huggingface.co/dangtr0408/StyleTTS2-lite-vi/tree/main/Models)
- viVoice dataset: [Download (CC BY NC 4.0 SA)](https://huggingface.co/datasets/capleaf/viVoice) 
- Demo: [StyleTTS2-lite-vi-space](https://huggingface.co/spaces/dangtr0408/StyleTTS2-lite-vi-space)
- Extended to 189 tokens.

Model Component Parameter Summary

| Component              | Parameters   | Used in Inference |
|------------------------|--------------|--------------------|
| **Decoder**            | 54,289,492   | ✅ Yes             |
| **Predictor**          | 16,194,612   | ✅ Yes             |
| **Style Encoder**      | 13,845,440   | ✅ Yes             |
| **Text Encoder**       | 5,606,400    | ✅ Yes             |
| **Text Aligner (ASR)** | 7,865,252    | ❌ No              |
| **Pitch Extractor (JDC)** | 5,248,067  | ❌ No              |
| **mpd (Discriminator)**| 41,105,770   | ❌ No              |
| **msd (Discriminator)**| 280,902      | ❌ No              |
| **Total**              | **144,435,935** |                |


## How To Start Finetuning

**1. Install The requirements**
```bash
pip install .
```

**2. Download Model And Config File**

Download and place the base model in ***/Models/Finetune*** and the corresponding config file in ***/Configs***.

**3. Format Your Dataset.**

Format: filename.wav | transcription

For reference, see ***val.txt*** in [LibriTTS dataset](https://huggingface.co/datasets/dangtr0408/LibriTTS-clean-460/tree/main).

**4. (Optional) Extend The Token Set To Support Additional Languages**

If you plan to train on a new language with symbols not included in the original token set (see config file that comes with the pretrained you downloaded), follow these steps after completing step 2.

- Locate the ***extend.ipynb***, set the "extend_to" variable to the total number of symbols you want to support (including the new ones), then run the notebook. You may over-extend (i.e reserve extra slots beyond your current needs), but it's strongly recommended to only extend up to the actual number of new symbols you plan to use to avoid unnecessary memory usage or complexity.
- Find the extended weights in ***/Extend/New_Weights/***, replace the original weights with it.
- Add new symbols to the "_extend_list" in the config file. You may also want to set:
<pre lang="yaml">
load_only_params: true
#Prevent loading old optimizer state
</pre>
- ⚠️ Important: *Do not add any new symbols to the config file ***before running extend.ipynb***. This will lead to misalignment between the model and the symbol set.*

**5. Adjust Your Configs File**

For a single GPU with 24 GB VRAM, i find the following works well.
<pre lang="yaml">
batch_size: 2 
max_len: 310 # maximum number of frames
</pre>

Change the location of your dataset. For example:
<pre lang="yaml">
data_params:
  train_data: ../Data_Speech/LibriTTS/train.txt
  val_data: ../Data_Speech/LibriTTS/val.txt
  root_path: ../Data_Speech/
</pre>



**6. Start training**
```bash
python train.py
```

## Disclaimer  

**Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.**


## References

- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)

- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)

## License

**Code: MIT License**