<p align="center">
  <img src="demo/ssrspeech.webp" alt="SSR-Speech" width="300" height="300" style="max-width: 100%;">
</p>


<!-- [![Paper](https://img.shields.io/badge/arXiv-2409.07556-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2409.07556)  [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/) [![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/westbrook/SSR-Speech-English)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g4-Oqd1Fu9WfDFb-nicfxqsWIPvsGb91?usp=sharing)  [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://wanghelin1997.github.io/SSR-Speech-Demo/) -->

[![Paper](https://img.shields.io/badge/arXiv-2409.07556-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2409.07556)  [![Mandarin Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/westbrook/SSR-Speech-Mandarin) [![English Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/westbrook/SSR-Speech-English)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g4-Oqd1Fu9WfDFb-nicfxqsWIPvsGb91?usp=sharing)  [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://wanghelin1997.github.io/SSR-Speech-Demo/)

Official Pytorch implementation of the paper: SSR-Speech: Towards Stable, Safe and Robust Zero-shot Speech Editing and Synthesis.

:star: Work done during an internship at Tencent AI Lab

## TODO
- [x] Release English model weights
- [x] Release Mandarin model weights
- [ ] HuggingFace Spaces demo
- [ ] Fix gradio app
- [x] arxiv paper
- [x] WhisperX forced alignment
- [x] ASR for automatically transcipt the prompt for TTS
- [x] Simplify the inference stage


## Environment setup
```bash
conda create -n ssr python=3.9.16
conda activate ssr

pip install git+https://github.com/WangHelin1997/SSR-Speech.git#subdirectory=audiocraft
pip install xformers==0.0.22
pip install torchaudio torch
apt-get install ffmpeg
apt-get install espeak-ng
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2

# only use for inference
pip install gradio==3.50.2
pip install nltk>=3.8.1
pip install openai-whisper>=20231117
pip install whisperx==3.1.5
pip install faster-whisper==1.0.0
pip install num2words==0.5.13
pip install opencc-python-reimplemented
```

<!-- ```bash
conda create -n ssr python=3.9.16
conda activate ssr

pip install git+https://github.com/WangHelin1997/SSR-Speech.git#subdirectory=audiocraft

pip install xformers==0.0.22
pip install torchaudio torch
apt-get install ffmpeg
apt-get install espeak-ng
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2

# only use for inference
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
mfa model download dictionary mandarin_china_mfa
mfa model download acoustic mandarin_mfa
``` -->

## Pretrained Models

Download our pretrained English models from [huggingface](https://huggingface.co/westbrook/SSR-Speech-English).
We provide an Watemark Encodec model, a pretrained English model on GigaSpeech XL set, and a pretrained Mandarin model on internal data (25,000 hours).

After downloading the files, put them under this repo, like:
```
SSR-Speech/
    -data/
    -demo/
    -pretrained_models/
    ....
```

## Gradio
### Run in colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g4-Oqd1Fu9WfDFb-nicfxqsWIPvsGb91?usp=sharing)

### Run locally
After environment setup, run gradio server from terminal:
```bash
python gradio_app.py
```
It is ready to use on [default url](http://127.0.0.1:7860).

### How to use it
1. (optionally) Select models
2. Load models
3. Transcribe
4. Align
5. Run

## Inference examples

For English speech editing test, please run:

```bash
python inference_v2.py  \
    --seed 2024 \
    --sub_amount 0.12 \
    --aug_text \
    --use_watermark \
    --language 'en' \
    --model_path "./pretrained_models/English.pth" \
    --codec_path "./pretrained_models/wmencodec.th" \
    --orig_audio "./demo/84_121550_000074_000000.wav" \
    --target_transcript "But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any marks," \
    --temp_folder "./demo/temp" \
    --output_dir "./demo/generated_se" \
    --savename "84_121550_000074_00000" \
    --whisper_model_name "base.en"
```

<!-- ```bash
python inference.py  \
    --use_downloaded_mfa \
    --mfa_dict_path "./pretrained_models/english_us_arpa.dict" \
    --mfa_path "./pretrained_models/english_us_arpa.zip" \
    --seed 2024 \
    --sub_amount 0.12 \
    --top_p 0.8 \
    --stop_repetition 2 \
    --sample_batch_size 1 \
    --cfg_coef 1.5 \
    --aug_text \
    --use_watermark \
    --language 'en' \
    --model_path "./pretrained_models/English_10k/e830M/best_bundle.pth" \
    --codec_path "./pretrained_models/WMEncodec/checkpoint.th" \
    --orig_audio "./demo/84_121550_000074_000000.wav" \
    --orig_transcript "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks," \
    --target_transcript "But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any marks," \
    --temp_folder "./demo/temp"\
    --output_dir "./demo/generated_se"\
    --savename "84_121550_000074_00000"
``` -->

<!-- For Mandarin zero-shot TTS test, please run:

```bash
python inference.py  \
    --use_downloaded_mfa \
    --mfa_dict_path "./pretrained_models/mandarin_china_mfa.dict" \
    --mfa_path "./pretrained_models/mandarin_mfa.zip" \
    --seed 2024 \
    --sub_amount 0.01 \
    --top_p 0.8 \
    --stop_repetition 2 \
    --sample_batch_size 1 \
    --cfg_coef 1.5 \
    --aug_text \
    --use_watermark \
    --tts \
    --language 'zh' \
    --model_path "./pretrained_models/Chinese_25k/e830M/best_bundle.pth" \
    --codec_path "./pretrained_models/WMEncodec/checkpoint.th" \
    --orig_audio "./demo/pony.wav" \
    --orig_transcript "能够更有效率地结合给用户提升更多的这种体验也包括他的这个他的后台的效率提升等等我相信这些额额业界的解决方案应该说是" \
    --target_transcript "能够更有效率地结合给用户提升更多的这种体验在游戏业务的强势反弹下，腾讯在二季度拿出了一份十分亮眼的成绩单" \
    --temp_folder "./demo/temp"\
    --output_dir "./demo/generated_tts"\
    --savename "pony"
``` -->

For English zero-shot TTS test, please run:

```bash
python inference_v2.py  \
    --seed 2024 \
    --tts \
    --aug_text \
    --use_watermark \
    --language 'en' \
    --model_path "./pretrained_models/English.pth" \
    --codec_path "./pretrained_models/wmencodec.th" \
    --orig_audio "./demo/5895_34622_000026_000002.wav" \
    --prompt_length 3 \
    --target_transcript "I cannot believe that the same model can also do text to speech synthesis too!" \
    --temp_folder "./demo/temp" \
    --output_dir "./demo/generated_tts" \
    --savename "5895_34622_000026_000002" \
    --whisper_model_name "base.en"
```

For Mandarin speech editing test, please run:

```bash
python inference_v2.py  \
    --seed 2024 \
    --sub_amount 0.12 \
    --aug_text \
    --use_watermark \
    --language 'zh' \
    --model_path "./pretrained_models/Mandarin.pth" \
    --codec_path "./pretrained_models/wmencodec.th" \
    --orig_audio "./demo/aishell3_test.wav" \
    --target_transcript "食品价格以基本都在一万到两万之间" \
    --temp_folder "./demo/temp" \
    --output_dir "./demo/generated_se" \
    --savename "aishell3_test" \
    --whisper_model_name "base"
```

For Mandarin zero-shot TTS test, please run:

```bash
python inference_v2.py  \
    --seed 2024 \
    --tts \
    --aug_text \
    --use_watermark \
    --language 'zh' \
    --model_path "./pretrained_models/Mandarin.pth" \
    --codec_path "./pretrained_models/wmencodec.th" \
    --orig_audio "./demo/aishell3_test.wav" \
    --prompt_length 3 \
    --target_transcript "我简直不敢相信同一个模型也可以进行文本到语音的合成" \
    --temp_folder "./demo/temp" \
    --output_dir "./demo/generated_tts" \
    --savename "aishell3_test" \
    --whisper_model_name "base"
```


## Training
To train an SSR-Speech model, you need to prepare the following parts:
1. Prepare a json file saving data in the following format (including utterances and their transcripts):
```
{
"segment_id": "YOU1000000012_S0000106",
"wav": "/data/gigaspeech/wavs/xl/YOU1000000012/YOU1000000012_S0000106.wav",
"trans": "then you can look at o b s or wirecast as a professional solution then. if you're on a mac and you're looking for a really cheap and easy way to create a professional live stream.",
"duration": 9.446044921875
}
```

2. Encode the utterances into codes using e.g. Encodec. Run:

```bash
export CUDA_VISIBLE_DEVICES=0
cd ./data
AUDIO_PATH=''
SAVE_DIR=''
ENCODEC_PATH=''
DATA_NAME=''
python encode.py \
--dataset_name ${DATA_NAME} \
--audiopath ${AUDIO_PATH} \
--save_dir ${SAVE_DIR} \
--encodec_model_path ${ENCODEC_PATH} \
--batch_size 32 \
--start 0 \
--end 10000000
```
Here, `AUDIO_PATH` is the path where the json file was saved, `SAVE_DIR` is the path where the processed data will be saved, `ENCODEC_PATH` is the path of a pretrained encodec model and `DATA_NAME` is the saved name of the dataset. Here the `start` and `end` indexes are used for multi-gpu processing.

3. Convert transcripts into phoneme sequence. Run:

```bash
AUDIO_PATH=''
SAVE_DIR=''
DATA_NAME=''
python phonemize.py \
--dataset_name ${DATA_NAME} \
--dataset_dir ${AUDIO_PATH} \
--save_dir ${SAVE_DIR}
```
Add `language='cmn'` in Line 47 (`phonemize.py`) when you process Mandarin.

4. Prepare manifest (i.e. metadata). Run:

```bash
AUDIO_PATH=''
SAVE_DIR=''
DATA_NAME=''
python filemaker.py \
--dataset_name ${DATA_NAME} \
--dataset_dir ${AUDIO_PATH} \
--save_dir ${SAVE_DIR}
```

5. Prepare a phoneme set (we named it vocab.txt)

```bash
SAVE_DIR=''
DATA_NAME=''
python vocab.py \
--dataset_name ${DATA_NAME} \
--save_dir ${SAVE_DIR}
```


Now, you are good to start training!

```bash
cd ./z_scripts
bash e830M.sh
```

If your dataset introduce new phonemes (which is very likely) that doesn't exist in the giga checkpoint, make sure you combine the original phonemes with the phoneme from your data when construction vocab. And you need to adjust `--text_vocab_size` and `--text_pad_token` so that the former is bigger than or equal to you vocab size, and the latter has the same value as `--text_vocab_size` (i.e. `--text_pad_token` is always the last token). From our experience, you can set `--text_vocab_size` to `100` for an English model and `200` for a Mandarin model.


## Training WaterMarking Encodec

To train the Watermarking Encodec, you need to:

1. install our audiocraft package,

```bash
cd ./audiocraft
pip install -e .
```

2. prepare data (for training, validataion and test), e.g.

```bash
python makefile.py
```

3. change the settings in `./audiocraft/config/` to your own and start training,

```bash
dora run -d solver='compression/encodec_audiogen_16khz' dset='internal/sounds_16khz'
```

## License
The codebase is under [MIT LICENSE](./LICENSE). Note that we use some of the code from other repository that are under different licenses: `./models/modules`, `./steps/optim.py`, `data/tokenizer.py` are under Apache License, Version 2.0; the phonemizer we used is under GNU 3.0 License.



## Acknowledgement
We thank Puyuan for his [VoiceCraft](https://github.com/jasonppy/VoiceCraft).


## Citation
```
@article{wang2024ssrspeech,
  author    = {Wang, Helin and Yu, Meng and Hai, Jiarui and Chen, Chen and Hu, Yuchen and Chen, Rilin and Dehak, Najim and Yu, Dong},
  title     = {SSR-Speech: Towards Stable, Safe and Robust Zero-shot Text-based Speech Editing and Synthesis},
  journal   = {arXiv},
  year      = {2024},
}
```
