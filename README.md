# SSR-Speech
[![Paper](https://img.shields.io/badge/arXiv-2403.16973-brightgreen.svg?style=flat-square)](https://arxiv.org/)  [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IOjpglQyMTO2C3Y94LD9FY0Ocn-RJRg6?usp=sharing)  [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://wanghelin1997.github.io/SSR-Speech-Demo/)

Official Pytorch implementation of the paper: SSR-Speech: Towards Stable, Safe and Robust Zero-shot Speech Editing and Synthesis.

:star: Work done during an internship at Tencent AI Lab

## TODO
- [x] Release English model weights
- [ ] Release Mandarin model weights
- [ ] HuggingFace Spaces demo


## Environment setup
```bash
conda create -n ssr python=3.9.16
conda activate ssr

cd ./audiocraft
pip install .

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
```

## Pretrained Models

Download our pretrained English models from [huggingface](https://huggingface.co/westbrook/SSR-Speech-English).
We provide MFA models, an Watemark Encodec model and a pretrained English model on GigaSpeech XL set.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IOjpglQyMTO2C3Y94LD9FY0Ocn-RJRg6?usp=sharing)

### Run locally
After environment setup install additional dependencies:
```bash
apt-get install -y espeak espeak-data libespeak1 libespeak-dev
apt-get install -y festival*
apt-get install -y build-essential
apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools
apt-get install -y libxml2-dev libxslt-dev zlib1g-dev
pip install -r gradio_requirements.txt
```

Run gradio server from terminal or [`gradio_app.ipynb`](./gradio_app.ipynb):
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

## Inference examples
For Mandarin speech editing test, please run:

```bash
python inference.py  \
    --use_downloaded_mfa \
    --mfa_dict_path "./pretrained_models/mandarin_china_mfa.dict" \
    --mfa_path "./pretrained_models/mandarin_mfa.zip" \
    --seed 2024 \
    --sub_amount 0.12 \
    --top_p 0.8 \
    --stop_repetition 2 \
    --sample_batch_size 1 \
    --cfg_coef 1.5 \
    --aug_text \
    --use_watermark \
    --language 'zh' \
    --model_path "./pretrained_models/Chinese_25k/e830M/best_bundle.pth" \
    --codec_path "./pretrained_models/WMEncodec/checkpoint.th" \
    --orig_audio "./demo/pony.wav" \
    --orig_transcript "能够更有效率地结合给用户提升更多的这种体验也包括他的这个他的后台的效率提升等等我相信这些额额业界的解决方案应该说是" \
    --target_transcript "能够更有效率地结合给用户提升更多的体验也包括他的这个他的后台的效率提升等等在这个基础上我相信这些额额业界的解决方案应该说是" \
    --temp_folder "./demo/temp"\
    --output_dir "./demo/generated_se"\
    --savename "pony"
```

For English speech editing test, please run:

```bash
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
```

For Mandarin zero-shot TTS test, please run:

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
```

For English zero-shot TTS test, please run:

```bash
python inference.py  \
    --use_downloaded_mfa \
    --mfa_dict_path "./pretrained_models/english_us_arpa.dict" \
    --mfa_path "./pretrained_models/english_us_arpa.zip" \
    --seed 2024 \
    --sub_amount 0.01 \
    --top_p 0.8 \
    --stop_repetition 2 \
    --sample_batch_size 1 \
    --cfg_coef 1.5 \
    --aug_text \
    --use_watermark \
    --tts \
    --language 'en' \
    --model_path "./pretrained_models/English_10k/e830M/best_bundle.pth" \
    --codec_path "./pretrained_models/WMEncodec/checkpoint.th" \
    --orig_audio "./demo/5895_34622_000026_000002.wav" \
    --orig_transcript "Gwynplaine had, besides, for his work and for his feats of strength, round his neck and over his shoulders, an esclavine of leather." \
    --target_transcript "Gwynplaine had, besides, for his work and for his feats of strength, I cannot believe that the same model can also do text to speech synthesis too!" \
    --temp_folder "./demo/temp"\
    --output_dir "./demo/generated_tts"\
    --savename "5895_34622_000026_000002"
```

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


## Acknowledgement
We thank Puyuan for his [VoiceCraft](https://github.com/jasonppy/VoiceCraft).

