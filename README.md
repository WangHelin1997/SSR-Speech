# SSR-Speech

Official Pytorch implementation of the paper: SSR-Speech: Towards Stable, Safe and Robust Zero-shot Speech Editing and Synthesis.

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

conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
mfa model download dictionary mandarin_china_mfa
mfa model download acoustic mandarin_mfa
```

## Pretrained Models

Download our pretrained models from [huggingface](https://huggingface.co/westbrook/voicecraft).
We provide MFA models, an Encodec model and pretrained Chinese and English models.

After downloading the files, put them under this repo, like:
```
SSR-Speech/
    -data/
    -demo/
    -pretrained_models/
    ....
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
Here, `AUDIO_PATH` is the path where the json file was saved, `SAVE_DIR` is the path where the processed data will be saved, `ENCODEC_PATH` is the path of a pretrained encodec model and `DATA_NAME` is the saved name of the dataset.

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

## Inference
For Mandarin speech editing test, please run:

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_se_zh.py
```

For English speech editing test, please run:

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_se_en.py
```

For Mandarin zero-shot tts test, please run:

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_tts_zh.py
```

For English zero-shot tt test, please run:

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_tts_en.py
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

3. start training,

```bash
dora run -d solver='compression/encodec_audiogen_16khz' dset='internal/sounds_16khz'
```


## Acknowledgement
We thank Puyuan for his [VoiceCraft](https://github.com/jasonppy/VoiceCraft).

