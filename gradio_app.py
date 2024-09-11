import os
import re
from num2words import num2words
import gradio as gr
import torch
import torchaudio
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from edit_utils_zh import parse_edit_zh
from edit_utils_en import parse_edit_en
from edit_utils_zh import parse_tts_zh
from edit_utils_en import parse_tts_en
from inference_scale import inference_one_sample
import librosa
import soundfile as sf
from models import ssr
import io
import numpy as np
import random
import uuid
import nltk
nltk.download('punkt')

DEMO_PATH = os.getenv("DEMO_PATH", "./demo")
TMP_PATH = os.getenv("TMP_PATH", "./demo/temp")
MODELS_PATH = os.getenv("MODELS_PATH", "./pretrained_models")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model, align_model, ssrspeech_model = None, None, None
_whitespace_re = re.compile(r"\s+")

def get_random_string():
    return "".join(str(uuid.uuid4()).split("-"))


def seed_everything(seed):
    if seed != -1:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_mask_interval(transcribe_state, word_span):
    print(transcribe_state)
    seg_num = len(transcribe_state['segments'])
    data = []
    for i in range(seg_num):
      words = transcribe_state['segments'][i]['words']
      for item in words:
        data.append([item['start'], item['end'], item['word']])

    s, e = word_span[0], word_span[1]
    assert s <= e, f"s:{s}, e:{e}"
    assert s >= 0, f"s:{s}"
    assert e <= len(data), f"e:{e}"
    if e == 0: # start
        start = 0.
        end = float(data[0][0])
    elif s == len(data): # end
        start = float(data[-1][1])
        end = float(data[-1][1]) # don't know the end yet
    elif s == e: # insert
        start = float(data[s-1][1])
        end = float(data[s][0])
    else:
        start = float(data[s-1][1]) if s > 0 else float(data[s][0])
        end = float(data[e][0]) if e < len(data) else float(data[-1][1])

    return (start, end)

class WhisperxAlignModel:
    def __init__(self):
        from whisperx import load_align_model
        self.model, self.metadata = load_align_model(language_code="en", device=device)

    def align(self, segments, audio_path):
        from whisperx import align, load_audio
        audio = load_audio(audio_path)
        return align(segments, self.model, self.metadata, audio, device, return_char_alignments=False)["segments"]


class WhisperModel:
    def __init__(self, model_name):
        from whisper import load_model
        self.model = load_model(model_name, device)

        from whisper.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(multilingual=False)
        self.supress_tokens = [-1] + [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path, suppress_tokens=self.supress_tokens, word_timestamps=True)["segments"]


class WhisperxModel:
    def __init__(self, model_name, align_model: WhisperxAlignModel):
        from whisperx import load_model
        self.model = load_model(model_name, device, asr_options={"suppress_numerals": True, "max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None})
        self.align_model = align_model

    def transcribe(self, audio_path):
        segments = self.model.transcribe(audio_path, batch_size=8)["segments"]
        for segment in segments:
            segment['text'] = replace_numbers_with_words(segment['text'])
        return self.align_model.align(segments, audio_path)


def load_models(whisper_backend_name, whisper_model_name, alignment_model_name, ssrspeech_model_name):
    global transcribe_model, align_model, ssrspeech_model

    if ssrspeech_model_name == "English":
        ssrspeech_model_name = "English"
        text_tokenizer = TextTokenizer(backend="espeak")
    elif ssrspeech_model_name == "Mandarin":
        ssrspeech_model_name = "Mandarin"
        text_tokenizer = TextTokenizer(backend="espeak", language='cmn')

    if alignment_model_name is not None:
        align_model = WhisperxAlignModel()

    if whisper_model_name is not None:
        if whisper_backend_name == "whisper":
            transcribe_model = WhisperModel(whisper_model_name)
        else:
            if align_model is None:
                raise gr.Error("Align model required for whisperx backend")
            transcribe_model = WhisperxModel(whisper_model_name, align_model)

    ssrspeech_fn = f"{MODELS_PATH}/{ssrspeech_model_name}.pth"
    if not os.path.exists(ssrspeech_fn):
        os.system(f"wget https://huggingface.co/westbrook/SSR-Speech-{ssrspeech_model_name}/resolve/main/{ssrspeech_model_name}.pth -O " + ssrspeech_fn)

    ckpt = torch.load(ssrspeech_fn)
    model = ssr.SSR_Speech(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    config = model.args
    phn2num = ckpt["phn2num"]
    model.to(device)

    encodec_fn = f"{MODELS_PATH}/wmencodec.th"
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/wmencodec.th -O " + encodec_fn)

    ssrspeech_model = {
        "config": config,
        "phn2num": phn2num,
        "model": model,
        "text_tokenizer": text_tokenizer,
        "audio_tokenizer": AudioTokenizer(signature=encodec_fn)
    }
    return gr.Accordion()


def get_transcribe_state(segments):
    words_info = [word_info for segment in segments for word_info in segment["words"]]
    transcript = " ".join([segment["text"] for segment in segments])
    transcript = transcript[1:] if transcript[0] == " " else transcript
    return {
        "segments": segments,
        "transcript": transcript,
        "words_info": words_info,
        "transcript_with_start_time": " ".join([f"{word['start']} {word['word']}" for word in words_info]),
        "transcript_with_end_time": " ".join([f"{word['word']} {word['end']}" for word in words_info]),
        "word_bounds": [f"{word['start']} {word['word']} {word['end']}" for word in words_info]
    }


def transcribe(seed, audio_path):
    if transcribe_model is None:
        raise gr.Error("Transcription model not loaded")
    seed_everything(seed)

    segments = transcribe_model.transcribe(audio_path)
    state = get_transcribe_state(segments)
    success_message = "<span style='color:green;'>Success: Transcribe completed successfully!</span>"

    return [
        state["transcript"], state["transcript_with_start_time"], state["transcript_with_end_time"],
        state, success_message
    ]


def align_segments(transcript, audio_path):
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    import json
    config_string = 'task_language=eng|os_task_file_format=json|is_text_type=plain'

    tmp_transcript_path = os.path.join(TMP_PATH, f"{get_random_string()}.txt")
    tmp_sync_map_path = os.path.join(TMP_PATH, f"{get_random_string()}.json")
    with open(tmp_transcript_path, "w") as f:
        f.write(transcript)

    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(tmp_transcript_path)
    task.sync_map_file_path_absolute = os.path.abspath(tmp_sync_map_path)
    ExecuteTask(task).execute()
    task.output_sync_map_file()

    with open(tmp_sync_map_path, "r") as f:
        return json.load(f)


def align(seed, transcript, audio_path):
    if align_model is None:
        raise gr.Error("Align model not loaded")
    seed_everything(seed)
    transcript = replace_numbers_with_words(transcript).replace("  ", " ").replace("  ", " ")
    fragments = align_segments(transcript, audio_path)
    segments = [{
        "start": float(fragment["begin"]),
        "end": float(fragment["end"]),
        "text": " ".join(fragment["lines"])
    } for fragment in fragments["fragments"]]
    segments = align_model.align(segments, audio_path)
    state = get_transcribe_state(segments)
    success_message = "<span style='color:green;'>Success: Alignment completed successfully!</span>"

    return [
        state["transcript_with_start_time"], state["transcript_with_end_time"],
        state, success_message
    ]


def get_output_audio(audio_tensors, codec_audio_sr):
    result = torch.cat(audio_tensors, 1)
    buffer = io.BytesIO()
    torchaudio.save(buffer, result, int(codec_audio_sr), format="wav")
    buffer.seek(0)
    return buffer.read()

def replace_numbers_with_words(sentence):
    sentence = re.sub(r'(\d+)', r' \1 ', sentence) # add spaces around numbers
    def replace_with_words(match):
        num = match.group(0)
        try:
            return num2words(num) # Convert numbers to words
        except:
            return num # In case num2words fails (unlikely with digits but just to be safe)
    return re.sub(r'\b\d+\b', replace_with_words, sentence) # Regular expression that matches numbers

def run(seed, sub_amount, ssrspeech_model_choice, codec_audio_sr, codec_sr, top_k, top_p, temperature,
        stop_repetition, kvcache, silence_tokens, aug_text, cfg_coef,
        audio_path, transcribe_state, original_transcript, transcript,
        mode, selected_sentence, previous_audio_tensors):

    aug_text = True if aug_text == 1 else False
    if ssrspeech_model is None:
        raise gr.Error("ssrspeech model not loaded")
    
    # resample audio
    audio, _ = librosa.load(audio_path, sr=16000)
    sf.write(audio_path, audio, 16000)

    seed_everything(seed)
    transcript = replace_numbers_with_words(transcript).replace("  ", " ").replace("  ", " ") # replace numbers with words, so that the phonemizer can do a better job

    if mode == "Rerun":
        colon_position = selected_sentence.find(':')
        selected_sentence_idx = int(selected_sentence[:colon_position])
        sentences = [selected_sentence[colon_position + 1:]]
    else:
        sentences = [transcript.replace("\n", " ")]

    audio_tensors = []
    inference_transcript = ""
    for sentence in sentences:
        decode_config = {"top_k": top_k, "top_p": top_p, "temperature": temperature, "stop_repetition": stop_repetition,
                         "kvcache": kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr}
        
         # run the script to turn user input to the format that the model can take
        if mode == "Edit":
            operations, orig_spans = parse_edit_en(original_transcript, sentence) if ssrspeech_model_choice == 'English' else parse_edit_zh(original_transcript, sentence)
            print(operations)
            print("orig_spans: ", orig_spans)
            
            if len(orig_spans) > 3:
                raise gr.Error("Current model only supports maximum 3 editings")
                
            starting_intervals = []
            ending_intervals = []
            for orig_span in orig_spans:
                start, end = get_mask_interval(transcribe_state, orig_span)
                starting_intervals.append(start)
                ending_intervals.append(end)
        
            print("intervals: ", starting_intervals, ending_intervals)
        
            info = torchaudio.info(audio_path)
            audio_dur = info.num_frames / info.sample_rate
            
            def combine_spans(spans, threshold=0.2):
                spans.sort(key=lambda x: x[0])
                combined_spans = []
                current_span = spans[0]
        
                for i in range(1, len(spans)):
                    next_span = spans[i]
                    if current_span[1] >= next_span[0] - threshold:
                        current_span[1] = max(current_span[1], next_span[1])
                    else:
                        combined_spans.append(current_span)
                        current_span = next_span
                combined_spans.append(current_span)
                return combined_spans
            
            morphed_span = [[max(start - sub_amount, 0), min(end + sub_amount, audio_dur)]
                            for start, end in zip(starting_intervals, ending_intervals)] # in seconds
            morphed_span = combine_spans(morphed_span, threshold=0.2)
            print("morphed_spans: ", morphed_span)
            mask_interval = [[round(span[0]*codec_sr), round(span[1]*codec_sr)] for span in morphed_span]
            mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now

            gen_audio = inference_one_sample(
                ssrspeech_model["model"],
                ssrspeech_model["config"],
                ssrspeech_model["phn2num"],
                ssrspeech_model["text_tokenizer"], 
                ssrspeech_model["audio_tokenizer"],
                audio_path, original_transcript, sentence, mask_interval,
                cfg_coef, aug_text, False, True, False,
                device, decode_config
                )
        else:
            orig_spans = parse_tts_en(original_transcript, sentence) if ssrspeech_model_choice == 'English' else parse_tts_zh(original_transcript, sentence)
            print("orig_spans: ", orig_spans)
                
            starting_intervals = []
            ending_intervals = []
            for orig_span in orig_spans:
                start, end = get_mask_interval(transcribe_state, orig_span)
                starting_intervals.append(start)
                ending_intervals.append(end)
        
            print("intervals: ", starting_intervals, ending_intervals)
        
            info = torchaudio.info(audio_path)
            audio_dur = info.num_frames / info.sample_rate
            
            morphed_span = [(max(start, 1/codec_sr), min(end, audio_dur))
                            for start, end in zip(starting_intervals, ending_intervals)] # in seconds
            mask_interval = [[round(span[0]*codec_sr), round(span[1]*codec_sr)] for span in morphed_span]
            mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now
            print("mask_interval: ", mask_interval)
            gen_audio = inference_one_sample(
                ssrspeech_model["model"],
                ssrspeech_model["config"],
                ssrspeech_model["phn2num"],
                ssrspeech_model["text_tokenizer"], 
                ssrspeech_model["audio_tokenizer"],
                audio_path, original_transcript, sentence, mask_interval,
                cfg_coef, aug_text, False, True, True,
                device, decode_config
                )

        gen_audio = gen_audio[0].cpu()
        audio_tensors.append(gen_audio)

    if mode != "Rerun":
        output_audio = get_output_audio(audio_tensors, codec_audio_sr)
        sentences = [f"{idx}: {text}" for idx, text in enumerate(sentences)]
        component = gr.Dropdown(choices=sentences, value=sentences[0])
        return output_audio, inference_transcript, component, audio_tensors
    else:
        previous_audio_tensors[selected_sentence_idx] = audio_tensors[0]
        output_audio = get_output_audio(previous_audio_tensors, codec_audio_sr)
        sentence_audio = get_output_audio(audio_tensors, codec_audio_sr)
        return output_audio, inference_transcript, sentence_audio, previous_audio_tensors


def load_sentence(selected_sentence, codec_audio_sr, audio_tensors):
    if selected_sentence is None:
        return None
    colon_position = selected_sentence.find(':')
    selected_sentence_idx = int(selected_sentence[:colon_position])
    return get_output_audio([audio_tensors[selected_sentence_idx]], codec_audio_sr)

smart_transcript_info = """
If enabled, the target transcript will be constructed for you:</br>
 - In TTS and Long TTS mode just write the text you want to synthesize.</br>
 - In Edit mode just write the text to replace selected editing segment.</br>
If disabled, you should write the target transcript yourself:</br>
 - In TTS mode write prompt transcript followed by generation transcript.</br>
 - In Long TTS select split by newline (<b>SENTENCE SPLIT WON'T WORK</b>) and start each line with a prompt transcript.</br>
 - In Edit mode write full prompt</br>
"""

demo_original_transcript = "Gwynplaine had, besides, for his work and for his feats of strength, round his neck and over his shoulders, an esclavine of leather."

demo_text = {
    "TTS": {
        "smart": "I cannot believe that the same model can also do text to speech synthesis too!",
        "regular": "Gwynplaine had, besides, for his work and for his feats of strength, I cannot believe that the same model can also do text to speech synthesis too!"
    },
    "Edit": {
        "smart": "take over the stage for half an hour,",
        "regular": "Gwynplaine had, besides, for his work and for his feats of strength, take over the stage for half an hour, an esclavine of leather."
    },
    "Long TTS": {
        "smart": "You can run the model on a big text!\n"
                 "Just write it line-by-line. Or sentence-by-sentence.\n"
                 "If some sentences sound odd, just rerun the model on them, no need to generate the whole text again!",
        "regular": "Gwynplaine had, besides, for his work and for his feats of strength, You can run the model on a big text!\n"
                   "Gwynplaine had, besides, for his work and for his feats of strength, Just write it line-by-line. Or sentence-by-sentence.\n"
                   "Gwynplaine had, besides, for his work and for his feats of strength, If some sentences sound odd, just rerun the model on them, no need to generate the whole text again!"
    }
}

all_demo_texts = {vv for k, v in demo_text.items() for kk, vv in v.items()}

demo_words = ['0.069 Gwynplain 0.611', '0.671 had, 0.912', '0.952 besides, 1.414', '1.494 for 1.634', '1.695 his 1.835', '1.915 work 2.136', '2.196 and 2.297', '2.337 for 2.517', '2.557 his 2.678', '2.758 feats 3.019', '3.079 of 3.139', '3.2 strength, 3.561', '4.022 round 4.263', '4.303 his 4.444', '4.524 neck 4.705', '4.745 and 4.825', '4.905 over 5.086', '5.146 his 5.266', '5.307 shoulders, 5.768', '6.23 an 6.33', '6.531 esclavine 7.133', '7.213 of 7.293', '7.353 leather. 7.614']

demo_words_info = [{'word': 'Gwynplain', 'start': 0.069, 'end': 0.611, 'score': 0.833}, {'word': 'had,', 'start': 0.671, 'end': 0.912, 'score': 0.879}, {'word': 'besides,', 'start': 0.952, 'end': 1.414, 'score': 0.863}, {'word': 'for', 'start': 1.494, 'end': 1.634, 'score': 0.89}, {'word': 'his', 'start': 1.695, 'end': 1.835, 'score': 0.669}, {'word': 'work', 'start': 1.915, 'end': 2.136, 'score': 0.916}, {'word': 'and', 'start': 2.196, 'end': 2.297, 'score': 0.766}, {'word': 'for', 'start': 2.337, 'end': 2.517, 'score': 0.808}, {'word': 'his', 'start': 2.557, 'end': 2.678, 'score': 0.786}, {'word': 'feats', 'start': 2.758, 'end': 3.019, 'score': 0.97}, {'word': 'of', 'start': 3.079, 'end': 3.139, 'score': 0.752}, {'word': 'strength,', 'start': 3.2, 'end': 3.561, 'score': 0.742}, {'word': 'round', 'start': 4.022, 'end': 4.263, 'score': 0.916}, {'word': 'his', 'start': 4.303, 'end': 4.444, 'score': 0.666}, {'word': 'neck', 'start': 4.524, 'end': 4.705, 'score': 0.908}, {'word': 'and', 'start': 4.745, 'end': 4.825, 'score': 0.882}, {'word': 'over', 'start': 4.905, 'end': 5.086, 'score': 0.847}, {'word': 'his', 'start': 5.146, 'end': 5.266, 'score': 0.791}, {'word': 'shoulders,', 'start': 5.307, 'end': 5.768, 'score': 0.729}, {'word': 'an', 'start': 6.23, 'end': 6.33, 'score': 0.854}, {'word': 'esclavine', 'start': 6.531, 'end': 7.133, 'score': 0.803}, {'word': 'of', 'start': 7.213, 'end': 7.293, 'score': 0.772}, {'word': 'leather.', 'start': 7.353, 'end': 7.614, 'score': 0.896}]


def update_demo(mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word):
    if transcript not in all_demo_texts:
        return transcript, edit_from_word, edit_to_word

    replace_half = edit_word_mode == "Replace half"
    change_edit_from_word = edit_from_word == demo_words[2] or edit_from_word == demo_words[3]
    change_edit_to_word = edit_to_word == demo_words[11] or edit_to_word == demo_words[12]
    demo_edit_from_word_value = demo_words[2] if replace_half else demo_words[3]
    demo_edit_to_word_value = demo_words[12] if replace_half else demo_words[11]
    return [
        demo_text[mode]["smart" if smart_transcript else "regular"],
        demo_edit_from_word_value if change_edit_from_word else edit_from_word,
        demo_edit_to_word_value if change_edit_to_word else edit_to_word,
    ]


def get_app():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=2):
                load_models_btn = gr.Button(value="Load models")
            with gr.Column(scale=5):
                with gr.Accordion("Select models", open=False) as models_selector:
                    with gr.Row():
                        ssrspeech_model_choice = gr.Radio(label="ssrspeech model", value="English",
                                                        choices=["English", "Mandarin"])
                        whisper_backend_choice = gr.Radio(label="Whisper backend", value="whisperX", choices=["whisperX", "whisper"])
                        whisper_model_choice = gr.Radio(label="Whisper model", value="base.en",
                                                        choices=[None, "base.en", "small.en", "medium.en", "large"])
                        align_model_choice = gr.Radio(label="Forced alignment model", value="whisperX", choices=["whisperX", None])

        with gr.Row():
            with gr.Column(scale=2):
                input_audio = gr.Audio(value=f"{DEMO_PATH}/5895_34622_000026_000002.wav", label="Input Audio", type="filepath", interactive=True)
                with gr.Group():
                    original_transcript = gr.Textbox(label="Original transcript", lines=5, value=demo_original_transcript,
                                                    info="Use whisperx model to get the transcript. Fix and align it if necessary.")
                    with gr.Accordion("Word start time", open=False):
                        transcript_with_start_time = gr.Textbox(label="Start time", lines=5, interactive=False, info="Start time before each word")
                    with gr.Accordion("Word end time", open=False):
                        transcript_with_end_time = gr.Textbox(label="End time", lines=5, interactive=False, info="End time after each word")

                    transcribe_btn = gr.Button(value="Transcribe")
                    align_btn = gr.Button(value="Align")

            with gr.Column(scale=3):
                with gr.Group():
                    transcript = gr.Textbox(label="Text", lines=7, value=demo_text["TTS"]["smart"])

                    with gr.Row():
                        mode = gr.Radio(label="Mode", choices=["Edit", "TTS"], value="Edit")

                    run_btn = gr.Button(value="Run")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio")
                with gr.Accordion("Inference transcript", open=False):
                    inference_transcript = gr.Textbox(label="Inference transcript", lines=5, interactive=False,
                                                    info="Inference was performed on this transcript.")
                with gr.Group(visible=False) as long_tts_sentence_editor:
                    sentence_selector = gr.Dropdown(label="Sentence", value=None,
                                                    info="Select sentence you want to regenerate")
                    sentence_audio = gr.Audio(label="Sentence Audio", scale=2)
                    rerun_btn = gr.Button(value="Rerun")

        with gr.Row():
            with gr.Accordion("Generation Parameters - change these if you are unhappy with the generation", open=False):
                stop_repetition = gr.Radio(label="stop_repetition", choices=[-1, 1, 2, 3, 4], value=2,
                                        info="if there are long silence in the generated audio, reduce the stop_repetition to 2 or 1. -1 = disabled")
                seed = gr.Number(label="seed", value=-1, precision=0, info="random seeds always works :)")
                kvcache = gr.Radio(label="kvcache", choices=[0, 1], value=1,
                                    info="set to 0 to use less VRAM, but with slower inference")
                aug_text = gr.Radio(label="aug_text", choices=[0, 1], value=1,
                                    info="set to 1 to use cfg")
                cfg_coef = gr.Number(label="cfg_coef", value=1.5,
                                    info="cfg guidance scale, 1.5 is a good value")
                sub_amount = gr.Number(label="sub_amount", value=0.12, info="margin to the left and right of the editing segment")
                top_p = gr.Number(label="top_p", value=0.8, info="0.9 is a good value, 0.8 is also good")
                temperature = gr.Number(label="temperature", value=1, info="haven't try other values, do not recommend to change")
                top_k = gr.Number(label="top_k", value=0, info="0 means we don't use topk sampling, because we use topp sampling")
                codec_audio_sr = gr.Number(label="codec_audio_sr", value=16000, info='encodec specific, Do not change')
                codec_sr = gr.Number(label="codec_sr", value=50, info='encodec specific, Do not change')
                silence_tokens = gr.Textbox(label="silence tokens", value="[1388,1898,131]", info="encodec specific, do not change")

        success_output = gr.HTML()
        audio_tensors = gr.State()
        transcribe_state = gr.State(value={"words_info": demo_words_info})

        load_models_btn.click(fn=load_models,
                            inputs=[whisper_backend_choice, whisper_model_choice, align_model_choice, ssrspeech_model_choice],
                            outputs=[models_selector])


        transcribe_btn.click(fn=transcribe,
                            inputs=[seed, input_audio],
                            outputs=[original_transcript, transcript_with_start_time, transcript_with_end_time, transcribe_state, success_output])
        align_btn.click(fn=align,
                        inputs=[seed, original_transcript, input_audio],
                        outputs=[transcript_with_start_time, transcript_with_end_time, transcribe_state, success_output])

        run_btn.click(fn=run,
                    inputs=[
                        seed, sub_amount, ssrspeech_model_choice,
                        codec_audio_sr, codec_sr,
                        top_k, top_p, temperature,
                        stop_repetition,
                        kvcache, silence_tokens, aug_text, cfg_coef,
                        input_audio, transcribe_state, original_transcript, transcript,
                        mode, sentence_selector, audio_tensors
                    ],
                    outputs=[output_audio, inference_transcript, sentence_selector, audio_tensors])

        sentence_selector.change(fn=load_sentence,
                                inputs=[sentence_selector, codec_audio_sr, audio_tensors],
                                outputs=[sentence_audio])
        rerun_btn.click(fn=run,
                        inputs=[
                            seed, sub_amount, ssrspeech_model_choice,
                            codec_audio_sr, codec_sr,
                            top_k, top_p, temperature,
                            stop_repetition,
                            kvcache, silence_tokens, aug_text, cfg_coef, 
                            input_audio, transcribe_state, original_transcript, transcript,
                            gr.State(value="Rerun"), sentence_selector, audio_tensors
                        ],
                        outputs=[output_audio, inference_transcript, sentence_audio, audio_tensors])

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ssrspeech gradio app.")
    
    parser.add_argument("--demo-path", default="./demo", help="Path to demo directory")
    parser.add_argument("--tmp-path", default="./demo/temp", help="Path to tmp directory")
    parser.add_argument("--models-path", default="./pretrained_models", help="Path to ssrspeech models directory")
    parser.add_argument("--port", default=7860, type=int, help="App port")
    parser.add_argument("--share", action="store_true", help="Launch with public url")
    parser.add_argument("--server_name", default="127.0.0.1", type=str, help="Server name for launching the app. 127.0.0.1 for localhost; 0.0.0.0 to allow access from other machines in the local network. Might also give access to external users depends on the firewall settings.")

    os.environ["USER"] = os.getenv("USER", "user")
    args = parser.parse_args()
    DEMO_PATH = args.demo_path
    TMP_PATH = args.tmp_path
    MODELS_PATH = args.models_path

    app = get_app()
    app.queue().launch(share=args.share, server_name=args.server_name, server_port=args.port)
