# python inference.py  \
#     --use_downloaded_mfa \
#     --mfa_dict_path "./pretrained_models/english_us_arpa.dict" \
#     --mfa_path "./pretrained_models/english_us_arpa.zip" \
#     --top_p 0.8 \
#     --stop_repetition 2 \
#     --sample_batch_size 1 \
#     --cfg_coef 1.5 \
#     --aug_text \
#     --use_watermark \
#     --language 'en' \
#     --model_path "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/English_10k/e830M/best_bundle.pth" \
#     --codec_path "/apdcephfs_cq10/share_1603164/user/helinhwang/audiocraft/tmp/audiocraft_root/xps/4d60535d/checkpoint_3.th" \
#     --orig_audio "./demo/5895_34622_000026_000002.wav" \
#     --orig_transcript "Gwynplaine had, besides, for his work and for his feats of strength, round his neck and over his shoulders, an esclavine of leather." \
#     --target_transcript "Gwynplaine had, besides, for his work and for his feats of strength, I cannot believe that the same model can also do text to speech synthesis too!" \
#     --temp_folder "./demo/temp"\
#     --output_dir "./demo/generated_se"\
#     --savename "5895_34622_000026_000002"


# python inference.py  \
#     --use_downloaded_mfa \
#     --mfa_dict_path "./pretrained_models/mandarin_china_mfa.dict" \
#     --mfa_path "./pretrained_models/mandarin_mfa.zip" \
#     --top_p 0.8 \
#     --stop_repetition 2 \
#     --sample_batch_size 1 \
#     --cfg_coef 1.5 \
#     --aug_text \
#     --use_watermark \
#     --language 'zh' \
#     --model_path "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/Chinese_25k/e830M/best_bundle.pth" \
#     --codec_path "/apdcephfs_cq10/share_1603164/user/helinhwang/audiocraft/tmp/audiocraft_root/xps/4d60535d/checkpoint_3.th" \
#     --orig_audio "./demo/pony3.wav" \
#     --orig_transcript "能够更有效率地结合给用户提升更多的这种体验也包括他的这个他的后台的效率提升等等我相信这些额额业界的解决方案应该说是" \
#     --target_transcript "能够更有效率地结合给用户提升更多的体验也包括他的这个他的后台的效率提升等等在这个基础上我相信这些额额业界的解决方案应该说是" \
#     --temp_folder "./demo/temp"\
#     --output_dir "./demo/generated_se"\
#     --savename "pony3"


python inference.py  \
    --use_downloaded_mfa \
    --mfa_dict_path "./pretrained_models/mandarin_china_mfa.dict" \
    --mfa_path "./pretrained_models/mandarin_mfa.zip" \
    --top_p 0.8 \
    --stop_repetition 2 \
    --sample_batch_size 1 \
    --cfg_coef 1.5 \
    --aug_text \
    --use_watermark \
    --tts \
    --language 'zh' \
    --model_path "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/Chinese_25k/e830M/best_bundle.pth" \
    --codec_path "/apdcephfs_cq10/share_1603164/user/helinhwang/audiocraft/tmp/audiocraft_root/xps/4d60535d/checkpoint_3.th" \
    --orig_audio "./demo/pony3.wav" \
    --orig_transcript "能够更有效率地结合给用户提升更多的这种体验也包括他的这个他的后台的效率提升等等我相信这些额额业界的解决方案应该说是" \
    --target_transcript "能够更有效率地结合给用户提升更多的这种体验在游戏业务的强势反弹下，腾讯在二季度拿出了一份十分亮眼的成绩单" \
    --temp_folder "./demo/temp"\
    --output_dir "./demo/generated_tts"\
    --savename "pony3"
