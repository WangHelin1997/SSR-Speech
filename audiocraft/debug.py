from audiocraft.solvers import WMCompressionSolver
import torch
import torchaudio
# encodec_fn = "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/VoiceCraft/encodec_4cb2048_giga.th"
# model = WMCompressionSolver.model_from_encodec_checkpoint(encodec_fn)
# model_path = '/apdcephfs_cq10/share_1603164/user/helinhwang/audiocraft/tmp/audiocraft_root/xps/4d60535d/checkpoint.th'
# model = WMCompressionSolver.model_from_checkpoint(model_path)
# print(model)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

model_path = '/apdcephfs_cq10/share_1603164/user/helinhwang/audiocraft/tmp/audiocraft_root/xps/4d60535d/checkpoint.th'

class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device = None,
        signature = None
    ) -> None:
        model = WMCompressionSolver.model_from_checkpoint(signature)
        self.sample_rate = model.sample_rate
        self.channels = model.channels
        
        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self.codec.encode(wav.to(self.device))
        return [(codes[0], None)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames[0][0] # [1,4,T]
        return self.codec.decode(frames.to(self.device))

    def wmdecode(self, frames: torch.Tensor, marks: torch.Tensor, wav: torch.Tensor):
        frames = frames[0][0] # [1,4,T]
        out, _ = self.codec.wmdecode(frames.to(self.device), marks.to(self.device), wav.to(self.device))
        return out

    def detect_watermark(self, wav: torch.Tensor):
        marks = self.codec.detect_watermark(wav.to(self.device))
        return marks


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str, offset = -1, num_frames=-1):
    # Load and pre-process the audio waveform
    if offset != -1 and num_frames!=-1:
        wav, sr = torchaudio.load(audio_path, frame_offset=offset, num_frames=num_frames)
    else:
        wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)
    leng = (wav.shape[-1] // 320) * 320
    wav = wav[...,:leng]
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
        decoded_wav = tokenizer.decode(encoded_frames)
        q_res, mark, mark_label = tokenizer.codec.forward(wav.cuda())
    return wav, encoded_frames, decoded_wav, q_res.x, mark, mark_label
        
audiotokenizer =  AudioTokenizer('cuda', model_path)   
audiopath = '/apdcephfs_cq10/share_1603164/user/helinhwang/Speech-Editing/demo/temp_test/YOU1000000001_S0000037.wav'
wav, encoded_frames, decoded_wav, wmdecoded_wav, mark, mark_label = tokenize_audio(audiotokenizer, audiopath)
pred = torch.argmax(mark, dim=-1).squeeze()
print(mark.shape)
print(pred)
print(mark_label)

torchaudio.save('test_ori.wav', wav[0].cpu(), 16000)
torchaudio.save('test_encodec.wav', decoded_wav[0].cpu(), 16000)
torchaudio.save('test_wmencodec.wav', wmdecoded_wav[0].cpu(), 16000)

# torch.Size([8, 128, 50]) torch.Size([8, 16000]) torch.Size([8, 1, 16000])
# x, y, z = torch.randn(8, 128, 50), torch.zeros(8, 50, dtype=torch.long), torch.randn(8, 1, 16000)
# out, m = model.wmdecoder(x,y,z)
# print(out.shape,m.shape)

# for name, param in model.named_parameters():
#     print(f"Name: {name}")
    # print(f"Parameter: {param}")

# print(model.wmdecoder.model)
# Sequential(                                                                                                                      
#   (0): StreamableConv1d(                                                                                                         
#     (conv): NormConv1d(                                                                                                          
#       (conv): Conv1d(128, 1024, kernel_size=(7,), stride=(1,))                                                                   
#       (norm): Identity()                                                                                                         
#     )                                                                                                                            
#   )                                                                                                                              
#   (1): StreamableLSTM(                                                                                                           
#     (lstm): LSTM(1024, 1024, num_layers=2)                                                                                       
#   )                                                                                                                              
#   (2): ELU(alpha=1.0)                                                                                                            
#   (3): StreamableConvTranspose1d(                                                                                                
#     (convtr): NormConvTranspose1d(                                                                                               
#       (convtr): ConvTranspose1d(1024, 512, kernel_size=(16,), stride=(8,))                                                       
#       (norm): Identity()                                                                                                         
#     )                                                                                                                            
#   )
# (4): SEANetResnetBlock(
#     (block): Sequential(
#       (0): ELU(alpha=1.0)
#       (1): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(512, 256, kernel_size=(3,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#       (2): ELU(alpha=1.0)
#       (3): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(256, 512, kernel_size=(1,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#     )
#     (shortcut): Identity()
#   )
#   (5): ELU(alpha=1.0)
#   (6): StreamableConvTranspose1d(
#     (convtr): NormConvTranspose1d(
#       (convtr): ConvTranspose1d(512, 256, kernel_size=(10,), stride=(5,))
#       (norm): Identity()
#     )
#   )
# (7): SEANetResnetBlock(
#     (block): Sequential(
#       (0): ELU(alpha=1.0)
#       (1): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(256, 128, kernel_size=(3,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#       (2): ELU(alpha=1.0)
#       (3): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(128, 256, kernel_size=(1,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#     )
#     (shortcut): Identity()
#   )
#   (8): ELU(alpha=1.0)
#   (9): StreamableConvTranspose1d(
#     (convtr): NormConvTranspose1d(
#       (convtr): ConvTranspose1d(256, 128, kernel_size=(8,), stride=(4,))
#       (norm): Identity()
#     )
#   )
# (10): SEANetResnetBlock(                                                                                                       
#     (block): Sequential(                                                                                                         
#       (0): ELU(alpha=1.0)                                                                                                        
#       (1): StreamableConv1d(                                                                                                     
#         (conv): NormConv1d(                                                                                                      
#           (conv): Conv1d(128, 64, kernel_size=(3,), stride=(1,))                                                                 
#           (norm): Identity()                                                                                                     
#         )                                                                                                                        
#       )                                                                                                                          
#       (2): ELU(alpha=1.0)                                                                                                        
#       (3): StreamableConv1d(                                                                                                     
#         (conv): NormConv1d(                                                                                                      
#           (conv): Conv1d(64, 128, kernel_size=(1,), stride=(1,))                                                                 
#           (norm): Identity()                                                                                                     
#         )                                                                                                                        
#       )                                                                                                                          
#     )
#     (shortcut): Identity()
#   )
#   (11): ELU(alpha=1.0)
#   (12): StreamableConvTranspose1d(
#     (convtr): NormConvTranspose1d(
#       (convtr): ConvTranspose1d(128, 64, kernel_size=(4,), stride=(2,))
#       (norm): Identity()
#     )
#   )
#   (13): SEANetResnetBlock(
#     (block): Sequential(
#       (0): ELU(alpha=1.0)
#       (1): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(64, 32, kernel_size=(3,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#       (2): ELU(alpha=1.0)
#       (3): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#     )
#     (shortcut): Identity()
#   )
#   (14): ELU(alpha=1.0)
#   (15): StreamableConv1d(
#     (conv): NormConv1d(
#       (conv): Conv1d(64, 1, kernel_size=(7,), stride=(1,))
#       (norm): Identity()
#     )
#   )
# )
# print(model.encoder.model)
# Sequential(                                                                                                                      
#   (0): StreamableConv1d(                                                                                                         
#     (conv): NormConv1d(                                                                                                          
#       (conv): Conv1d(1, 64, kernel_size=(7,), stride=(1,))                                                                       
#       (norm): Identity()                                                                                                         
#     )                                                                                                                            
#   )                                                                                                                              
#   (1): SEANetResnetBlock(                                                                                                        
#     (block): Sequential(                                                                                                         
#       (0): ELU(alpha=1.0)                                                                                                        
#       (1): StreamableConv1d(                                                                                                     
#         (conv): NormConv1d(                                                                                                      
#           (conv): Conv1d(64, 32, kernel_size=(3,), stride=(1,))                                                                  
#           (norm): Identity()                                                                                                     
#         )                                                                                                                        
#       )                                                                                                                          
#       (2): ELU(alpha=1.0)                                                                                                        
#       (3): StreamableConv1d(                                                                                                     
#         (conv): NormConv1d(                                                                                                      
#           (conv): Conv1d(32, 64, kernel_size=(1,), stride=(1,))                                                                  
#           (norm): Identity()                                                                                                     
#         )                                                                                                                        
#       )                                                                                                                          
#     )                                                                                                                            
#     (shortcut): Identity()                                                                                                       
#   )                                                                                                                              
#   (2): ELU(alpha=1.0)                                                                                                            
#   (3): StreamableConv1d(                                                                                                         
#     (conv): NormConv1d(                                                                                                          
#       (conv): Conv1d(64, 128, kernel_size=(4,), stride=(2,))                                                                     
#       (norm): Identity()                                                                                                         
#     )                                                                                                                            
#   )                                                                                                                              
#   (4): SEANetResnetBlock(                                                                                                        
#     (block): Sequential(                                                                                                         
#       (0): ELU(alpha=1.0)                                                                                                        
#       (1): StreamableConv1d(                                                                                                     
#         (conv): NormConv1d(                                                                                                      
#           (conv): Conv1d(128, 64, kernel_size=(3,), stride=(1,))                                                                 
#           (norm): Identity()                                                                                                     
#         )
#       )
#       (2): ELU(alpha=1.0)
#       (3): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(64, 128, kernel_size=(1,), stride=(1,)) 
#           (norm): Identity()
#         )
#       )
#     )
#     (shortcut): Identity()
#   )
#   (5): ELU(alpha=1.0)
#   (6): StreamableConv1d(
#     (conv): NormConv1d(
#       (conv): Conv1d(128, 256, kernel_size=(8,), stride=(4,))
#       (norm): Identity()
#     )
#   )
#   (7): SEANetResnetBlock(
#     (block): Sequential(
#       (0): ELU(alpha=1.0)
#       (1): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(256, 128, kernel_size=(3,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#       (2): ELU(alpha=1.0)
#       (3): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(128, 256, kernel_size=(1,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#     )
#     (shortcut): Identity()
#   )
#   (8): ELU(alpha=1.0)
#   (9): StreamableConv1d(
#     (conv): NormConv1d(
#       (conv): Conv1d(256, 512, kernel_size=(10,), stride=(5,))
#       (norm): Identity()
#     )
#   )
#   (10): SEANetResnetBlock(
#     (block): Sequential(
#       (0): ELU(alpha=1.0)
#       (1): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(512, 256, kernel_size=(3,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#       (2): ELU(alpha=1.0)
#       (3): StreamableConv1d(
#         (conv): NormConv1d(
#           (conv): Conv1d(256, 512, kernel_size=(1,), stride=(1,))
#           (norm): Identity()
#         )
#       )
#     )
#     (shortcut): Identity()
#   )
#   (11): ELU(alpha=1.0)
#   (12): StreamableConv1d(
#     (conv): NormConv1d(
#       (conv): Conv1d(512, 1024, kernel_size=(16,), stride=(8,))
#       (norm): Identity()
#     )
#   )
#   (13): StreamableLSTM(
#     (lstm): LSTM(1024, 1024, num_layers=2)
#   )
#   (14): ELU(alpha=1.0)
#   (15): StreamableConv1d(
#     (conv): NormConv1d(
#       (conv): Conv1d(1024, 128, kernel_size=(7,), stride=(1,))
#       (norm): Identity()
#     )
#   )
# )