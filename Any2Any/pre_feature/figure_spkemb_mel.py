import os
import random
import librosa
import torchaudio
import pickle
import yaml
import numpy as np
import torch
import torch.utils.data
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn


def load_wav(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(str(full_path))
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_normalize(S,clip_val=1e-5):
    S = (S - torch.log(torch.Tensor([clip_val])))*1.0/(0-torch.log(torch.Tensor([clip_val])))
    return S

mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def get_dataset_filelist(wavs_dir):
    figure_wavs={}
    for figure in os.listdir(wavs_dir):
        figure_dir = os.path.join(wavs_dir,figure)
        figure_wavs[figure] = [os.path.join(figure_dir, file_name) for file_name in os.listdir(figure_dir)]
    return figure_wavs

def get_spk_encoder(wav2mel_path, dvector_path):
    wav2mel_model = torch.jit.load(wav2mel_path)
    dvector_model = torch.jit.load(dvector_path).eval()
    return wav2mel_model,dvector_model

def get_spkemb_mels(config):
        figure_wavs_map = get_dataset_filelist(config["wav_dir"])
        wav2mel,dvector = get_spk_encoder(config["wav2mel_model_path"], config["dvector_model_path"])
        spk_emb_mel_label_map = {}
        for figure, file_list in figure_wavs_map.items():
            mel_label_list = []
            print("figure:",figure)
            for filename in file_list:
                try:
                    wav_tensor, sample_rate = torchaudio.load(filename)
                    mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
                    spk_emb = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
                    spk_emb = spk_emb.detach().squeeze().cpu().numpy()
                    file_label = os.path.basename(filename).split(".")[0]
                    audio, sampling_rate = load_wav(filename)
                    audio = normalize(audio) * 0.95
                    audio = torch.FloatTensor(audio)
                    audio = audio.unsqueeze(0)
                    mel = mel_spectrogram(audio, config["n_fft"], config["num_mels"], config["sampling_rate"],
                                          config["hop_size"], config["win_size"], config["fmin"], config["fmax"],
                                          center=False)
                    mel = mel.squeeze(0).transpose(0, 1)
                    mel = mel_normalize(mel)
                    # spk_emb:56-dim  mel:[len,80]  file_label: p225_001
                    mel_label_list.append([spk_emb, mel, file_label])
                except:
                    print("filename:",filename)

            spk_emb_mel_label_map[figure] = mel_label_list
            break
        with open(os.path.join(config["out_dir"], 'spk_emb_mel_label.pkl'), 'wb') as handle:
            pickle.dump(spk_emb_mel_label_map, handle)

if __name__ == '__main__':
    config_path = r"Any2Any/pre_feature/preprocess_config.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    get_spkemb_mels(config)







