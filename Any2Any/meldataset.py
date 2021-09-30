import torch.utils.data
import librosa
from librosa.filters import mel as librosa_mel_fn
import torchaudio
from torch.utils.data import Dataset, SequentialSampler
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import datetime
from librosa.util import normalize

def load_wav(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(str(full_path))
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
    # return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output



def mel_normalize(S,clip_val=1e-5):
    S = (S - torch.log(torch.Tensor([clip_val])))*1.0/(0-torch.log(torch.Tensor([clip_val])))
    return S
#
#
def mel_denormalize(S,clip_val=1e-5):
    S = S*(0-torch.log(torch.Tensor([clip_val])).cuda()) + torch.log(torch.Tensor([clip_val])).cuda()
    return S



mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # print('min value is ', torch.min(y))
    # print('max value is ', torch.max(y))
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


def get_spk_encoder(wav2mel_path, dvector_path):
    wav2mel_model = torch.jit.load(wav2mel_path)
    dvector_model = torch.jit.load(dvector_path).eval()
    return wav2mel_model,dvector_model

def get_test_dataset_filelist(wavs_dir):
    file_list = [os.path.join(wavs_dir, file_name) for file_name in os.listdir(wavs_dir)]
    choice_list = []
    for i in range(10):
        source_file = random.choice(file_list)
        target_file = random.choice(file_list)
        choice_list.append([source_file,target_file])
    return choice_list

def get_infer_dataset_filelist(wavs_dir):
    figure_list = os.listdir(wavs_dir)
    choice_list = []
    for i in range(100):
        source_figure = random.choice(figure_list)
        source_figure_dir = os.path.join(wavs_dir, source_figure)
        source_file_list = [os.path.join(source_figure_dir, file_name) for file_name in os.listdir(source_figure_dir)]
        source_file = random.choice(source_file_list)
        target_figure = random.choice(figure_list)
        while target_figure==source_figure:
            target_figure = random.choice(figure_list)
        target_figure_dir = os.path.join(wavs_dir, target_figure)
        target_file_list = [os.path.join(target_figure_dir, file_name) for file_name in os.listdir(target_figure_dir)]
        target_file = random.choice(target_file_list)
        choice_list.append([source_file, target_file])
    return choice_list



class Test_MelDataset(torch.utils.data.Dataset):
    def __init__(self, test_files, wav2mel_path,dvector_path,n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax,device=None):
        self.audio_files = test_files
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.device = device
        self.wav2mel, self.dvector = get_spk_encoder(wav2mel_path, dvector_path)

    def __getitem__(self, index):
        src_filename, tar_filename = self.audio_files[index]
        src_file_label = os.path.basename(src_filename).split(".")[0]
        tar_file_label = os.path.basename(tar_filename).split(".")[0]
        convert_label = src_file_label+'TO'+tar_file_label
        wav_tensor, sample_rate = torchaudio.load(tar_filename)
        mel_tensor = self.wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
        spk_emb = self.dvector.embed_utterance(mel_tensor).detach()  # shape: (emb_dim)
        spk_emb = normalize(spk_emb.squeeze().cpu().numpy())
        audio,sampling_rate = load_wav(src_filename)
        ### split
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                              self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                              center=False)
        mel = mel.squeeze(0).transpose(0,1)
        mel = mel_normalize(mel)
        return (spk_emb,mel,convert_label)

    def __len__(self):
        return len(self.audio_files)


class PickleDataset(Dataset):
    def __init__(self, figure_mel_data_dir, sample_num):
        with open(figure_mel_data_dir, 'rb') as f:
            self.figure_mel_label_map = pickle.load(f)

        self.sample_num = sample_num
        self.figure_list = list(self.figure_mel_label_map.keys())
        self.count = 0

    def __getitem__(self, index):
        if self.count == 0:
            random.seed(datetime.datetime.now())
            self.init_spk = random.choice(self.figure_list)
            self.speech_data = self.figure_mel_label_map[self.init_spk]

        self.count = (self.count + 1) % self.sample_num
        index = random.randint(0, len(self.speech_data) - 1)

        ori_mel = self.speech_data[index][1]
        ori_mel_tensor = torch.from_numpy(np.array(ori_mel))

        word = self.speech_data[index][2]

        spk_mel = self.speech_data[index][0]
        spk_mel = normalize(spk_mel)
        spk_input_mel_tensor = torch.from_numpy(np.array(spk_mel))

        return spk_input_mel_tensor, ori_mel_tensor, word

    def __len__(self):
        return len(self.figure_list) * 100


def collate_batch(batch):
    """Collate a batch of data."""
    # batch = batch[0]
    spk_input_mels, ori_mels, word = zip(*batch)
    spk_input_mels = torch.stack(spk_input_mels)
    ori_lens = [len(ori_mel) for ori_mel in ori_mels]

    overlap_lens = ori_lens
    ori_mels = pad_sequence(ori_mels, batch_first=True)
    mel_masks = [torch.arange(ori_mels.size(1)) >= mel_len for mel_len in ori_lens]
    mel_masks = torch.stack(mel_masks)  #

    return spk_input_mels, ori_mels, mel_masks, word, overlap_lens


def get_data_loader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=True):
    dataloader = DataLoader(dataset, collate_fn=collate_batch, batch_size=batch_size, num_workers=0, shuffle=True,
                            pin_memory=True, drop_last=drop_last)
    return dataloader