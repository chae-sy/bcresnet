# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random
from glob import glob
import shutil
import requests
import tarfile


import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from collections import Counter
from tqdm import tqdm

### GSC
label_dict = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
}
print("labels:\t", label_dict)
sample_per_cls_v1 = [1854, 258, 257]
sample_per_cls_v2 = [3077, 371, 408]
SR = 16000


def ScanAudioFiles(root_dir, ver):
    sample_per_cls = sample_per_cls_v1 if ver == 1 else sample_per_cls_v2
    audio_paths, labels = [], []
    for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
        random.shuffle(files)
        for idx, filename in enumerate(files):
            if not filename.endswith(".wav"):
                continue
            dataset, class_name = path.split("/")[-2:] #data/speech_commands_v0.02/yes -> get the dataset name, class_name
            if class_name in ("_unknown_", "_silence_"):  # balancing
                if "train" in dataset and idx == sample_per_cls[0]: # only get 3077 (=sample_per_cls_v2[0]) samples 
                    break
                if "valid" in dataset and idx == sample_per_cls[1]:  # only get 371 (=sample_per_cls_v2[1]) samples 
                    break
                if "test" in dataset and idx == sample_per_cls[2]: # only get 408 (=sample_per_cls_v2[2]) samples
                    break
            audio_paths.append(os.path.join(path, filename))
            labels.append(label_dict[class_name])
    return audio_paths, labels 


class SpeechCommandWithSpeaker(Dataset):
    def __init__(self, root_dir, ver, transform=None):
        self.transform = transform
        self.data = []  # [(file_path, label, speaker_id)]

        for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
            for file in files:
                if not file.endswith(".wav"): # skip the License, README.md, etc
                    continue
                class_name = path.split("/")[-1]
                if class_name not in label_dict: #if the class name is not in label_dict, skip
                    continue
                speaker_id = file.split("_")[0]
                file_path = os.path.join(path, file)
                label = label_dict[class_name]
                self.data.append((file_path, label, speaker_id)) # Add a tuple containing (file_path, label, speaker_id) for each audio file to the data

        self.speaker2idx = {spk: i for i, spk in enumerate(sorted(set(d[2] for d in self.data)))} #get the speaker index {spk : 0}, {spk : 1} ...
        print("Loaded samples:", len(self.data)) # total number of .wav files
        print("Unique speakers:", len(self.speaker2idx)) # total number of speakers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ get the waveform, label, speaker_id, audio_path """
        path, label, speaker = self.data[idx]
        waveform, _ = torchaudio.load(path)
        speaker_id = self.speaker2idx[speaker]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label, speaker_id, path


def spec_augment(
    x, frequency_masking_para=20, time_masking_para=20, frequency_mask_num=2, time_mask_num=2
):
    """
    refer to https://arxiv.org/abs/1904.08779
    masking blocks of frequency channels, and masking blocks of time steps for data augmentation
    """
    lenF, lenT = x.shape[1:3]
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, lenF - f)
        x[:, f0 : f0 + f, :] = 0
    # Time masking
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, lenT - t)
        x[:, :, t0 : t0 + t] = 0
    return x


class Preprocess:
    def __init__(
        self,
        noise_loc,
        device,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
        specaug=False,
        sample_rate=SR,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
    ):
        if noise_loc is None:
            self.background_noise = [] # if there is no background noise folder in the data directory, return empty list
        else:
            self.background_noise = [
                torchaudio.load(file_name)[0] for file_name in glob(noise_loc + "/*.wav")
            ]
            assert len(self.background_noise) != 0
        self.feature = LogMel(
            device,
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.sample_len = sample_rate
        self.specaug = specaug
        self.device = device
        if self.specaug:
            self.frequency_masking_para = frequency_masking_para
            self.time_masking_para = time_masking_para
            self.frequency_mask_num = frequency_mask_num
            self.time_mask_num = time_mask_num
            print(
                "frequency specaug %d %d" % (self.frequency_mask_num, self.frequency_masking_para)
            )
            print("time specaug %d %d" % (self.time_mask_num, self.time_masking_para))

    def __call__(self, x, labels, augment=True, noise_prob=0.8, is_train=True):
        assert len(x.shape) == 3
        if torch.is_tensor(labels):
            if labels.dim() == 0:
                labels=[labels.item()]
            elif labels.dim() == 1:
                labels=labels.tolist()
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
        if augment:
            for idx in range(x.shape[0]):
                if labels[idx].item() != 0 and (not is_train or random.random() > noise_prob):
                    # if test data or random number > 0.8 (probability of 20%), skip adding noise to the input audio
                    continue
                    # if (train data) and randum number < 0.8 (probability of 80%), add noise to the input audio
                noise_amp = (
                    np.random.uniform(0, 0.1) if labels[idx] != 0 else np.random.uniform(0, 1)
                )
                noise = random.choice(self.background_noise).to(self.device) #randomly choose the noise audio file among doing_the_dishes.wav  dude_miaowing.wav  exercise_bike.wav  pink_noise.wav  running_tap.wav  white_noise.wav
                sample_loc = random.randint(0, noise.shape[-1] - self.sample_len) # get the random 1 second piece from the noise.wav
                noise = noise_amp * noise[:, sample_loc : sample_loc + SR] 

                if is_train:
                    x_shift = int(np.random.uniform(-0.1, 0.1) * SR) #randomly shift the audio waveform in x direction
                    zero_padding = torch.zeros(1, np.abs(x_shift)).to(self.device) # pad the blanks which were created by x_shift
                    if x_shift < 0:
                        temp_x = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        temp_x = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
                    x[idx] = temp_x + noise # add the noise to the audio file
                else:  # valid
                    x[idx] = x[idx] + noise
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        x = self.feature(x) #apply log+mel to the input waveform
        if self.specaug: #if spec_aug, apply spec_aug to the input waveform
            for i in range(x.shape[0]):
                x[i] = spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                )
        return x


class LogMel: 
    """ get the Melspectrogram + log """
    def __init__(
        self, device, sample_rate=SR, hop_length=160, win_length=480, n_fft=512, n_mels=40
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.device = device

    def __call__(self, x):
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output


class Padding:
    """zero pad to have 1 sec len"""

    def __init__(self):
        self.output_len = SR

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x

def DownloadDataset(loc, url):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576
    with open(os.path.join(loc, filename), "wb") as f:
        for data in response.iter_content(block_size):
            f.write(data)
            read_so_far = f.tell()
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                print(f"Downloaded {read_so_far} of {total_size} bytes ({percent:.2f}%)")
    with tarfile.open(os.path.join(loc, filename), "r:gz") as tar:
        tar.extractall(loc)

def make_empty_audio(loc, num):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    for i in range(num):
        path = os.path.join(loc, "%s.wav" % str(i))
        zeros = torch.zeros([1, SR])  # 1 sec long.
        torchaudio.save(path, zeros, SR)


def make_12class_dataset(base, target):#base = speech_command_v0.02_split/train, target = speech_command_v0.02/train_12class
    os.mkdir(target)
    os.mkdir(target + "/_unknown_")
    class10 = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    for clsdir in glob(os.path.join(base, "*")): # for every dir in speech_command_v0.02_split/train
        class_name = os.path.basename(clsdir)
        if class_name in class10:
            target_dir = os.path.join(target, class_name)
            shutil.copytree(clsdir, target_dir) 
            # copy the every dir in speech_command_v0.02_split/train to speech_command_v0.02/train_12class
            # (=make speech_command_v0.02/train_12class/{10 keyword name})
            print(f"Copied {clsdir} to {target_dir}")
        else: # if the keyword is not included in class10, append it to speech_command_v0.02/train_12class/_unknown_
            for file_path in glob(os.path.join(clsdir, "*")):
                filename = os.path.basename(file_path)
                target_dir = os.path.join(target, "_unknown_")
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, class_name + "_" + filename)
                shutil.copy(file_path, target_file)
                print(f"Copied {file_path} to {target_file}")

def split_data(base, target, valid_list, test_list):
    #split the data into test, validation, train according to the validation_list.txt & testing_list.txt
    #copy and paste the files into each dir (test, val, train) from the speech_command_v0.01 & speech_command_v0.02 dir
    with open(valid_list, "r") as f:
        valid_names = [item.rstrip() for item in f.readlines()]
    with open(test_list, "r") as f:
        test_names = [item.rstrip() for item in f.readlines()]

    trg_base_dirs = [
        os.path.join(target, "train"),
        os.path.join(target, "valid"),
        os.path.join(target, "test"),
    ]
    for item in trg_base_dirs:
        if not os.path.isdir(item):
            os.mkdir(item)

    for root, _, files in os.walk(base):
        for file_name in files:
            if not file_name.endswith(".wav"):
                continue

            if "_background_noise_" in os.path.join(root, file_name):
                continue

            class_name = root.split("/")[-1]
            for item in trg_base_dirs:
                if not os.path.isdir(os.path.join(item, class_name)):
                    os.mkdir(os.path.join(item, class_name))
            org_file_name = os.path.join(root, file_name)
            trg_file_name = os.path.join(class_name, file_name)
            if trg_file_name in valid_names:
                target_dir = trg_base_dirs[1]
            elif trg_file_name in test_names:
                target_dir = trg_base_dirs[-1]
            else:
                target_dir = trg_base_dirs[0]
            target_path = os.path.join(target_dir, trg_file_name)
            shutil.copy(org_file_name, target_path)
            print(f"Copied {org_file_name} to {target_path}")


def SplitDataset(loc):
    target_loc = "%s_split" % loc 
    if not os.path.isdir(target_loc):
        os.mkdir(target_loc) #make "speech_commands_v0.01_split", "speech_commands_v0.02_split"
    split_data( 
        loc,
        target_loc,
        os.path.join(loc, "validation_list.txt"),
        os.path.join(loc, "testing_list.txt"),
    )

    sample_per_cls = sample_per_cls_v1 if "v0.01" in loc else sample_per_cls_v2
    for idx, split_name in enumerate(["train", "valid", "test"]):
        make_12class_dataset(
            # if the class is in 10 keywords, append it to corresponding dir. 
            # if not, append it to _unknown_ dir
            "%s/%s" % (target_loc, split_name), "%s/%s_12class" % (loc, split_name) #i.e., speech_command_v0.02_split/train, speech_command_v0.02/train_12class
        )
        make_empty_audio("%s/%s_12class/_silence_" % (loc, split_name), sample_per_cls[idx]) #make empty audio and append it to _silence_ dir.
        
        
import random
from torch.utils.data import Dataset

class PKMTLDataset(Dataset):
    """
    summary:
    
    Custom torch.utils.data.Dataset designed to simulate the PK-MTL training pipeline described in the paper (e.g., Fig. 2 and Sec 2.2),
    where each training sample requires a group of 5 audio clips with specific keyword/speaker relations.
    
    function: 
            __init__ 
            input [class] : base_dataset (e.g., SpeechCommandWithSpeaker, where each item is (waveform, label, speaker_id, path))
              1) indexes all samples by keyword and speaker
                    self.index_by_label: maps each label -> list of indices
                    self.index_by_speaker : maps each speaker -> list of indices
              2) precomputes valid_indices 
                    for each sample, checks if all 4 companion samples(ts-tk, ts-ntk, nts-tk, nts-ntk) exist.
                    stores only valid anchor indices

            _precompute_valid_indices
            ensures that for a given anchor, the dataset contains valid samples for all 4 pair types.
            output [list] : a list of indexes whose data is valid

            __getitem__
            input [int] : data index

            picks an index from the list of valid anchors
            uses the helper sample() to construct the 4 companions
            if any of the 4 are missing, it resamples a different anchor

            output [dict]: {
                    "anchor": Tensor(waveform),
                    "ts_tk": Tensor(waveform),
                    "ts_ntk": Tensor(waveform),
                    "nts_tk": Tensor(waveform),
                    "nts_ntk": Tensor(waveform),
                    "target_label": Tensor(keyword_id),
                    "target_speaker": Tensor(speaker_id)
                    }
    """
    def __init__(self, base_dataset): #base_dataset : SpeechCommandWithSpeaker
        self.base = base_dataset
        self.index_by_label = {}
        self.index_by_speaker = {}

        for i, (_, label, speaker_id, _) in enumerate(self.base):
            #ensures that for each label, there's a list in the dictionary, and then it appends index of the data to that list.
            self.index_by_label.setdefault(label, []).append(i)
            #ensures that for each speaker, there's a list in the dictionary, and then it appends index of the data to that list.
            self.index_by_speaker.setdefault(speaker_id, []).append(i)

        self.valid_indices = self._precompute_valid_indices() # list of valid idx
        print(f"✅ Valid anchor indices: {len(self.valid_indices)}")

    def _precompute_valid_indices(self):
        valid = []
        for idx in range(len(self.base)): # for every data in SpeechCommandWithSpeaker
            _, label, speaker, _ = self.base[idx] #get the label(=keyword) & speaker (=speaker_id)
            other_labels = list(set(self.index_by_label.keys()) - {label}) # list of keywords other than 'label'
            other_speakers = list(set(self.index_by_speaker.keys()) - {speaker}) # list of speakers other than 'speaker

            def has(label, speaker, exclude=None): 
                """ checks whether there is any other sample from this label+speaker pair"""
                candidates = set(self.index_by_label[label]) & set(self.index_by_speaker[speaker]) 
                # union of index_by_label of the given label & index_by_spaker of the given speaker
                if exclude is not None:
                    candidates.discard(exclude)
                return len(candidates) > 0 # if candidates > 0, it means there are multiple data that contain the same word spoken by the same speaker

            if not other_labels or not other_speakers: # if there is no data which is included in both other_labels and other_speakers (no ts-ntk, nts-tk, & nts-ntk)
                continue #skip the data

            if ( #if all 4 types of samples exist
                has(label, speaker, exclude=idx) and #ts-tk, "exclude" is for not picking the anchor itself
                any(has(lbl, speaker) for lbl in other_labels) and #ts-ntk
                any(has(label, spk) for spk in other_speakers) and #nts-tk
                any(has(lbl, spk) for lbl in other_labels for spk in other_speakers) #nts-ntk
            ):
                valid.append(idx) #append the index of data to valid
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, valid_idx):
        idx = self.valid_indices[valid_idx] # get the index of the valid data
        anchor_wave, anchor_label, anchor_spk, _ = self.base[idx]

        def sample(label=None, speaker=None, exclude=None):
            candidates = set(self.index_by_label[label]) & set(self.index_by_speaker[speaker]) # candidates = other data with same word & same speaker
            if exclude is not None: # to prevent picking an anchor itself again
                candidates.discard(exclude)
            if not candidates: # if there is no candidates
                return None
            wav, _, _, _ = self.base[random.choice(list(candidates))]
            return wav

        other_labels = list(set(self.index_by_label.keys()) - {anchor_label})
        other_speakers = list(set(self.index_by_speaker.keys()) - {anchor_spk})

        ts_tk = sample(anchor_label, anchor_spk, exclude=idx) #"exclude" is for not picking the anchor itself
        ts_ntk = sample(random.choice(other_labels), anchor_spk)
        nts_tk = sample(anchor_label, random.choice(other_speakers)) if other_speakers else None
        nts_ntk = sample(random.choice(other_labels), random.choice(other_speakers)) if other_speakers else None


        # if any of the 4 required samples are None, 
        # randomly picks a different index and tries again by calling __getitem__() recursively.
        if None in (ts_tk, ts_ntk, nts_tk, nts_ntk):
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "anchor": anchor_wave, #waveform of anchor
            "ts_tk": ts_tk, #waveform of ts_tk, same speaker & same keyword as anchor (but different waveform. this is for generalization
                            #i.e., “Can the model tell that this other utterance is from the same speaker, same keyword — even though it’s a different recording?”)
            "ts_ntk": ts_ntk, #waveform of ts_ntk, same speaker, different keyword
            "nts_tk": nts_tk, #waveform of nts_tk, different speaker, same keyword
            "nts_ntk": nts_ntk, #waveform of nts_ntk, different speaker, different keyword
            "target_label": torch.tensor(anchor_label),
            "target_speaker": torch.tensor(anchor_spk)
        }

def preprocess_and_save(dataset, preprocess_fn, device, save_dir, batch_size=256):
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_data, all_labels = [], []
    for batch in tqdm(loader, desc=f"Preprocessing -> {save_dir}"):
        x = batch['anchor'].to(device)
        labels = batch['target_label'].to(device)
        x = preprocess_fn(x, labels, augment=False, is_train=False)
        all_data.append(x.cpu())
        all_labels.append(labels.cpu())

    x_tensor = torch.cat(all_data)
    y_tensor = torch.cat(all_labels)

    torch.save(x_tensor, os.path.join(save_dir, "data.pt"))
    torch.save(y_tensor, os.path.join(save_dir, "labels.pt"))
    print(f"✅ Saved: {x_tensor.shape[0]} samples to {save_dir}")

    if hasattr(dataset.base, 'speaker2idx'):
        torch.save(dataset.base.speaker2idx, f"{save_dir}/speaker2idx.pt")
        print(f"✅ Saved: {save_dir}/speaker2idx.pt")

def show_label_distribution(label_path, label_name="Label"):
    labels = torch.load(label_path)
    labels = labels.tolist() if torch.is_tensor(labels) else labels

    counter = Counter(labels)
    print(f"📊 {label_name} distribution:")
    for label, count in sorted(counter.items()):
        print(f"  - Label {label}: {count} samples")
