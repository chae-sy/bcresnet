import os
import time
import torch
import shutil
import numpy as np
from glob import glob
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

from pkmtl import PKMTLNet, train_epoch, evaluate, evaluate_with_far_frr
from model import BCResNets
from utils import show_label_distribution, DownloadDataset, Padding, Preprocess, SpeechCommandWithSpeaker, PKMTLDataset, SplitDataset, preprocess_and_save


class Trainer:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--ver", default=2, type=int, help="GSC version")
        parser.add_argument("--tau", default=3, type=float, choices=[1, 1.5, 2, 3, 6, 8])
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--download", action="store_true")
        parser.add_argument("--epoch", default=50, type=int)
        parser.add_argument("--batch_size", default=4096, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        args = parser.parse_args()
        self.__dict__.update(vars(args))

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        print(f'The code is on {self.device}')

        #load the data & show the label distribution
        for case in ['train', 'valid']:
            self._load_data_and_label(case)
        
        self._load_model()

    
    def pkmtl_collate(self, batch):
        # batch: List[Dict[str,Tensor]]
        return {
            k: torch.stack([sample[k] for sample in batch], dim=0)
            for k in batch[0].keys()
        }


    def _load_data_and_label(self, case):
        save_dir = f"cached/{case}"
        base_dir = "./data/speech_commands_v0.01"
        noise_dir = f"{base_dir}/_background_noise_"
        data_path = os.path.join(save_dir, "data.pt")
        label_path = os.path.join(save_dir, "labels.pt")
        specaugment = self.tau >= 1.5
        freq_masking = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        if os.path.exists(data_path) and os.path.exists(label_path) and  os.path.exists(os.path.join(save_dir, "speaker2idx.pt")):
            print(f"✅ Cache found in {save_dir}, skipping preprocessing.")
            x = torch.load(data_path)
            y = torch.load(label_path)
            shuffle = True if case == 'train' else False
            dataset = TensorDataset(x, y)
            if case=='train':
                self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.pkmtl_collate, num_workers=self.num_workers)
                self.speaker2idx = torch.load("cached/train/speaker2idx.pt")
                self.preprocess_train = Preprocess(noise_dir, self.device, specaug=specaugment, frequency_masking_para=freq_masking[self.tau])

            elif case == 'valid':
                self.preprocess_test = Preprocess(noise_dir, self.device)
                self.valid_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.pkmtl_collate, num_workers=self.num_workers)
            print(f"📦 Loaded {len(dataset)} samples from cache ({save_dir})")
        else:
            self._load_data(case)
        show_label_distribution(f"cached/{case}/labels.pt", label_name=f"{case} Set")

    def __call__(self):
        total_epoch =self.epoch
        learning_rate = 0.001
        embedding_dim = 128
        num_keywords = 12
        num_speakers = len(self.speaker2idx)
        batch_size = self.batch_size

        model = PKMTLNet(self.model, embedding_dim, num_keywords, num_speakers, alpha=0.5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        frr_list, far_list, acc_list = [], [], []
        alpha_grid = np.linspace(0.0, 1.0, 11)
        threshold_grid = np.linspace(-1.0, 1.0, 101)
        
        # keyword‐spotting loss
        kws_criterion = nn.CrossEntropyLoss()

        # speaker‐verification (classification) loss
        sv_criterion  = nn.CrossEntropyLoss()

        for split in range(10):
            print(f"\n🔁 Evaluating Split {split + 1}/10")
            for epoch in range(total_epoch):
                train_loss = train_epoch(model, self.train_loader, optimizer, self.device, kws_criterion, sv_criterion)

            best_alpha, best_thresh, best_frr = self.grid_search_threshold_alpha(model, self.valid_loader, alpha_grid, threshold_grid)
            print(f"Best Alpha: {best_alpha:.2f}, Best Threshold: {best_thresh:.2f}, FRR: {best_frr:.4f}")

            model.scm.alpha = best_alpha
            frr, far = evaluate_with_far_frr(model, self.valid_loader, self.device, threshold=best_thresh, task='scm', preprocess_fn=self.preprocess_test)
            acc_kws, _ = evaluate(model, self.valid_loader, self.device, self.preprocess_test)

            frr_list.append(frr)
            far_list.append(far)
            acc_list.append(acc_kws)

            print(f"Split {split+1} — FAR: {far:.4f}, FRR: {frr:.4f}, Top-1 Acc: {acc_kws:.4f}, ERR: {1 - acc_kws:.4f}")

        avg_frr = np.mean(frr_list)
        avg_far = np.mean(far_list)
        avg_acc = np.mean(acc_list)
        print(f"\n📊 Final Avg over 10 splits → FAR: {avg_far:.4f}, FRR: {avg_frr:.4f}, Top-1 Acc: {avg_acc:.4f}, ERR: {1 - avg_acc:.4f}")

        self.save_model(model)

    def grid_search_threshold_alpha(self, model, val_loader, alpha_grid, threshold_grid):
        best_alpha, best_thresh, best_frr = None, None, float('inf')
        for alpha in alpha_grid:
            model.scm.alpha = alpha
            for thresh in threshold_grid:
                frr, far = evaluate_with_far_frr(model, val_loader, self.device, threshold=thresh, task='scm', preprocess_fn=self.preprocess_test)
                if far <= 0.01 and frr < best_frr:
                    best_frr = frr
                    best_thresh = thresh
                    best_alpha = alpha
        return best_alpha, best_thresh, best_frr

    def save_model(self, model):
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][time.localtime().tm_mon - 1]
        date = time.localtime().tm_mday
        today = f'{month}{date}'
        now_time = f'{time.localtime().tm_hour}{time.localtime().tm_min}'
        file_name = f'model_{today}_{now_time}.pt'
        torch.save(model.state_dict(), file_name)
        print('✅ Model saved:', file_name)

    def _load_data(self, case):
        print("Checking dataset...")
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        if self.download:
            DownloadDataset(base_dir, url)
            SplitDataset(base_dir)

        data_dir = f"{base_dir}/{case}_12class"
        noise_dir = f"{base_dir}/_background_noise_"

        transform = transforms.Compose([Padding()])
        print(f"load {case} dataset..")
        
        specaugment = self.tau >= 1.5
        freq_masking = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        if case == 'train':
            self.train_dataset = PKMTLDataset(SpeechCommandWithSpeaker(data_dir, self.ver, transform=transform))
            self.train_loader=DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.pkmtl_collate, num_workers=self.num_workers, pin_memory=True)
            self.preprocess_train = Preprocess(noise_dir, self.device, specaug=specaugment, frequency_masking_para=freq_masking[self.tau])
            preprocess_and_save(self.train_dataset, self.preprocess_train, self.device, f"cached/{case}")
            self.speaker2idx=self.train_dataset.base.speaker2idx
        else: 
            self.valid_dataset = PKMTLDataset(SpeechCommandWithSpeaker(data_dir, self.ver, transform=transform))
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.pkmtl_collate, num_workers=self.num_workers, pin_memory=True)
            self.preprocess_test = Preprocess(noise_dir, self.device)
            preprocess_and_save(self.valid_dataset, self.preprocess_test, self.device, f"cached/{case}")

    def _load_model(self):
        self.model = BCResNets(int(self.tau * 8)).to(self.device)

if __name__ == "__main__":
    trainer = Trainer()
    trainer()
