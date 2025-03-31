import os
import time
import torch
import shutil
import numpy as np
from glob import glob
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pkmtl import PKMTLNet, train_epoch, evaluate, evaluate_with_far_frr
from model import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommandWithSpeaker, PKMTLDataset, SplitDataset


class Trainer:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--ver", default=1, type=int, help="GSC version")
        parser.add_argument("--tau", default=3, type=float, choices=[1, 1.5, 2, 3, 6, 8])
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--download", action="store_true")
        args = parser.parse_args()
        self.__dict__.update(vars(args))

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def __call__(self):
        total_epoch = 50
        learning_rate = 0.001
        embedding_dim = 128
        num_keywords = 12
        num_speakers = len(self.train_dataset.base.speaker2idx)
        batch_size = 64

        model = PKMTLNet(self.model, embedding_dim, num_keywords, num_speakers, alpha=0.5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        frr_list, far_list, acc_list = [], [], []
        alpha_grid = np.linspace(0.0, 1.0, 11)
        threshold_grid = np.linspace(-1.0, 1.0, 101)

        for split in range(10):
            print(f"\n🔁 Evaluating Split {split + 1}/10")
            for epoch in range(total_epoch):
                train_loss, loss_kws, loss_sv = train_epoch(model, self.train_loader, optimizer, self.device, self.preprocess_train)

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

    def _load_data(self):
        print("Checking dataset...")
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        if self.download:
            DownloadDataset(base_dir, url)
            SplitDataset(base_dir)

        train_dir = f"{base_dir}/train_12class"
        valid_dir = f"{base_dir}/valid_12class"
        noise_dir = f"{base_dir}/_background_noise_"

        transform = transforms.Compose([Padding()])
        self.train_dataset = PKMTLDataset(SpeechCommandWithSpeaker(train_dir, self.ver, transform=transform))
        self.valid_dataset = PKMTLDataset(SpeechCommandWithSpeaker(valid_dir, self.ver, transform=transform))
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=64, shuffle=False, num_workers=2)

        specaugment = self.tau >= 1.5
        freq_masking = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        self.preprocess_train = Preprocess(noise_dir, self.device, specaug=specaugment, frequency_masking_para=freq_masking[self.tau])
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        self.model = BCResNets(int(self.tau * 8)).to(self.device)


if __name__ == "__main__":
    trainer = Trainer()
    trainer()