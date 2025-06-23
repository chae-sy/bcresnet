# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# revision
import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm

from bcresnet import BCResNets, PACTActivation
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset


class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch =20
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))\
                
            ## ---- PACT ---- ##
            for name, m in self.model.named_modules():
                if isinstance(m, PACTActivation):
                    print(f"epoch {epoch:02d} │ {name}.alpha = {m.alpha.item():.4f}")

        test_acc = self.Test(self.test_dataset, self.test_loader, augment=True)  # official testset
        print("test acc: %.3f" % (test_acc))
        print("End.")
        self.save_model(self.model) 

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        torch.save(dataset, '40x101_mnist.pt')
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            #inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0  # percentage
        return acc

    def save_model(self, model):
        import torch
        import time
        month=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][time.localtime().tm_mon-1]
        date=time.localtime().tm_mday
        today=f'{month}{date}'
        time=f'{time.localtime().tm_hour}{time.localtime().tm_min}'
        file_name=f'model_{today}_{time}.pt'
        torch.save(model, file_name)
        print('model saved : ', file_name)

    def _load_data(self):
        """
        Private method that loads data into the object.
        Downloads and splits the data if necessary.
         """
        # 1) Transform 정의
        transform = transforms.Compose([
            transforms.Resize((40, 101)),
            transforms.ToTensor(),
        ])

        # 2) 전체 학습 데이터 다운로드 및 train/val split
        full_train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        total_train = len(full_train_dataset)
        val_size = int(total_train * 0.1)
        train_size = total_train - val_size

        self.train_dataset, self.valid_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 3) 테스트 데이터셋
        self.test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        # 4) DataLoader 설정
        batch_size = 64
        num_workers = 2

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # 5) 로더 배치 확인
        for loader, name in zip(
            [self.train_loader, self.valid_loader, self.test_loader],
            ['Train', 'Validation', 'Test']
        ):
            images, labels = next(iter(loader))
            print(f"{name} loader batch shape: {images.shape}")
            # 예: torch.Size([64, 1, 40, 101])
    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)



if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()
    torch.save(_trainer.model.state_dict(),"model_params.pth")
    print("params saved")
    torch.save(_trainer.model, "model.pth")
    print("model saved")
