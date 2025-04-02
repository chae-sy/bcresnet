import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BCResNets, ConvBNReLU  # assuming this is the original BCResNet
from torch.nn.functional import normalize
from torch.nn.functional import cosine_similarity

import math


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = normalize(x, dim=1)
        w = normalize(self.weight, dim=1)
        return self.scale * torch.matmul(x, w.t())


class SharedEncoder(nn.Module):
    def __init__(self, bcresnet_tau3: BCResNets, num_shared_stages=2):
        super().__init__()
        self.cnn_head = bcresnet_tau3.cnn_head
        self.body = nn.ModuleList(bcresnet_tau3.BCBlocks[:num_shared_stages])

        # Get output channels from the last block in the last kept stage
        last_stage = bcresnet_tau3.BCBlocks[num_shared_stages - 1]
        last_block = last_stage[-1]  # last block in that stage
        self.out_channels = last_block.f1[0].block[0].out_channels  # get from Conv2d layer

    def forward(self, x):
        x = self.cnn_head(x)
        for stage in self.body:
            for block in stage:
                x = block(x)
        return x



class SubNet(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, idx=4),
            ConvBNReLU(in_channels, in_channels, idx=5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
        )

    def forward(self, x):
        return self.blocks(x)



class SCM(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, psi_k, psi_s):
        return self.alpha * psi_k + (1 - self.alpha) * psi_s


class TRM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn_fc1 = nn.Linear(in_dim * 2, 2)
        self.attn_fc2 = nn.Linear(2, 2)

    def forward(self, z_k, z_s):
        z_k = normalize(z_k, dim=1)
        z_s = normalize(z_s, dim=1)
        z = torch.cat([z_k, z_s], dim=1)
        attn = self.attn_fc2(F.relu(self.attn_fc1(z)))
        attn = F.softmax(attn, dim=1)
        z_task = attn[:, 0:1] * z_k + attn[:, 1:2] * z_s
        return z_task


class PKMTLNet(nn.Module):
    def __init__(self, backbone: BCResNets, embedding_dim=128, num_keywords=12, num_speakers=1881, alpha=0.5):
        super().__init__()
        self.shared_encoder = SharedEncoder(backbone, num_shared_stages=2)
        shared_out_channels = self.shared_encoder.out_channels

        self.kws_subnet = SubNet(in_channels=shared_out_channels, out_dim=embedding_dim)
        self.sv_subnet = SubNet(in_channels=shared_out_channels, out_dim=embedding_dim)

        self.kws_classifier = CosineClassifier(embedding_dim, num_keywords)
        self.sv_classifier = CosineClassifier(embedding_dim, num_speakers)
        self.scm = SCM(alpha=alpha)
        self.trm = TRM(in_dim=embedding_dim)
        self.lambda_speaker_loss = 0.1

    def forward(self, x, task='mtl', return_embeddings=False):
        shared_feat = self.shared_encoder(x)
        z_k = self.kws_subnet(shared_feat)
        z_s = self.sv_subnet(shared_feat)

        if task == 'mtl':
            out_kws = self.kws_classifier(z_k)
            out_sv = self.sv_classifier(z_s)
            return out_kws, out_sv

        elif task == 'scm':
            return z_k, z_s

        elif task == 'trm':
            z_task = self.trm(z_k, z_s)
            if return_embeddings:
                return z_task
            return z_task

        else:
            raise ValueError("Unsupported task type.")


def compute_mtl_loss(out_kws, out_sv, label_kws, label_sv, lambda_speaker=0.1):
    loss_kws = F.cross_entropy(out_kws, label_kws)
    loss_sv = F.cross_entropy(out_sv, label_sv)
    loss = loss_kws + lambda_speaker * loss_sv
    return loss, loss_kws.item(), loss_sv.item()


def train_epoch(model, dataloader, optimizer, device, preprocess_fn):
    model.train()
    total_loss, total_kws, total_sv = 0.0, 0.0, 0.0
    for batch in dataloader:
        anchor, target_label, target_speaker = map(lambda t: t.to(device, non_blocking=True), 
                                           (batch['anchor'], batch['target_label'], batch['target_speaker']))

        x = preprocess_fn(anchor, target_label)
        label_kws = target_label
        label_sv = target_speaker

        optimizer.zero_grad()
        out_kws, out_sv = model(x, task='mtl')
        loss, loss_kws, loss_sv = compute_mtl_loss(out_kws, out_sv, label_kws, label_sv)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kws += loss_kws
        total_sv += loss_sv

    return total_loss / len(dataloader), total_kws / len(dataloader), total_sv / len(dataloader)


def evaluate(model, dataloader, device, preprocess_fn):
    model.eval()
    correct_kws, correct_sv = 0, 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            anchor, target_label, target_speaker = map(lambda t: t.to(device, non_blocking=True), 
                                           (batch['anchor'], batch['target_label'], batch['target_speaker']))

            x = preprocess_fn(anchor, target_label)
            label_kws = target_label
            label_sv = target_speaker
            out_kws, out_sv = model(x, task='mtl')
            pred_kws = out_kws.argmax(dim=1)
            pred_sv = out_sv.argmax(dim=1)
            correct_kws += (pred_kws == label_kws).sum().item()
            correct_sv += (pred_sv == label_sv).sum().item()
            total += label_kws.size(0)

    acc_kws = correct_kws / total
    acc_sv = correct_sv / total
    return acc_kws, acc_sv




def evaluate_with_far_frr(model, dataloader, device, threshold=0.0, task='scm', preprocess_fn=None):
    """
    Evaluate FAR and FRR based on task-specific similarity scores (SCM or TRM)
    """
    model.eval()
    num_false_accept = 0
    num_false_reject = 0
    num_positive = 0
    num_negative = 0

    with torch.no_grad():
        for batch in dataloader:
            ts_tk = preprocess_fn(batch['ts_tk'][0].unsqueeze(0).to(device), [batch['target_label']]).to(device)
            ts_ntk = preprocess_fn(batch['ts_ntk'][0].unsqueeze(0).to(device), [batch['target_label']]).to(device)
            nts_tk = preprocess_fn(batch['nts_tk'][0].unsqueeze(0).to(device), [batch['target_label']]).to(device)
            nts_ntk = preprocess_fn(batch['nts_ntk'][0].unsqueeze(0).to(device), [batch['target_label']]).to(device)

            if task == 'scm':
                z_k_pos, z_s_pos = model(ts_tk, task='scm')
                z_k_neg1, z_s_neg1 = model(ts_ntk, task='scm')
                z_k_neg2, z_s_neg2 = model(nts_tk, task='scm')
                z_k_neg3, z_s_neg3 = model(nts_ntk, task='scm')

                sim_pos = model.scm(cosine_similarity(z_k_pos, z_k_pos), cosine_similarity(z_s_pos, z_s_pos))
                sim_neg1 = model.scm(cosine_similarity(z_k_neg1, z_k_pos), cosine_similarity(z_s_neg1, z_s_pos))
                sim_neg2 = model.scm(cosine_similarity(z_k_neg2, z_k_pos), cosine_similarity(z_s_neg2, z_s_pos))
                sim_neg3 = model.scm(cosine_similarity(z_k_neg3, z_k_pos), cosine_similarity(z_s_neg3, z_s_pos))

            elif task == 'trm':
                z_task_pos = model(ts_tk, task='trm', return_embeddings=True)
                z_task_neg1 = model(ts_ntk, task='trm', return_embeddings=True)
                z_task_neg2 = model(nts_tk, task='trm', return_embeddings=True)
                z_task_neg3 = model(nts_ntk, task='trm', return_embeddings=True)

                sim_pos = cosine_similarity(z_task_pos, z_task_pos)
                sim_neg1 = cosine_similarity(z_task_neg1, z_task_pos)
                sim_neg2 = cosine_similarity(z_task_neg2, z_task_pos)
                sim_neg3 = cosine_similarity(z_task_neg3, z_task_pos)

            else:
                raise ValueError("Invalid task: choose 'scm' or 'trm'")

            # Positive: should be above threshold
            if sim_pos < threshold:
                num_false_reject += 1
            num_positive += 1

            # Negatives: should be below threshold
            for sim_neg in [sim_neg1, sim_neg2, sim_neg3]:
                if sim_neg >= threshold:
                    num_false_accept += 1
                num_negative += 1

    frr = num_false_reject / num_positive if num_positive > 0 else 0.0
    far = num_false_accept / num_negative if num_negative > 0 else 0.0
    return frr, far
