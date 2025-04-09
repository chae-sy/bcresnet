import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BCResNets, ConvBNReLU  # assuming this is the original BCResNet
from torch.nn.functional import normalize
from torch.nn.functional import cosine_similarity

import math


class CosineClassifier(nn.Module):
    """
     w * cosine_similarity (z, w_c) + b
    
    w = learnable scaling factor (self.scale)
    b = optional bias term (not included in the below code)
    z = input embedding (x)  shape: (batch_size, in_dim)
    w_c = class weight vector (self.weight) shape: (num_classes, in_dim)
    
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_dim)) #self.weight.shape = (num_classes, in_dim) # W \in R^{C×d} W_k, the prototype for class k.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) #initialize the self.weight values using a Kaiming uniform distribution
        self.scale = nn.Parameter(torch.ones(1)) #initialize the self.sacle by tensor [1]

    def forward(self, x):
        x = normalize(x, dim=1)
        w = normalize(self.weight, dim=1)
        return self.scale * torch.matmul(x, w.t()) #output shape: [batch_size, num_classes]

class SharedEncoder(nn.Module):
    def __init__(self, bcresnet: BCResNets, num_shared_blocks=10):
        super().__init__()
        self.cnn_head = bcresnet.cnn_head
        
        # Flatten all BCResBlocks
        all_blocks = [block for stage in bcresnet.BCBlocks for block in stage]

        # First 10 go to shared encoder
        self.blocks = nn.ModuleList(all_blocks[:num_shared_blocks])

        # Dynamically infer output channels from the last block
        last_block = self.blocks[-1]
        self.out_channels = last_block.f1[0].block[0].out_channels


    def forward(self, x):
        x = self.cnn_head(x)
        for block in self.blocks:
            x = block(x)
        return x



class SubNet(nn.Module):
    def __init__(self, bcresnet: BCResNets, start_block=10, out_dim=128):
        super().__init__()
        # Flatten all BCResBlocks
        all_blocks = [block for stage in bcresnet.BCBlocks for block in stage]

        # Get the last 2 blocks
        self.blocks = nn.Sequential(*all_blocks[start_block:])

        # Dynamically determine input channels from first block
        in_channels = all_blocks[start_block].f1[0].block[0].in_channels

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim)
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.head(x)



class SCM(nn.Module):
    """
    simple linear scoring function for both task
    psi_task = alpha * psi_kws + (1-alpha) * psi_sv
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, psi_k, psi_s):
        return self.alpha * psi_k + (1 - self.alpha) * psi_s


class TRM(nn.Module):
    """
    neural network based scoring function for both task
    """
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
        self.shared_encoder = SharedEncoder(backbone, num_shared_blocks=10)
        shared_out_channels = self.shared_encoder.out_channels

        self.kws_subnet = SubNet(backbone, start_block=10, out_dim=128)
        self.sv_subnet = SubNet(backbone, start_block=10, out_dim=128)

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

from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, device, kws_criterion, sv_criterion):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # === 1) COLLATE ===
        # batch is a list of sample‐dicts, so stack each key
        batch = {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in batch[0].keys()
        }

        # === 2) MOVE TO DEVICE ===
        for k, v in batch.items():
            batch[k] = v.to(device)

        # === 3) UNPACK ===
        a   = batch['anchor']
        t1  = batch['ts_tk']
        t2  = batch['ts_ntk']
        n1  = batch['nts_tk']
        n2  = batch['nts_ntk']
        lbl = batch['target_label']
        spk = batch['target_speaker']

        # === 4) STAGE 1: MTL CLASSIFICATION LOSS (Eq.3) ===
        out_kws, out_sv = model(a, task='mtl')
        kw_loss  = kws_criterion(out_kws, lbl)
        sv_loss  = sv_criterion(out_sv, spk) * model.lambda_speaker_loss
        mtl_loss = kw_loss + sv_loss

        # === 5) STAGE 2: TRM METRIC LOSS ===
        # get task‐specific embeddings
        z_t_a   = model(a,  task='trm', return_embeddings=True)
        z_t_t1  = model(t1, task='trm', return_embeddings=True)
        z_t_t2  = model(t2, task='trm', return_embeddings=True)
        z_t_n1  = model(n1, task='trm', return_embeddings=True)
        z_t_n2  = model(n2, task='trm', return_embeddings=True)

        trm_loss = model.trm.angular_prototypical_loss(
            anchor   = z_t_a,
            pos_same = z_t_t1,
            neg_same = z_t_n1,
            pos_diff = z_t_t2,
            neg_diff = z_t_n2
        )

        # === 6) BACKWARD & STEP ===
        loss = mtl_loss + trm_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # === 7) UPDATE PROGRESS BAR ===
        pbar.set_postfix({
            "kw_loss":  f"{kw_loss.item():.4f}",
            "sv_loss":  f"{sv_loss.item():.4f}",
            "trm_loss": f"{trm_loss.item():.4f}"
        })

    return total_loss / len(train_loader)

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
            label = batch['target_label']
            if torch.is_tensor(label) and label.dim() > 0:
                label = label[0].item()  # safely extract scalar
            elif torch.is_tensor(label):
                label = label.item()

            ts_tk = preprocess_fn(batch['ts_tk'][0].unsqueeze(0).to(device), [label]).to(device)
            ts_ntk = preprocess_fn(batch['ts_ntk'][0].unsqueeze(0).to(device), [label]).to(device)
            nts_tk = preprocess_fn(batch['nts_tk'][0].unsqueeze(0).to(device), [label]).to(device)
            nts_ntk = preprocess_fn(batch['nts_ntk'][0].unsqueeze(0).to(device), [label]).to(device)

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
