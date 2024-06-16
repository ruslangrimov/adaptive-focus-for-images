import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import re

from utils import hard_softmax, get_position_encoding


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class GlobalMaxPooling2D(nn.Module):
    def forward(self, x):
        x, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        return x


class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x


class FeaturesModelPt5v0BN(nn.Module):
    pt_sz = 5  # Patch size
    pt_start = 0  #
    pt_step = 2

    def __init__(self, inp_channels=3, f_sz=32):
        super().__init__()

        self.f_sz = f_sz

        self.layers = nn.Sequential(
            nn.Conv2d(1, f_sz//2, 3, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz, 3, stride=2, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class FeaturesModelPt7v0BN(nn.Module):
    pt_sz = 7  # Patch size
    pt_start = 0  #
    pt_step = 4

    def __init__(self, inp_channels=3, f_sz=32):
        super().__init__()

        self.f_sz = f_sz

        self.layers = nn.Sequential(
            nn.Conv2d(inp_channels, f_sz//2, 3, stride=2, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz, 3, stride=2, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class FeaturesModelPt7v1BN(nn.Module):
    pt_sz = 7  # Patch size
    pt_start = 0  #
    pt_step = 2

    def __init__(self, inp_channels=3, f_sz=32):
        super().__init__()

        self.f_sz = f_sz

        self.layers = nn.Sequential(
            nn.Conv2d(inp_channels, f_sz//2, 3, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz, 3, stride=2, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 2, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class FeaturesModelPt9v2BN(nn.Module):
    pt_sz = 9  # Patch size
    pt_start = 0  #
    pt_step = 2

    def __init__(self, inp_channels=3, f_sz=32):
        super().__init__()

        self.f_sz = f_sz

        self.layers = nn.Sequential(
            nn.Conv2d(inp_channels, f_sz//2, 3, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz, 3, stride=2, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 3, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class FeaturesModelPt7v2BN(nn.Module):
    pt_sz = 7  # Patch size
    pt_start = 0  #
    pt_step = 2

    def __init__(self, inp_channels=3, f_sz=32):
        super().__init__()

        self.f_sz = f_sz

        self.layers = nn.Sequential(
            nn.Conv2d(inp_channels, f_sz//2, 3, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz//2, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz//2),
            nn.ReLU(),

            nn.Conv2d(f_sz//2, f_sz, 3, stride=2, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 2, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),

            nn.Conv2d(f_sz, f_sz, 1, stride=1, padding=0),
            nn.BatchNorm2d(f_sz),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class GlanceModelGSv1(nn.Module):
    def __init__(
        self, features_model_class, img_sz, inp_channels, num_classes,
        f_sz, p_sz, h_sz, layers_num, steps, r_dropout=0.0,
        trainable_abs_pos=True, init_abs_pos=True, gauss=False,
        is_full_train=False
    ):
        super().__init__()
        # img_sz - image width and height in pixels
        self.img_sz = img_sz
        self.inp_channels = inp_channels
        self.num_classes = num_classes
        self.f_sz = f_sz
        self.p_sz = p_sz
        self.h_sz = h_sz
        self.steps = steps
        self.trainable_abs_pos = trainable_abs_pos
        self.init_abs_pos = init_abs_pos
        self.gauss = gauss
        self.is_full_train = is_full_train

        # Model for generating features of thumbnail image
        self.features_model_g = features_model_class(inp_channels=self.inp_channels, f_sz=f_sz)
        # Model for generating features of image patches
        self.features_model_l = features_model_class(inp_channels=self.inp_channels, f_sz=f_sz)

        # This parameters depend on the features_model
        self.pt_sz = features_model_class.pt_sz
        self.pt_step = features_model_class.pt_step
        self.pt_start = features_model_class.pt_start

        self.grid_sz = (self.img_sz - self.pt_sz) // self.pt_step + 1

        if self.gauss:
            g_grid = torch.stack(torch.meshgrid([
                torch.linspace(0, 1, self.grid_sz),]*2, indexing='ij'
            ), -1).view(-1, 2)

            self.register_buffer('gauss_grid', g_grid)
        else:
            if self.init_abs_pos:
                h_pos_encoding = get_position_encoding(self.grid_sz, self.p_sz // 2, n=50)
                v_pos_encoding = get_position_encoding(self.grid_sz, self.p_sz // 2, n=50)
            else:
                h_pos_encoding = 0.02 * torch.rand(self.p_sz // 2, self.grid_sz)
                v_pos_encoding = 0.02 * torch.rand(self.p_sz // 2, self.grid_sz)

            if self.trainable_abs_pos:
                self.h_pos_encoding = nn.Parameter(h_pos_encoding)
                self.v_pos_encoding = nn.Parameter(v_pos_encoding)
            else:
                self.register_buffer('h_pos_encoding', h_pos_encoding)
                self.register_buffer('v_pos_encoding', v_pos_encoding)

        for indexing in ['ij', 'xy']:
            pt_coords = torch.stack(torch.meshgrid([
                torch.arange(self.pt_start, self.img_sz-self.pt_sz, self.pt_step),]*2, indexing=indexing
            ), -1).view(-1, 2)
            pt_coords = torch.cat([pt_coords, pt_coords+self.pt_sz], dim=1)

            self.register_buffer(f'pt_coords_{indexing}', pt_coords)

        self.rnn_model = nn.LSTM(
            self.f_sz + (2 if self.gauss else self.p_sz) + self.steps,
            self.h_sz,
            layers_num,
            dropout=r_dropout
        )

        if self.gauss:
            self.choice_model = nn.Sequential(
                nn.Linear(self.h_sz, 2),
                nn.Sigmoid()
            )
            self.choice_w = nn.Parameter(torch.tensor(self.grid_sz**0.5))
            # self.choice_w = nn.Parameter(torch.tensor(1.5*np.log(self.grid_sz)))
        else:
            self.choice_model = nn.Sequential(
                nn.Linear(self.h_sz, self.p_sz),
                nn.LayerNorm(self.p_sz),
            )

        # Model for final prediction
        self.predict_model = nn.Sequential(
            nn.BatchNorm1d(self.h_sz),
            nn.Linear(self.h_sz, self.num_classes, bias=False)
        )

        if self.is_full_train:
            self.adv_block0 = nn.Sequential(
                nn.Conv2d(self.f_sz, 2*self.f_sz, 3, stride=2, padding=0),
                nn.BatchNorm2d(2*self.f_sz),
                nn.ReLU(),

                nn.Conv2d(2*self.f_sz, 2*self.f_sz, 3, stride=2, padding=0),
                nn.BatchNorm2d(2*self.f_sz),
                nn.ReLU(),

                GlobalMaxPooling2D()
            )

            self.adv_block1 = nn.Sequential(
                nn.Conv2d(self.f_sz, 2*self.f_sz, 1, stride=1, padding=0),
                nn.BatchNorm2d(2*self.f_sz),
                nn.ReLU(),
                nn.Flatten(1, -1)
            )

            self.adv_block3 = nn.Sequential(
                nn.Linear(2*self.f_sz, 2*self.f_sz),
                nn.BatchNorm1d(2*self.f_sz),
                nn.ReLU(),

                nn.Linear(2*self.f_sz, self.num_classes),
                nn.Flatten(1, -1)
            )

    def separate_parameters(self, no_decay=False):
        params = []

        no_decay_ptns = [
            'h_pos_encoding',
            'v_pos_encoding',
            'choice_model',
            'choice_w'
        ]

        # smaller_lr_ptns = []

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay_ptns = any([re.match(p, n) for p in no_decay_ptns]) or n.endswith('bias')
            if no_decay != is_no_decay_ptns:
                continue
            params.append(p)

        return params

    def forward(
        self, imgs, hard=True, tau=1.0, add_gumbels=False, debug=False,
        is_train=None, is_full_train=False
        ):
        if is_train is None:
            is_train = self.training

        b_sz = imgs.size(0)
        device = imgs.device

        if is_train:
            if is_full_train:
                features_2d = self.features_model_l(imgs)
                features = features_2d.view(b_sz, -1, self.grid_sz**2).transpose(1, 2)
            else:
                self.features_model_l.eval()
                with torch.no_grad():
                    features = self.features_model_l(imgs).view(b_sz, -1, self.grid_sz**2).transpose(1, 2)
                self.features_model_l.train(self.training)

            # Is doesn't copy any memory. Just creates a view
            patches = F.unfold(imgs, (self.pt_sz, self.pt_sz), dilation=1, padding=0, stride=self.pt_step).\
                view(-1, self.inp_channels, self.pt_sz, self.pt_sz, self.grid_sz, self.grid_sz).permute(0, 4, 5, 1, 2, 3)
            patches = patches.view(b_sz, self.grid_sz*self.grid_sz, self.inp_channels, self.pt_sz,self.pt_sz)

        # For the first step
        next_poses = torch.zeros((b_sz, self.grid_sz*self.grid_sz), device=device)
        next_poses[:, self.grid_sz**2 // 2 + self.grid_sz//2] = 1.0
        h_state_ = None

        if debug:
            all_p_next_poses, all_next_poses, all_next_patches, all_h_states = [], [], [], []

        for n in range(self.steps):
            next_poses_idx = next_poses.argmax(-1)

            if self.gauss:
                chosen_pos_enc = self.gauss_grid[next_poses_idx]
            else:
                chosen_pos_enc = torch.cat([
                    self.v_pos_encoding[next_poses_idx // self.grid_sz],
                    self.h_pos_encoding[next_poses_idx % self.grid_sz]
                ], dim=-1)

            if n == 0:
                next_patches = F.interpolate(imgs, (self.features_model_g.pt_sz,)*2, mode='bilinear', antialias=True)
                g_features = self.features_model_g(next_patches).view(b_sz, self.f_sz)
                chosen_features = g_features
            else:
                if is_train:
                    next_patches = patches[torch.arange(b_sz), next_poses_idx]
                else:
                    # b_pt_coords = self.pt_coords_ij[next_poses_idx]
                    # next_patches = torch.stack([imgs[b, :, p[0]:p[2], p[1]:p[3]] for b, p in enumerate(b_pt_coords)], dim=0)
                    next_patches = torchvision.ops.roi_pool(
                        imgs,
                        torch.cat([
                            torch.arange(b_sz, device=device)[:, None],
                            self.pt_coords_xy[next_poses_idx]+torch.tensor([0, 0, -1, -1], device=device)
                        ], dim=1).float(),
                        self.pt_sz
                    )

                chosen_features_hard = self.features_model_l(next_patches).view(b_sz, self.f_sz)

                if is_train:
                    chosen_features = (next_poses[:, :, None] * features).sum(1)
                    chosen_features = chosen_features_hard + chosen_features - chosen_features_hard.detach()
                else:
                    chosen_features = chosen_features_hard

            tm_enc = torch.zeros((b_sz, self.steps), device=device)
            tm_enc[:, n] = 1.0

            inp = torch.cat([
                chosen_features,
                chosen_pos_enc,
                tm_enc,
            ], dim=1)

            out, h_state_ = self.rnn_model(inp[None], h_state_)  # (h_n, c_n)
            h_state = h_state_[1][-1]

            if debug:
                all_p_next_poses.append(next_poses if n == 0 else p_next_poses)
                all_next_poses.append(next_poses)
                all_next_patches.append(next_patches)
                all_h_states.append(h_state)

            if n < self.steps - 1:
                next_desc = self.choice_model(h_state)
                if self.gauss:
                    # Get probabilities of next patches using their distance to point
                    p_next_poses = - self.choice_w**2 * (next_desc[:, None] - self.gauss_grid[None]).norm(p=2, dim=-1)
                else:
                    # Get probabilities of next patches using their similarity to descriptor
                    p_next_poses = (
                        ((next_desc[:, :self.p_sz//2] @ self.v_pos_encoding.T)[:, None] +
                         (next_desc[:, self.p_sz//2:] @ self.h_pos_encoding.T)[:, :, None]).view(b_sz, -1)
                    ) / self.p_sz**0.5
                # Choose next patches using their probabilities
                next_poses = hard_softmax(p_next_poses, tau=tau, hard=hard, add_gumbels=add_gumbels)

        logits = self.predict_model(h_state)

        if is_train and is_full_train:
            assert self.is_full_train, "The model should have been created with is_full_train=True"
            logits_a = self.adv_block3(self.adv_block0(features_2d) + self.adv_block1(g_features[..., None, None]))
            logits = (logits, logits_a)

        if debug:
            return logits, (all_p_next_poses, all_next_poses, all_next_patches, all_h_states)
        else:
            return logits


def get_patches_frame(input_frames, actions, patch_size, image_size, in_ch=3):
    input_frames = input_frames.view(-1, in_ch, image_size, image_size)  # [NT,C,H,W]
    theta = torch.zeros(input_frames.size(0), 2, 3).to(input_frames.device)
    patch_coordinate = (actions * (image_size - patch_size))
    x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size, \
                     patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size

    theta[:, 0, 0], theta[:, 1, 1] = patch_size / image_size, patch_size / image_size
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size

    grid = F.affine_grid(theta.float(), torch.Size((input_frames.size(0), 3, patch_size, patch_size)), align_corners=True)
    patches = F.grid_sample(input_frames, grid, align_corners=True)  # [NT,C,H1,W1] align_corners=True - the old behavior
    return patches


class GlanceModelBLv1(nn.Module):
    def __init__(
        self, features_model_class, img_sz, inp_channels, num_classes,
        f_sz, h_sz, layers_num, steps, r_dropout=0.0
    ):
        super().__init__()
        # img_sz - image width and height in pixels
        self.img_sz = img_sz
        self.inp_channels = inp_channels
        self.num_classes = num_classes
        self.f_sz = f_sz
        self.h_sz = h_sz
        self.steps = steps

        # Model for generating features of thumbnail image
        self.features_model_g = features_model_class(inp_channels=self.inp_channels, f_sz=f_sz)
        # Model for generating features of image patches
        self.features_model_l = features_model_class(inp_channels=self.inp_channels, f_sz=f_sz)

        # This parameters depend on the features_model
        self.pt_sz = features_model_class.pt_sz

        self.rnn_model = nn.LSTM(
            self.f_sz + 2 + self.steps,
            self.h_sz,
            layers_num,
            dropout=r_dropout
        )

        self.choice_model = nn.Sequential(
            nn.Linear(self.h_sz, 2),
            nn.Sigmoid()
        )

        # Model for final prediction
        self.predict_model = nn.Sequential(
            nn.BatchNorm1d(self.h_sz),
            nn.Linear(self.h_sz, self.num_classes, bias=False)
        )

    def separate_parameters(self, no_decay=False):
        params = []

        no_decay_ptns = [
            'choice_model',
            'choice_w'
        ]

        # smaller_lr_ptns = []

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay_ptns = any([re.match(p, n) for p in no_decay_ptns]) or n.endswith('bias')
            if no_decay != is_no_decay_ptns:
                continue
            params.append(p)

        return params

    def forward(self, imgs, debug=False, is_train=None):
        if is_train is None:
            is_train = self.training

        b_sz = imgs.size(0)
        device = imgs.device

        # For the first step
        actions = torch.full((b_sz, 2), 0.5, device=device)
        h_state_ = None

        if debug:
            all_actions, all_next_patches, all_h_states = [], [], []

        for n in range(self.steps):
            if n == 0:
                next_patches = F.interpolate(imgs, (self.features_model_g.pt_sz,)*2, mode='bilinear', antialias=True)
                chosen_features = self.features_model_g(next_patches).view(b_sz, self.f_sz)
            else:
                next_patches = get_patches_frame(imgs, actions, self.pt_sz, self.img_sz, in_ch=self.inp_channels)
                chosen_features = self.features_model_l(next_patches).view(b_sz, self.f_sz)

            tm_enc = torch.zeros((b_sz, self.steps), device=device)
            tm_enc[:, n] = 1.0

            inp = torch.cat([
                chosen_features,
                actions.detach(),
                tm_enc,
            ], dim=1)

            out, h_state_ = self.rnn_model(inp[None], h_state_)  # (h_n, c_n)
            h_state = h_state_[1][-1]

            if debug:
                all_actions.append(actions)
                all_next_patches.append(next_patches)
                all_h_states.append(h_state)

            if n < self.steps - 1:
                actions = self.choice_model(h_state)

        logits = self.predict_model(h_state)

        if debug:
            return logits, (all_actions, all_next_patches, all_h_states)
        else:
            return logits
