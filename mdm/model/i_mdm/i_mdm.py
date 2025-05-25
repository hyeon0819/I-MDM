import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################
# (A) Simple G-Conv 
##################################################
class SimpleGConv(nn.Module):
    """
    - channels: (C//2)
    - edges: learnable adjacency W shape=(V, V)
    - input shape: [B, C, T, V]
    """
    def __init__(self, num_points, in_channels):
        super().__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.A = nn.Parameter(torch.randn(num_points, num_points))
        nn.init.trunc_normal_(self.A, std=0.02)

    def forward(self, x):
        """
        x: [B, C, T, V]
        - C = in_channels
        """
        #  Einstein sum: z_{b,c,t,v} = sum_{u} x_{b,c,t,u} * A[u,v]
        # x chunk to a portion if needed
        z = torch.einsum('b c t u, u v -> b c t v', x, self.A)
        return z


##################################################
# (B) Simple T-Conv
##################################################
class SimpleTConv(nn.Module):
    """
    - kernel_size=3, groups=1
    """
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=(kernel_size,1),
                              padding=(padding,0),
                              groups=1,
                              bias=False)
    def forward(self, x):
        """
        x: [B, C, T, V]
        - kernel_size=(3,1) 1D conv
        """
        return self.conv(x)
    


class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(self.reduction(x))
        x = x.permute(0, 2, 1)
        return x




##################################################
# (C) Skate-Embedding in a simplified manner
##################################################


class SkateEmbedding(nn.Module):
    """
    - Joint Embedding: learnable [num_joints, d_model]
    - Time Embedding: fixed sinusoidal for [T, d_model]
    => outer-product => shape (T, V, d_model)
    => transpose => [1, T, V, d_model] and broadcast to input shape
    """
    def __init__(self, num_joints, d_model, max_frames=100):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.max_frames = max_frames

        # (1) learnable skeleton embedding: shape [V, d_model]
        self.skel_embed = nn.Parameter(torch.zeros(num_joints, d_model))
        nn.init.trunc_normal_(self.skel_embed, std=0.02)

        # (2) fixed time embedding (sinusoidal): shape [max_frames, d_model]
        pe = torch.zeros(self.max_frames, d_model)
        position = torch.arange(0, self.max_frames, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('time_embed', pe)  # [max_frames, d_model]

    def forward(self, B, T, V):
        """
        return shape: [T, V, d_model]
        """
        # clamp T <= max_frames
        T_ = min(T, self.max_frames)
        time_part = self.time_embed[:T_]  # shape [T_, d_model]
        # skeleton part: shape [V, d_model]
        # outer product => [T_, V, d_model]
        # broadcast => time_part[:, None, d_model] * skel_embed[None, :, d_model]
        embed = time_part.unsqueeze(1) * self.skel_embed.unsqueeze(0)
        # if T < max_frames, then embed has shape [T, V, d_model]
        # if T>max_frames, we only have partial 
        if T_ < T:
            # pad zeros
            pad = torch.zeros(T - T_, V, self.d_model, device=embed.device, dtype=embed.dtype)
            embed = torch.cat([embed, pad], dim=0)  # [T, V, d_model]
        return embed

class GTModule(nn.Module):
    def __init__(self, num_joints, d_model):
        super().__init__()
        self.gconv = SimpleGConv(num_points=(2*num_joints), in_channels=(d_model//4))
        self.tconv = SimpleTConv(in_channels=(d_model//4), kernel_size=3)
        self.CLinear = nn.Identity()
        pass

    def forward(self, x):
        # but x shape [B, T, d_model]. We want [B, d_model, T, 1?] => let's do
        x_4d = x.permute(0,2,1).unsqueeze(-1)  # => [B, d_model, T, 1]
        c = x_4d.shape[1]
        # split => c//2, c - c//2
        xg, xc, xt = torch.split(x_4d, [c//4, c//2, c//4], dim=1)  # 2 lumps
        out_list = []

        xg_out = self.gconv(xg)   # => same shape [B,c//2,T,V]
        xg_out = xg_out.mean(dim=3, keepdim=True)  # [B, c//2, T, 1]
        out_list.append(xg_out + xg)  # skip real GConv or do a fake identity

        out_list.append(self.CLinear(xc))

        xt_out = self.tconv(xt)  # [B, c//2, T, 1]
        out_list.append(xt_out + xt)

        x_cat = torch.cat(out_list, dim=1)  # [B, c, T, 1]
        # flatten back => [B, T, c]
        x = x_cat.squeeze(-1).permute(0,2,1).contiguous()  # => [B, T, d_model]

        return x


class DSEncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, nhead, dim_feedforward, activation, dropout, down_sample = False):
        if down_sample:
            self.dsample = PatchMergingTconv(in_dim, out_dim)
        else:
            self.dsample = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # here we can use batch_first=True for convenience
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.dsample1(x)
        out = self.transformer_encoder2(x)  # => [B, T, d_model]
        return out


##################################################
# (D) My Inverse MDM + partial Skate approach
##################################################
class I_MDM_SkateTrans(nn.Module):
    def __init__(
        self,
        num_actions=120,
        num_joints=25,  # 24+root
        n_feats=6,
        d_model=512,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_frames=100,
        activation='relu',
        use_gconv=True,
        use_tconv=True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_joints = num_joints
        self.n_feats = n_feats
        self.d_model = d_model
        self.use_gconv = use_gconv
        self.use_tconv = use_tconv

        # shape: [2, 25, 6] = 2*25*6=300 per frame
        self.per_frame_dim = 2 * num_joints * n_feats

        # (1) linear embedding
        self.stem = nn.ModuleList([nn.Linear(self.per_frame_dim, self.per_frame_dim*2),
                nn.GELU(),
                nn.Linear(self.per_frame_dim*2, self.per_frame_dim*3),
                nn.GELU(),
                nn.Linear(self.per_frame_dim*3, d_model)])

        #self.embedding = nn.ModuleList(stem)
        self.embedding = nn.Linear(self.per_frame_dim, d_model)

        # (3) Skate-Embedding 
        self.skate_embed = SkateEmbedding(
            num_joints=(2*num_joints),
            d_model=d_model,
            max_frames=max_frames
        )

        # (4) Transformer (same as original)

        self.gtmodule1 = GTModule(num_joints=num_joints, d_model=d_model)
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # here we can use batch_first=True for convenience
        )
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layer1,
            num_layers=num_layers
        )
        self.dsample1 = PatchMergingTconv(d_model, d_model*2)

        self.gtmodule2 = GTModule(num_joints=num_joints, d_model=d_model*2)
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=d_model*2,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # here we can use batch_first=True for convenience
        )
        self.transformer_encoder2 = nn.TransformerEncoder(
            encoder_layer2,
            num_layers=num_layers
        )

        self.dsample2 = PatchMergingTconv(d_model*2, d_model*2)

        self.gtmodule3 = GTModule(num_joints=num_joints, d_model=d_model*2)
        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=d_model*2,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # here we can use batch_first=True for convenience
        )
        self.transformer_encoder3 = nn.TransformerEncoder(
            encoder_layer3,
            num_layers=num_layers
        )

        self.dsample3 = PatchMergingTconv(d_model*2, d_model*2)

        self.gtmodule4 = GTModule(num_joints=num_joints, d_model=d_model*2)
        encoder_layer4 = nn.TransformerEncoderLayer(
            d_model=d_model*2,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # here we can use batch_first=True for convenience
        )
        self.transformer_encoder4 = nn.TransformerEncoder(
            encoder_layer4,
            num_layers=num_layers
        )
        # (5) classification head
        self.fc_out = nn.Linear(d_model*2, num_actions)
    def forward(self, x, return_features=False):
        """
        x: [B, 2, 25, 6, T]
        """
        B, P, J, C, T = x.shape
        # 0) rearrange => [B, T, P, J, C] => flatten => [B, T, 2*25*6]
        x = x.permute(0, 4, 1, 2, 3).reshape(B, T, -1)  # => [B, T, 300]

        # 1) linear embedding => [B, T, d_model]
        for l in self.stem:
            x = l(x)
        
        embed_2d = self.skate_embed(B, T, (2*self.num_joints))  # [T, V, d_model]

        time_embed_only = embed_2d.mean(dim=1)   # [T, d_model]
        # broadcast
        x = x + time_embed_only.unsqueeze(0).to(x.device)  # [1, T, d_model]

        x = self.gtmodule1(x)

        i_mdm_features = []
        if return_features:
            x_enc = x
            for layer in self.transformer_encoder1.layers:
                x_enc = layer(x_enc)
                i_mdm_features.append(x_enc.clone())
            x = x_enc
        else:
            x = self.transformer_encoder1(x)  # => [B, T, d_model]
        # => [B, T, d_model]
        x = self.dsample1(x)

        x = self.gtmodule2(x)
        if return_features:
            x_enc = x
            for layer in self.transformer_encoder2.layers:
                x_enc = layer(x_enc)
                i_mdm_features.append(x_enc.clone())
            x = x_enc
        else:
            x = self.transformer_encoder2(x)  # => [B, T, d_model]
        # => [B, T, d_model]  # => [B, T, d_model]

        x = self.dsample2(x)

        x = self.gtmodule3(x)
        if return_features:
            x_enc = x
            for layer in self.transformer_encoder3.layers:
                x_enc = layer(x_enc)
                i_mdm_features.append(x_enc.clone())
            x = x_enc
        else:
            x = self.transformer_encoder3(x)  # => [B, T, d_model]
        # => [B, T, d_model]  # => [B, T, d_model]

        x = self.dsample3(x)
        
        x = self.gtmodule4(x)
        if return_features:
            x_enc = x
            for layer in self.transformer_encoder4.layers:
                x_enc = layer(x_enc)
                i_mdm_features.append(x_enc.clone())
            x = x_enc
        else:
            x = self.transformer_encoder4(x)  # => [B, T, d_model]

        # 5) gather => mean pool => [B, d_model]
        out_mean = x.mean(dim=1)
        # 6) final classification
        logits = self.fc_out(out_mean)

        if return_features:
            return logits, i_mdm_features
        else:
            return logits
