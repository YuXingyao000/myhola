import copy
import time
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
import torchvision

from src.brepnet.dataset import denormalize_coord1112

def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()


def profile_time(time_dict, key, v_timer):
    torch.cuda.synchronize()
    cur = time.time()
    time_dict[key] += cur - v_timer
    return cur


class res_linear(nn.Module):
    def __init__(self, dim_in, dim_out, norm="none"):
        super(res_linear, self).__init__()
        self.conv = nn.Linear(dim_in, dim_out)
        self.act = nn.LeakyReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.LayerNorm(dim_out)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_1D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm="none"):
        super(res_block_1D, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h -> ... h c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h c -> ... c h'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_2D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm="none"):
        super(res_block_2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h w -> ... h w c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h w c -> ... c h w'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_xd(nn.Module):
    def __init__(self, dim, dim_in, dim_out, kernel_size=3, stride=1, padding=1, v_norm=None, v_norm_shape=None):
        super(res_block_xd, self).__init__()
        self.downsample = None
        if v_norm is None or  v_norm == "none":
            norm = nn.Identity()
        elif v_norm == "layer":
            norm = nn.LayerNorm(v_norm_shape)

        if dim == 0:
            self.conv1 = nn.Linear(dim_in, dim_out)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Linear(dim_out, dim_out)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    copy.deepcopy(norm),
                )
        if dim == 1:
            self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 2:
            self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 3:
            self.conv1 = nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def inv_sigmoid(x):
    return torch.log(x / (1 - x))


class FourierPE(nn.Module):
    def __init__(self, in_dim, num_freqs=6):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.out_dim = in_dim + 2 * in_dim * num_freqs

    def forward(self, x):
        freqs = 2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=x.dtype)
        xb = x[..., None, :] * freqs[:, None]
        xb = xb.flatten(-2)
        return torch.cat((x, torch.sin(torch.pi * xb), torch.cos(torch.pi * xb)), dim=-1)


def make_uv_grid(h, w, device, dtype=torch.float32):
    u = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    v = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack((uu, vv), dim=-1).reshape(h * w, 2)


def make_t_grid(n, device, dtype=torch.float32):
    return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)[:, None]


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.q_norm = nn.LayerNorm(dim)
        self.mem_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q, mem):
        mem = self.mem_norm(mem)
        out, _ = self.attn(self.q_norm(q), mem, mem, need_weights=False)
        q = q + out
        return q + self.ffn(q)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + out
        return x + self.ffn(x)


class SurfaceTokenEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_dim,
            num_tokens=4,
            hidden_dim=128,
            depth=4,
            heads=8,
            pe_freqs=6,
            dropout=0.0,
    ):
        super().__init__()
        if out_dim % num_tokens != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_tokens={num_tokens}")
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.token_dim = out_dim // num_tokens

        self.uv_pe = FourierPE(2, pe_freqs)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + self.uv_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cross_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.self_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_dim),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")
        pts = rearrange(x, "n c h w -> n (h w) c")
        uv = make_uv_grid(h, w, x.device, x.dtype)
        uv = self.uv_pe(uv)[None].expand(n, -1, -1)
        mem = self.input_proj(torch.cat((pts, uv), dim=-1))

        tokens = self.queries.expand(n, -1, -1)
        tokens = self.cross_attn(tokens, mem)
        for block in self.self_attn:
            tokens = block(tokens)
        tokens = self.out_proj(tokens)
        return tokens.reshape(n, self.out_dim)


class CurveTokenEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_dim,
            num_tokens=4,
            hidden_dim=128,
            depth=4,
            heads=8,
            pe_freqs=6,
            dropout=0.0,
    ):
        super().__init__()
        if out_dim % num_tokens != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_tokens={num_tokens}")
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.token_dim = out_dim // num_tokens

        self.t_pe = FourierPE(1, pe_freqs)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + self.t_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cross_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.self_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_dim),
        )

    def forward(self, x):
        n, c, l = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")
        pts = rearrange(x, "n c l -> n l c")
        t = make_t_grid(l, x.device, x.dtype)
        t = self.t_pe(t)[None].expand(n, -1, -1)
        mem = self.input_proj(torch.cat((pts, t), dim=-1))

        tokens = self.queries.expand(n, -1, -1)
        tokens = self.cross_attn(tokens, mem)
        for block in self.self_attn:
            tokens = block(tokens)
        tokens = self.out_proj(tokens)
        return tokens.reshape(n, self.out_dim)


class SurfaceQueryDecoder(nn.Module):
    def __init__(
            self,
            in_dim,
            out_channels,
            num_tokens=4,
            hidden_dim=128,
            depth=4,
            heads=8,
            pe_freqs=6,
            out_h=16,
            out_w=16,
            dropout=0.0,
    ):
        super().__init__()
        if in_dim % num_tokens != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by num_tokens={num_tokens}")
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.token_dim = in_dim // num_tokens
        self.out_h = out_h
        self.out_w = out_w

        self.uv_pe = FourierPE(2, pe_freqs)
        self.token_proj = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.query_proj = nn.Sequential(
            nn.Linear(self.uv_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z):
        n = z.shape[0]
        if n == 0:
            return z.new_empty((0, self.out_h, self.out_w, self.out_channels))
        tokens = z.reshape(n, self.num_tokens, self.token_dim)
        mem = self.token_proj(tokens)
        for block in self.latent_attn:
            mem = block(mem)

        uv = make_uv_grid(self.out_h, self.out_w, z.device, z.dtype)
        q = self.query_proj(self.uv_pe(uv))[None].expand(n, -1, -1)
        q = self.query_attn(q, mem)
        return self.out(q).reshape(n, self.out_h, self.out_w, -1)


class CurveQueryDecoder(nn.Module):
    def __init__(
            self,
            in_dim,
            out_channels,
            num_tokens=4,
            hidden_dim=128,
            depth=4,
            heads=8,
            pe_freqs=6,
            out_l=16,
            dropout=0.0,
    ):
        super().__init__()
        if in_dim % num_tokens != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by num_tokens={num_tokens}")
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.token_dim = in_dim // num_tokens
        self.out_l = out_l

        self.t_pe = FourierPE(1, pe_freqs)
        self.token_proj = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.query_proj = nn.Sequential(
            nn.Linear(self.t_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z):
        n = z.shape[0]
        if n == 0:
            return z.new_empty((0, self.out_l, self.out_channels))
        tokens = z.reshape(n, self.num_tokens, self.token_dim)
        mem = self.token_proj(tokens)
        for block in self.latent_attn:
            mem = block(mem)

        t = make_t_grid(self.out_l, z.device, z.dtype)
        q = self.query_proj(self.t_pe(t))[None].expand(n, -1, -1)
        q = self.query_attn(q, mem)
        return self.out(q).reshape(n, self.out_l, -1)


class AttnIntersection3(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_latent,
        num_layers,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.attn_proj_in = nn.Linear(dim_in, dim_latent)
        layer = nn.TransformerDecoderLayer(
            dim_latent, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.attn_proj_out = nn.Linear(dim_latent, dim_in * 2)

        self.pos_encoding = nn.Parameter(torch.randn(1, 2, dim_latent))


    def forward(
            self,
            src) -> Tensor:
        output = self.attn_proj_in(src) + self.pos_encoding
        tgt = output[:,0:1]
        mem = output[:,1:2]

        for mod in self.layers:
            tgt = mod(tgt, mem)

        output = self.attn_proj_out(tgt)[:,0]
        return output


# Revised atten and 1112 new
class AutoEncoder_1119(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = v_conf["in_channels"]
        self.with_intersection = v_conf["with_intersection"]

        self.face_coords = nn.Sequential(
            nn.Conv2d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,16)),
            nn.LeakyReLU(),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 8, 16, 16)),
            res_block_xd(2, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16, 16)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 8, 8)),
            res_block_xd(2, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8, 8)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 4, 4)),
            res_block_xd(2, ds // 2, ds // 1, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4, 4)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_xd(2, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            res_block_xd(2, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_coords = nn.Sequential(
            nn.Conv1d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,)),
            nn.LeakyReLU(),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 8, 16,)),
            res_block_xd(1, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 8,)),
            res_block_xd(1, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 4,)),
            res_block_xd(1, ds // 2, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(10):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768 # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 12, nn.LayerNorm(bd))

        self.global_feature1 = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(df * 2, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )

        self.inter = AttnIntersection3(df, 768, 12)
        self.classifier = nn.Linear(df*2, 1)

        self.face_attn_proj_in2 = nn.Linear(df, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, df)
        layer2 = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 12)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            res_block_xd(2, dl, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2, 2)),
            res_block_xd(2, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2, 2)),
            nn.ConvTranspose2d(ds // 1, ds // 2, kernel_size=2, stride=2),
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4, 4)),
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4, 4)),
            nn.ConvTranspose2d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8, 8)),
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8, 8)),
            nn.ConvTranspose2d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16, 16)),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16, 16)),
            nn.Conv2d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
        )
        self.face_center_scale_decoder = nn.Sequential(
            res_block_xd(0, dl * 2 * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            res_block_xd(1, df, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2,)),
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2,)),
            nn.ConvTranspose1d(ds, ds // 2, kernel_size=2, stride=2),
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4,)),
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4,)),
            nn.ConvTranspose1d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8,)),
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8,)),
            nn.ConvTranspose1d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16,)),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16,)),
            nn.Conv1d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=self.in_channels),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            res_block_xd(0, df * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.sigmoid = v_conf["sigmoid"]
        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.Linear(self.df*2, self.df*2),
            )
        else:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df),
                nn.LeakyReLU(),
                nn.Linear(self.df, self.df),
                nn.Identity() if not self.sigmoid else nn.Sigmoid(),
            )

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

        self.loss_fn = nn.L1Loss() if v_conf["loss"] == "l1" else nn.MSELoss()

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            return self.gaussian_proj(v_fused_face_features), torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        if v_is_test:
            fused_face_features = mean
        # return fused_face_features, kl_loss
        return fused_face_features, kl_loss, torch.cat([mean, std], dim=1)

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def encode(self, v_data, v_test):
        face_points = rearrange(v_data["face_points"][..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :self.in_channels], 'b h n -> b n h').contiguous()
        face_features = self.face_coords(face_points)
        edge_features = self.edge_coords(edge_points)

        # Face attn
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x

        # # Face graph
        edge_face_connectivity = v_data["edge_face_connectivity"]
        x = fused_face_features
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        fused_face_features = x + fused_face_features

        # Global
        bs = v_data["num_face_record"].shape[0]
        index = torch.arange(bs, device=fused_face_features.device).repeat_interleave(v_data["num_face_record"])
        face_z = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)
        face_z = torch.cat((fused_face_features, gf), dim=1)
        face_z = self.global_feature2(face_z) + fused_face_features

        return {
            "face_features": face_z,
            "edge_features": edge_features,
        }

    def decode(self, v_encoding_result, v_data=None, v_deduplicated=False):
        face_z = v_encoding_result["face_z"]
        face_feature = self.face_attn_proj_in2(face_z)
        if v_data is None:
            num_faces = face_z.shape[0]
            attn_mask = torch.zeros((num_faces, num_faces), dtype=bool, device=face_z.device)
        else:
            attn_mask = v_data["attn_mask"]
        face_feature = self.face_attn2(face_feature, attn_mask)
        face_z = self.face_attn_proj_out2(face_feature)

        decoding_results = {}
        decoding_results["face_points_local"] = self.face_points_decoder(face_z)
        decoding_results["face_center_scale"] = self.face_center_scale_decoder(face_z)

        if v_deduplicated: # Deduplicate
            face_points_local = decoding_results["face_points_local"]
            face_center_scale = decoding_results["face_center_scale"]
            pred_face_points = denormalize_coord1112(face_points_local, face_center_scale)
            num_faces = face_z.shape[0]

            deduplicate_face_id = []
            for i in range(num_faces):
                is_duplicate = False
                for j in deduplicate_face_id:
                    if torch.sqrt(((pred_face_points[i]-pred_face_points[j])**2).sum(dim=-1)).mean() < 1e-3:
                        is_duplicate=True
                        break
                if not is_duplicate:
                    deduplicate_face_id.append(i)
            face_z = face_z[deduplicate_face_id]

        if v_data is None:
            num_faces = face_z.shape[0]
            device = face_z.device
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = face_z[indexes]

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred_labels = torch.sigmoid(pred) > 0.5

            intersected_edge_feature = feature_pair[pred_labels]
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
            decoding_results["pred_edge_face_connectivity"] = torch.cat((torch.arange(intersected_edge_feature.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        else:
            edge_face_connectivity = v_data["edge_face_connectivity"]
            v_zero_positions = v_data["zero_positions"]

            true_intersection_embedding = face_z[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_z[v_zero_positions]
            id_false_start = true_intersection_embedding.shape[0]
            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = feature_pair[:id_false_start]

            decoding_results["loss_edge_feature"] = self.loss_fn(
                intersected_edge_feature,
                v_encoding_result["edge_features"][edge_face_connectivity[:, 0]].detach(),
            )

            decoding_results["loss_edge"] = loss_edge

        decoding_results["edge_points_local"] = self.edge_points_decoder(intersected_edge_feature)
        decoding_results["edge_center_scale"] = self.edge_center_scale_decoder(intersected_edge_feature)
        if "edge_features" in v_encoding_result:
            decoding_results["edge_points_local1"] = self.edge_points_decoder(v_encoding_result["edge_features"])
            decoding_results["edge_center_scale1"] = self.edge_center_scale_decoder(v_encoding_result["edge_features"])

        decoding_results["face_features"] = v_encoding_result["face_z"]
        return decoding_results

    def loss(self, v_decoding_result, v_data):
        # Loss
        loss={}
        loss["face_norm"] = self.loss_fn(
            v_decoding_result["face_points_local"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = self.loss_fn(
            v_decoding_result["face_center_scale"],
            v_data["face_bbox"]
        )

        loss["edge_norm1"] = self.loss_fn(
            v_decoding_result["edge_points_local1"],
            v_data["edge_norm"]
        )
        loss["edge_bbox1"] = self.loss_fn(
            v_decoding_result["edge_center_scale1"],
            v_data["edge_bbox"]
        )
        loss["edge_feature"] = v_decoding_result["loss_edge_feature"]
        loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.1
        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss["edge_norm"] = self.loss_fn(
            v_decoding_result["edge_points_local"],
            v_data["edge_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_bbox"] = self.loss_fn(
            v_decoding_result["edge_center_scale"],
            v_data["edge_bbox"][edge_face_connectivity[:, 0]]
        )
        if self.gaussian_weights > 0:
            loss["kl_loss"] = v_decoding_result["kl_loss"]
        return loss

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        face_z, kl_loss, features = self.sample(encoding_result["face_features"], v_is_test=v_test)
        encoding_result["face_z"] = face_z
        decoding_result = self.decode(encoding_result, v_data)
        decoding_result["kl_loss"] = kl_loss
        loss = self.loss(decoding_result, v_data)
        loss["total_loss"] = sum(loss.values())
        data = {}
        if v_test:
            pred_data = self.decode(encoding_result)

            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            data["face_features"] = features.cpu().numpy()
            data["gt_face_adj"] = face_adj.reshape(-1)
            data["pred_face_adj"] = pred_data["pred_face_adj"].reshape(-1)
            data["gt_edge"] = v_data["edge_points"].detach().cpu().numpy()
            data["gt_edge_face_connectivity"] = v_data["edge_face_connectivity"].detach().cpu().numpy()
            pred_edge = denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])
            data["pred_edge"] = pred_edge.detach().cpu().numpy()
            data["pred_edge_face_connectivity"] = pred_data["pred_edge_face_connectivity"].detach().cpu().numpy()
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local"], decoding_result["edge_center_scale"])[..., :3],
                v_data["edge_points"][v_data["edge_face_connectivity"][:, 0]][..., :3]
            )

            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local1"], decoding_result["edge_center_scale1"])[..., :3],
                v_data["edge_points"][..., :3]
            )
            data["gt_face"] = v_data["face_points"].detach().cpu().numpy()
            pred_face = denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])
            data["pred_face"] = pred_face.detach().cpu().numpy()
            loss["face_coords"] = nn.functional.l1_loss(
                pred_face[..., :3],
                v_data["face_points"][..., :3]
            )

        return loss, data

    def inference(self, v_face_features):
        pred_data = self.decode({"face_z": v_face_features}, v_deduplicated=True)
        return {
            "face_features": v_face_features.to(torch.float32).cpu().numpy(),
            "pred_face_adj": pred_data["pred_face_adj"],
            "pred_face_adj_prob": pred_data["pred_edge_face_connectivity"].reshape(-1).to(torch.float32).cpu().numpy(),
            "pred_edge_face_connectivity": pred_data["pred_edge_face_connectivity"].to(torch.float32).cpu().numpy(),
            "pred_face": denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
            "pred_edge": denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
        }


class AutoEncoder_1119_TokenVAE(AutoEncoder_1119):
    """
    Tokenized VAE variant for comparison experiments.

    AutoEncoder_1119 is kept as the original CNN-grid baseline. This class
    only swaps primitive encoders and point decoders to latent-token modules;
    graph fusion, face-pair intersection, bbox decoders, losses, and inference
    stay inherited from AutoEncoder_1119.
    """

    def __init__(self, v_conf):
        super().__init__(v_conf)
        df = self.df

        tok_k = int(v_conf.get("primitive_token_num", 4))
        edge_tok_k = int(v_conf.get("edge_token_num", tok_k))
        tok_hidden = int(v_conf.get("primitive_token_hidden", 128))
        tok_depth = int(v_conf.get("primitive_token_depth", 4))
        tok_heads = int(v_conf.get("primitive_token_heads", 8))
        tok_freqs = int(v_conf.get("primitive_pe_freqs", 6))
        tok_dropout = float(v_conf.get("primitive_token_dropout", 0.0))

        self.face_coords = SurfaceTokenEncoder(
            in_channels=self.in_channels,
            out_dim=df,
            num_tokens=tok_k,
            hidden_dim=tok_hidden,
            depth=tok_depth,
            heads=tok_heads,
            pe_freqs=tok_freqs,
            dropout=tok_dropout,
        )
        self.edge_coords = CurveTokenEncoder(
            in_channels=self.in_channels,
            out_dim=df * 2,
            num_tokens=edge_tok_k,
            hidden_dim=tok_hidden,
            depth=tok_depth,
            heads=tok_heads,
            pe_freqs=tok_freqs,
            dropout=tok_dropout,
        )
        self.face_points_decoder = SurfaceQueryDecoder(
            in_dim=df,
            out_channels=self.in_channels,
            num_tokens=tok_k,
            hidden_dim=tok_hidden,
            depth=tok_depth,
            heads=tok_heads,
            pe_freqs=tok_freqs,
            out_h=16,
            out_w=16,
            dropout=tok_dropout,
        )
        self.edge_points_decoder = CurveQueryDecoder(
            in_dim=df * 2,
            out_channels=self.in_channels,
            num_tokens=edge_tok_k,
            hidden_dim=tok_hidden,
            depth=tok_depth,
            heads=tok_heads,
            pe_freqs=tok_freqs,
            out_l=16,
            dropout=tok_dropout,
        )

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            fused_face_features = self.gaussian_proj(v_fused_face_features)
            return fused_face_features, torch.zeros_like(v_fused_face_features[0, 0]), fused_face_features

        return super().sample(v_fused_face_features, v_is_test=v_is_test)


class AutoEncoder_1119_TokenEncConvDecVAE(AutoEncoder_1119):
    """
    Ablation variant: token encoders with the original convolutional decoders.

    This isolates whether the reconstruction gap comes from tokenizing the
    primitives or from the coordinate-query point decoders.
    """

    def __init__(self, v_conf):
        super().__init__(v_conf)
        df = self.df

        tok_k = int(v_conf.get("primitive_token_num", 4))
        edge_tok_k = int(v_conf.get("edge_token_num", tok_k))
        tok_hidden = int(v_conf.get("primitive_token_hidden", 128))
        tok_depth = int(v_conf.get("primitive_token_depth", 4))
        tok_heads = int(v_conf.get("primitive_token_heads", 8))
        tok_freqs = int(v_conf.get("primitive_pe_freqs", 6))
        tok_dropout = float(v_conf.get("primitive_token_dropout", 0.0))

        self.face_coords = SurfaceTokenEncoder(
            in_channels=self.in_channels,
            out_dim=df,
            num_tokens=tok_k,
            hidden_dim=tok_hidden,
            depth=tok_depth,
            heads=tok_heads,
            pe_freqs=tok_freqs,
            dropout=tok_dropout,
        )
        self.edge_coords = CurveTokenEncoder(
            in_channels=self.in_channels,
            out_dim=df * 2,
            num_tokens=edge_tok_k,
            hidden_dim=tok_hidden,
            depth=tok_depth,
            heads=tok_heads,
            pe_freqs=tok_freqs,
            dropout=tok_dropout,
        )

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            fused_face_features = self.gaussian_proj(v_fused_face_features)
            return fused_face_features, torch.zeros_like(v_fused_face_features[0, 0]), fused_face_features

        return super().sample(v_fused_face_features, v_is_test=v_is_test)


# Revised atten and 1112 new
class AutoEncoder_1119_light(AutoEncoder_1119):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = v_conf["in_channels"]
        self.with_intersection = v_conf["with_intersection"]

        self.face_coords = nn.Sequential(
            nn.Conv2d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,16)),
            nn.LeakyReLU(),
            res_block_xd(2, ds // 8, ds // 4, 3, 1, 1, v_norm=norm , v_norm_shape = (ds // 4, 16, 16)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_xd(2, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8, 8)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_xd(2, ds // 2, ds // 1, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4, 4)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_xd(2, ds // 1, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_coords = nn.Sequential(
            nn.Conv1d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,)),
            nn.LeakyReLU(),
            res_block_xd(1, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_xd(1, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_xd(1, ds // 2, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768 # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))

        self.global_feature1 = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(df * 2, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )

        self.inter = AttnIntersection3(df, 512, 8)
        self.classifier = nn.Linear(df*2, 1)

        self.face_attn_proj_in2 = nn.Linear(df, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, df)
        layer2 = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 8)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            res_block_xd(2, dl, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2, 2)),
            nn.ConvTranspose2d(ds // 1, ds // 2, kernel_size=2, stride=2),
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4, 4)),
            nn.ConvTranspose2d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8, 8)),
            nn.ConvTranspose2d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16, 16)),
            nn.Conv2d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
        )
        self.face_center_scale_decoder = nn.Sequential(
            res_block_xd(0, dl * 2 * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            res_block_xd(1, df, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2,)),
            nn.ConvTranspose1d(ds, ds // 2, kernel_size=2, stride=2),
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4,)),
            nn.ConvTranspose1d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8,)),
            nn.ConvTranspose1d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16,)),
            nn.Conv1d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=self.in_channels),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            res_block_xd(0, df * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.sigmoid = v_conf["sigmoid"]
        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.Linear(self.df*2, self.df*2),
            )
        else:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df),
                nn.LeakyReLU(),
                nn.Linear(self.df, self.df),
                nn.Identity() if not self.sigmoid else nn.Sigmoid(),
            )

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

        self.loss_fn = nn.L1Loss() if v_conf["loss"] == "l1" else nn.MSELoss()


class AutoEncoder_1225(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = v_conf["in_channels"]
        self.with_intersection = v_conf["with_intersection"]

        self.face_coords = nn.Sequential(
            nn.Conv2d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,16)),
            nn.LeakyReLU(),
            res_block_xd(2, ds // 8, ds // 4, 3, 1, 1, v_norm=norm , v_norm_shape = (ds // 4, 16, 16)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_xd(2, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8, 8)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_xd(2, ds // 2, ds // 1, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4, 4)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_xd(2, ds // 1, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_coords = nn.Sequential(
            nn.Conv1d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,)),
            nn.LeakyReLU(),
            res_block_xd(1, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_xd(1, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_xd(1, ds // 2, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768 # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))

        self.global_feature1 = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(df * 2, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )

        self.inter = AttnIntersection3(df, 512, 8)
        self.classifier = nn.Linear(df*2, 1)

        self.face_attn_proj_in2 = nn.Linear(df, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, df)
        layer2 = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 8)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            res_block_xd(2, dl, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2, 2)),
            nn.ConvTranspose2d(ds // 1, ds // 2, kernel_size=2, stride=2),
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4, 4)),
            nn.ConvTranspose2d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8, 8)),
            nn.ConvTranspose2d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16, 16)),
            nn.Conv2d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
        )
        self.face_center_scale_decoder = nn.Sequential(
            res_block_xd(0, dl * 2 * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            res_block_xd(1, df, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2,)),
            nn.ConvTranspose1d(ds, ds // 2, kernel_size=2, stride=2),
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4,)),
            nn.ConvTranspose1d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8,)),
            nn.ConvTranspose1d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16,)),
            nn.Conv1d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=self.in_channels),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            res_block_xd(0, df * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.sigmoid = v_conf["sigmoid"]
        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.Linear(self.df*2, self.df*2),
            )
        else:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df),
                nn.LeakyReLU(),
                nn.Linear(self.df, self.df),
                nn.Identity() if not self.sigmoid else nn.Sigmoid(),
            )

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

        self.loss_fn = nn.L1Loss() if v_conf["loss"] == "l1" else nn.MSELoss()

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            return self.gaussian_proj(v_fused_face_features), torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        if v_is_test:
            fused_face_features = mean
        # return fused_face_features, kl_loss
        return fused_face_features, kl_loss, torch.cat([mean, std], dim=1)

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def encode(self, v_data, v_test):
        face_points = rearrange(v_data["face_points"][..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :self.in_channels], 'b h n -> b n h').contiguous()
        face_features = self.face_coords(face_points)
        edge_features = self.edge_coords(edge_points)

        # Face attn
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x

        # # Face graph
        edge_face_connectivity = v_data["edge_face_connectivity"]
        x = fused_face_features
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        fused_face_features = x + fused_face_features

        # Global
        bs = v_data["num_face_record"].shape[0]
        index = torch.arange(bs, device=fused_face_features.device).repeat_interleave(v_data["num_face_record"])
        face_z = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)
        face_z = torch.cat((fused_face_features, gf), dim=1)
        face_z = self.global_feature2(face_z) + fused_face_features

        return {
            "face_features": face_z,
            "edge_features": edge_features,
        }

    def decode(self, v_encoding_result, v_data=None, v_deduplicated=False):
        face_z = v_encoding_result["face_z"]
        face_feature = self.face_attn_proj_in2(face_z)
        if v_data is None:
            num_faces = face_z.shape[0]
            attn_mask = torch.zeros((num_faces, num_faces), dtype=bool, device=face_z.device)
        else:
            attn_mask = v_data["attn_mask"]
        face_feature = self.face_attn2(face_feature, attn_mask)
        face_z = self.face_attn_proj_out2(face_feature)

        decoding_results = {}
        decoding_results["face_points_local"] = self.face_points_decoder(face_z)
        decoding_results["face_center_scale"] = self.face_center_scale_decoder(face_z)

        if v_deduplicated: # Deduplicate
            face_points_local = decoding_results["face_points_local"]
            face_center_scale = decoding_results["face_center_scale"]
            pred_face_points = denormalize_coord1112(face_points_local, face_center_scale)
            num_faces = face_z.shape[0]

            deduplicate_face_id = []
            for i in range(num_faces):
                is_duplicate = False
                for j in deduplicate_face_id:
                    if torch.sqrt(((pred_face_points[i]-pred_face_points[j])**2).sum(dim=-1)).mean() < 1e-3:
                        is_duplicate=True
                        break
                if not is_duplicate:
                    deduplicate_face_id.append(i)
            face_z = face_z[deduplicate_face_id]

        if v_data is None:
            num_faces = face_z.shape[0]
            device = face_z.device
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = face_z[indexes]

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred_labels = torch.sigmoid(pred) > 0.5

            intersected_edge_feature = feature_pair[pred_labels]
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
            decoding_results["pred_edge_face_connectivity"] = torch.cat((torch.arange(intersected_edge_feature.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        else:
            edge_face_connectivity = v_data["edge_face_connectivity"]
            v_zero_positions = v_data["zero_positions"]

            true_intersection_embedding = face_z[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_z[v_zero_positions]
            id_false_start = true_intersection_embedding.shape[0]
            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = feature_pair[:id_false_start]

            decoding_results["loss_edge_feature"] = self.loss_fn(
                intersected_edge_feature,
                v_encoding_result["edge_features"][edge_face_connectivity[:, 0]].detach(),
            )

            decoding_results["loss_edge"] = loss_edge

        decoding_results["edge_points_local"] = self.edge_points_decoder(intersected_edge_feature)
        decoding_results["edge_center_scale"] = self.edge_center_scale_decoder(intersected_edge_feature)
        if "edge_features" in v_encoding_result:
            decoding_results["edge_points_local1"] = self.edge_points_decoder(v_encoding_result["edge_features"])
            decoding_results["edge_center_scale1"] = self.edge_center_scale_decoder(v_encoding_result["edge_features"])

        decoding_results["face_features"] = v_encoding_result["face_z"]
        return decoding_results

    def loss(self, v_decoding_result, v_data):
        # Loss
        loss={}
        loss["face_norm"] = self.loss_fn(
            v_decoding_result["face_points_local"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = self.loss_fn(
            v_decoding_result["face_center_scale"],
            v_data["face_bbox"]
        )

        loss["edge_norm1"] = self.loss_fn(
            v_decoding_result["edge_points_local1"],
            v_data["edge_norm"]
        )
        loss["edge_bbox1"] = self.loss_fn(
            v_decoding_result["edge_center_scale1"],
            v_data["edge_bbox"]
        )
        loss["edge_feature"] = v_decoding_result["loss_edge_feature"]
        loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.1
        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss["edge_norm"] = self.loss_fn(
            v_decoding_result["edge_points_local"],
            v_data["edge_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_bbox"] = self.loss_fn(
            v_decoding_result["edge_center_scale"],
            v_data["edge_bbox"][edge_face_connectivity[:, 0]]
        )
        if self.gaussian_weights > 0:
            loss["kl_loss"] = v_decoding_result["kl_loss"]
        return loss

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        face_z, kl_loss, features = self.sample(encoding_result["face_features"], v_is_test=v_test)
        encoding_result["face_z"] = face_z
        decoding_result = self.decode(encoding_result, v_data)
        decoding_result["kl_loss"] = kl_loss
        loss = self.loss(decoding_result, v_data)
        loss["total_loss"] = sum(loss.values())
        data = {}
        if v_test:
            pred_data = self.decode(encoding_result)

            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            data["face_features"] = features.cpu().numpy()
            data["gt_face_adj"] = face_adj.reshape(-1)
            data["pred_face_adj"] = pred_data["pred_face_adj"].reshape(-1)
            data["gt_edge"] = v_data["edge_points"].detach().cpu().numpy()
            data["gt_edge_face_connectivity"] = v_data["edge_face_connectivity"].detach().cpu().numpy()
            pred_edge = denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])
            data["pred_edge"] = pred_edge.detach().cpu().numpy()
            data["pred_edge_face_connectivity"] = pred_data["pred_edge_face_connectivity"].detach().cpu().numpy()
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local"], decoding_result["edge_center_scale"])[..., :3],
                v_data["edge_points"][v_data["edge_face_connectivity"][:, 0]][..., :3]
            )

            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local1"], decoding_result["edge_center_scale1"])[..., :3],
                v_data["edge_points"][..., :3]
            )
            data["gt_face"] = v_data["face_points"].detach().cpu().numpy()
            pred_face = denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])
            data["pred_face"] = pred_face.detach().cpu().numpy()
            loss["face_coords"] = nn.functional.l1_loss(
                pred_face[..., :3],
                v_data["face_points"][..., :3]
            )

        return loss, data

    def inference(self, v_face_features):
        pred_data = self.decode({"face_z": v_face_features}, v_deduplicated=True)
        return {
            "face_features": v_face_features.to(torch.float32).cpu().numpy(),
            "pred_face_adj": pred_data["pred_face_adj"],
            "pred_face_adj_prob": pred_data["pred_edge_face_connectivity"].reshape(-1).to(torch.float32).cpu().numpy(),
            "pred_edge_face_connectivity": pred_data["pred_edge_face_connectivity"].to(torch.float32).cpu().numpy(),
            "pred_face": denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
            "pred_edge": denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
        }
