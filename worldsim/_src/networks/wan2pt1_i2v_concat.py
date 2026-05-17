from worldsim._src.networks.wan2pt1 import WanModel, Optional, rearrange, transforms, VideoSize, sinusoidal_embedding_1d
import torch
import torch.nn as nn
import torch.amp as amp
from torch.distributed._composable.fsdp import fully_shard


class Wan2pt1I2VConcat(WanModel):
    def __init__(self, *args,
                 additional_concat_ch: int=0,
                 additional_init_method: str='random_init',
                 additional_embed_alpha: float=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_concat_ch = additional_concat_ch
        self.additional_init_method = additional_init_method
        self.additional_embed_alpha = additional_embed_alpha
        assert self.conv_patchify
        if self.additional_concat_ch != 0:
            self.addition_patch_embedding = nn.Conv3d(self.additional_concat_ch, self.dim, kernel_size=self.patch_size, stride=self.patch_size)


    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'addition_patch_embedding'):
            if self.additional_init_method == 'random_init':
                print('==> random init weight')
                nn.init.xavier_uniform_(self.addition_patch_embedding.weight.flatten(1))
                nn.init.zeros_(self.addition_patch_embedding.bias)
            else:
                raise NotImplementedError

    def fully_shard(self, mesh, **fsdp_kwargs):
        super().fully_shard(mesh, **fsdp_kwargs)
        if hasattr(self, 'addition_patch_embedding'):
            fully_shard(self.addition_patch_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)

    def forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        seq_len=None,
        frame_cond_crossattn_emb_B_L_D=None,
        y_B_C_T_H_W=None,
        padding_mask: Optional[torch.Tensor] = None,
        is_uncond=False,
        slg_layers=None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x_B_C_T_H_W (Tensor):
                Input video tensor with shape [B, C_in, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            frame_cond_crossattn_emb_B_L_D (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y_B_C_T_H_W (Tensor, *optional*):
                Conditional video inputs for image-to-video mode, shape [B, C_in, T, H, W]

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, T, H / 8, W / 8]
        """
        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]
        del kwargs
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert frame_cond_crossattn_emb_B_L_D is not None and y_B_C_T_H_W is not None

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        if self.additional_concat_ch != 0:
            ori_x_B_C_T_H_W = x_B_C_T_H_W
            x_B_C_T_H_W = ori_x_B_C_T_H_W[:, :-self.additional_concat_ch]
            addition_x_B_C_T_H_W = ori_x_B_C_T_H_W[:, -self.additional_concat_ch:]


        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        # prepare the patch embedding
        x_B_D_T_H_W = self.patch_embedding(x_B_C_T_H_W)
        x_B_T_H_W_D = rearrange(x_B_D_T_H_W, "b d t h w -> b t h w d")
        if self.additional_concat_ch != 0:
            additional_x_B_D_T_H_W = self.addition_patch_embedding(addition_x_B_C_T_H_W)
            additional_x_B_D_T_H_W = rearrange(additional_x_B_D_T_H_W, "b d t h w -> b t h w d")
            x_B_T_H_W_D = x_B_T_H_W_D + self.additional_embed_alpha * additional_x_B_D_T_H_W

        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)
        seq_len = seq_lens.max().item()
        assert seq_lens.max() == seq_len

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
            e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))
            assert e_B_D.dtype == torch.float32 and e0_B_6_D.dtype == torch.float32

        # context
        context_lens = None
        context_B_L_D = self.text_embedding(crossattn_emb)

        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)  # bs x 257 (x2) x dim
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        # arguments
        kwargs = dict(
            e=e0_B_6_D,
            seq_lens=seq_lens,
            video_size=video_size,
            freqs=self.rope_position_embedding(x_B_T_H_W_D),
            context=context_B_L_D,
            context_lens=context_lens,
        )

        for block_idx, block in enumerate(self.blocks):
            if slg_layers is not None and block_idx in slg_layers and is_uncond:
                continue
            x_B_L_D = block(x_B_L_D, **kwargs)

        # head
        x_B_L_D = self.head(x_B_L_D, e_B_D)

        # unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_L_D,
            "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            t=t,
            h=h,
            w=w,
            d=self.out_dim,
        )

        return x_B_C_T_H_W

