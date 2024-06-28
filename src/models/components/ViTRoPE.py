"""
From  https://github.com/pranavphoenix/VisionXformer/
"""

from tkinter import NO
from turtle import pos

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from matplotlib.pylab import f
from numpy import isin, mask_indices
from torch import einsum, nn

from models.components.utils import (
    SimpleSegmentationHead,
    infer_output,
    load_state_dict,
)


def exists(val):
    return val is not None


def compress_mask(tensor, mask):
    """Return the value of tensor where mask is true.

    Args:
        tensor (torch.Tensor): tensor to compress (N, L, C)
        mask (torch.Tensor): boolean mask (N, L)

    Returns:
        torch.Tensor: the value of tensor where mask is true
    """
    # print(f"before compress {tensor.shape=}")
    tensor = tensor[mask].view(tensor.shape[0], -1, tensor.shape[2])
    # print(f"after compress {tensor.shape=}")
    return tensor


def uncompress_mask(tensor, mask, filler=None):
    """The value of tensor are placed at the true position of mask the rest is 0.

    Args:
        tensor (torch.Tensor): tensor to uncompress (N, l, C)
        mask (torch.Tensor): boolean mask (N, L)
        filler (torch.Tensor, optional): value to fill the tensor where mask is false (C). Defaults to None.

    Returns:
        torch.Tensor: the value of tensor where mask is true (N, L, C)
    """
    if isinstance(mask, bool) and mask:
        return tensor
    # print(f"before uncompress {tensor.shape=}")
    if filler is None:
        out = torch.zeros(
            (tensor.shape[0], mask.shape[1], tensor.shape[2]),
            device=tensor.device,
        )
        out[mask] = tensor.view(-1, tensor.shape[2])
    else:
        filler = filler.to(tensor.device)
        full = torch.zeros(
            (tensor.shape[0], mask.shape[1], tensor.shape[2]),
            device=tensor.device,
        )
        full[mask] = tensor.view(-1, tensor.shape[2])
        # print(f"{mask.shape=}") # mask.shape=torch.Size([16, 1024])
        # print(f"{full.shape=}") # full.shape=torch.Size([16, 1024, 384])
        # print(f"{filler.shape=}") # filler.shape=torch.Size([384])
        out = torch.where(
            mask.view(mask.shape[0], mask.shape[1], 1),
            full,
            filler.view(1, 1, -1),
        )
    # print(f"after uncompress {out.shape=}")
    return out


def rotate_every_two(x):
    """Rotate every other element of x.

    Args:
        x (torch.Tensor): tensor to rotate (..., 2d)

    Returns:
        torch.Tensor: rotated tensor (..., 2d)
    """
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(q, k, sinu_pos):
    """Apply.

    Args:
        q (torch.Tensor): query tensor (B, H, L, c)
        k (torch.Tensor): key tensor (B, H, L, c)
        sinu_pos (torch.Tensor): sinusoidal position embedding (B, L, 2c)

    Returns:
        tuple: tuple of query and key with rotary position embedding applied
    """
    sinu_pos = rearrange(sinu_pos, "b p (j d) -> b p j d", j=2)
    sin, cos = sinu_pos.unbind(dim=-2)

    sin, cos = map(
        lambda t: repeat(t, "b p n -> b h p (n j)", j=2, h=1), (sin, cos)
    )
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


class FixedPositionalEmbedding(nn.Module):
    """Fixed Positional Embedding module.

    Args:
        dim (int): The dimension of the embedding.
        max_seq_len (int): The maximum sequence length.

    Attributes:
        emb (torch.Tensor): The fixed positional embedding tensor.
    """

    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x):
        """Forward pass of the FixedPositionalEmbedding module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The positional embedding tensor.
        """
        return self.emb[None, : x.shape[1], :].to(x)


# helpers


def pair(t):
    """Returns a tuple with two elements.

    If the input is already a tuple, it is returned as is.
    If the input is not a tuple, it is duplicated and returned as a tuple.

    Args:
        t: The input value.

    Returns:
        A tuple with two elements.
    """
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    """Initializes a PreNorm module for applying normalization before any function.

    Args:
        dim (int): The input dimension.
        fn (nn.Module): The function to be applied after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, context=None, **kwargs):
        """Forward pass of the PreNorm module.

        Args:
            x (torch.Tensor or tuple): The input tensor or tuple containing the input tensor and a mask.

        Returns:
            torch.Tensor: The output tensor after applying normalization and the specified function.
        """
        is_tuple = isinstance(x, tuple)
        if context is not None:
            context = self.norm(context)  # seems necessary
        if is_tuple:
            x, mask = x
            return self.fn((self.norm(x), mask), context=context, **kwargs)
        return self.fn(self.norm(x), context=context, **kwargs)


class FeedForward(nn.Module):
    """FeedForward module used in the ViTRoPE model.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        conditionnal_posemb (bool, optional): Whether to use conditional positional embeddings. Defaults to False.
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
        conditionnal_posemb=False,
        cond_kernel_size=3,
        learned_masked_emb=False,
    ):
        super().__init__()
        self.conditionnal_posemb = conditionnal_posemb
        self.learned_masked_emb = learned_masked_emb
        if not self.conditionnal_posemb:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, cond_kernel_size, padding="same"),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(hidden_dim, dim, cond_kernel_size, padding="same"),
                nn.Dropout(dropout),
            )
            if self.learned_masked_emb:
                self.masked_emb = nn.Parameter(torch.zeros(dim))
            else:
                self.masked_emb = None

    def forward(self, x, context=None):
        """Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor or tuple): The input tensor or tuple containing the input tensor and a mask.


        Returns:
            torch.Tensor: Output tensor.
        """
        is_tuple = isinstance(x, tuple)
        if is_tuple:
            x, mask = x
        if not self.conditionnal_posemb:
            return self.net(x)
        else:
            if is_tuple:
                x = uncompress_mask(x, mask, filler=self.masked_emb)

            side = int(x.shape[1] ** 0.5)
            # convert from NLC to NCHW
            x = x.permute(0, 2, 1)
            x = x.reshape(x.shape[0], x.shape[1], side, side)
            # apply conv
            x = self.net(x)
            # convert from NCHW to NLC
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            x = x.permute(0, 2, 1)
            if is_tuple:
                x = compress_mask(x, mask)
            return x


class Attention(nn.Module):
    """Attention module used in the ViTRoPE model.

    Args:
        dim (int): Dimension of the input and output tensors.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, pos_emb=None, context=None):
        """Forward pass of the ViTRoPE model.

        Args:
            x (torch.Tensor or Tuple[torch.Tensor, torch.Tensor]): Input tensor or tuple of input tensor and mask.
            pos_emb (torch.Tensor, optional): Positional embeddings tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """
        is_tuple = isinstance(x, tuple)
        if is_tuple:
            x, mask = x
        b, n, _, h = *x.shape, self.heads

        if context is not None:
            n_context = context.shape[1]
            x = torch.cat([x, context], dim=1)
        else:
            n_context = 0

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        if exists(pos_emb):
            if n_context == 1:
                # context is placed at the center of the image
                pos_emb_context = pos_emb[:, len(pos_emb) // 2, :].unsqueeze(1)
                Q_context, K_context = apply_rotary_pos_emb(
                    q[:, :, n:, :], k[:, :, n:, :], pos_emb_context
                )
            elif n_context == 4:
                # context is placed at the corners of the image
                img_side = int(n**0.5)
                pos_emb_context = torch.stack(
                    [
                        pos_emb[:, 0, :],
                        pos_emb[:, img_side - 1, :],
                        pos_emb[:, n - img_side, :],
                        pos_emb[:, n - 1, :],
                    ],
                    dim=1,
                )
                Q_context, K_context = apply_rotary_pos_emb(
                    q[:, :, n:, :], k[:, :, n:, :], pos_emb_context
                )
            else:
                Q_context, K_context = q[:, :, n:, :], k[:, :, n:, :]
            if is_tuple:
                pos_emb = pos_emb.expand(b, -1, -1)[mask].view(
                    b, -1, pos_emb.shape[-1]
                )
            Q, K = apply_rotary_pos_emb(
                q[:, :, :n, :], k[:, :, :n, :], pos_emb
            )
            q, k = torch.cat((Q, Q_context), dim=2), torch.cat(
                (K, K_context), dim=2
            )

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out[:, :n, :], out[:, n:, :] if context is not None else None


class Transformer(nn.Module):
    """Transformer module.

    Args:
        dim (int): Dimension of the input and output tensors.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the feed-forward layer.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        conditionnal_posemb (bool, optional): Whether to conditionally add positional embeddings. Defaults to False.
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        conditionnal_posemb=False,
        cond_kernel_size=3,
        learned_masked_emb=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                                conditionnal_posemb=conditionnal_posemb,
                                cond_kernel_size=cond_kernel_size,
                                learned_masked_emb=learned_masked_emb,
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x, pos_emb, context=None):
        """Forward pass of the transformer.

        Args:
            x (torch.Tensor or tuple): Input tensor or tuple of input tensor and mask.
            pos_emb (torch.Tensor): Positional embeddings.

        Returns:
            torch.Tensor: Output tensor.
        """
        is_tuple = isinstance(x, tuple)
        if is_tuple:
            x, mask = x
        for attn, ff in self.layers:
            if is_tuple:
                attention = attn((x, mask), pos_emb=pos_emb, context=context)
                x, context = attention[0] + x, (
                    attention[1] + context if context is not None else None
                )
                x = ff((x, mask)) + x
            else:
                attention = attn(x, pos_emb=pos_emb, context=context)
                x, context = attention[0] + x, (
                    attention[1] + context if context is not None else None
                )
                x = ff(x) + x
            if context is not None:
                context = ff(context) + context
        return x


class ViTRoPE(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        rotary_position_emb=True,
        conditionnal_posemb=False,
        cond_kernel_size=3,
        learned_masked_emb=False,
        patch_embedder=None,
    ):
        super().__init__()
        self.rotary_position_emb = rotary_position_emb
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (
            image_width // patch_width
        )
        patch_dim = channels * patch_height * patch_width

        if patch_embedder is None:
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                nn.Linear(patch_dim, dim),
            )
        else:
            self.to_patch_embedding = patch_embedder

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            conditionnal_posemb=conditionnal_posemb,
            cond_kernel_size=cond_kernel_size,
            learned_masked_emb=learned_masked_emb,
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        )
        max_seq_len = num_patches
        if rotary_position_emb:
            self.layer_pos_emb = FixedPositionalEmbedding(
                dim_head, max_seq_len
            )

    def forward_features(self, img, embedded=False, context=None):
        is_tuple = isinstance(img, tuple)
        if is_tuple:
            img, mask = img
        else:
            img, mask = img, True

        if not embedded:
            x = self.to_patch_embedding(img)
        else:
            x = img

        b, n, _ = x.shape

        if not self.rotary_position_emb:
            x += self.pos_embedding.repeat(b, 1, 1)[mask].view(b, n, -1)
        x = self.dropout(x)
        # if context is not None:
        #     context = self.dropout(context) #TODO: check if this is necessary

        if self.rotary_position_emb:
            if is_tuple:
                layer_pos_emb = self.layer_pos_emb(uncompress_mask(x, mask))
            else:
                layer_pos_emb = self.layer_pos_emb(x)
        else:
            layer_pos_emb = None
        if is_tuple:
            x = self.transformer(
                (x, mask), pos_emb=layer_pos_emb, context=context
            )
        else:
            x = self.transformer(x, pos_emb=layer_pos_emb, context=context)
        return x

    def forward(self, img):
        x = self.forward_features(img)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViTRoPE_net(nn.Module):
    """ViTRoPE_net is a neural network model based on Vision Transformer (ViT) for image
    segmentation.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 4.
        num_channels (int, optional): Number of input channels. Defaults to 1.
        segmentation_head (nn.Module, optional): Segmentation head module (not initialized). Defaults to SimpleSegmentationHead.
        pretrained (bool, optional): Whether to use a pretrained ViT model. Defaults to True.
        pretrained_path (str, optional): Path to the pretrained model checkpoint. Defaults to None.
        img_size (int, optional): Size of the input image. Defaults to 512.
        patch_size (int, optional): Patch size for ViT. Defaults to 16.
        embed_dim (int, optional): Embedding dimension for ViT. Defaults to 384.
        depth (int, optional): Depth of the ViT model. Defaults to 12.
        num_heads (int, optional): Number of attention heads in ViT. Defaults to 6.
        mlp_ratio (int, optional): Ratio of MLP hidden size to embedding dimension in ViT. Defaults to 4.
        rotary_position_emb (bool, optional): Whether to use rotary position embeddings in ViT. Defaults to True.
        conditionnal_posemb (bool, optional): Whether to use conditional position embeddings in ViT. Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_classes=4,
        num_channels=1,
        segmentation_head=SimpleSegmentationHead,
        pretrained=True,
        pretrained_path=None,
        img_size=512,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        rotary_position_emb=True,
        conditionnal_posemb=False,
        cond_kernel_size=3,
        learned_masked_emb=False,
        context_token=0,
        context_network_shape=[1, 1, 1, 1],
        num_channels_token=None,
        patch_embedder=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.img_size = img_size
        self.num_channels_token = (
            num_channels_token
            if num_channels_token is not None
            else num_channels
        )

        if patch_embedder is not None:
            patch_embedder = patch_embedder(
                num_channels, patch_size, "NCHW", embed_dim
            )

        self.model = ViTRoPE(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,  # dimensions at each stage
            depth=depth,  # transformer of depth 4 at each stage
            heads=num_heads,  # heads at each stage
            mlp_dim=mlp_ratio * embed_dim,
            dropout=0.0,
            dim_head=32,
            channels=num_channels,
            rotary_position_emb=rotary_position_emb,
            conditionnal_posemb=conditionnal_posemb,
            cond_kernel_size=cond_kernel_size,
            learned_masked_emb=learned_masked_emb,
            patch_embedder=patch_embedder,
        )

        if pretrained:
            state_dict = load_state_dict(pretrained_path, model_name="model")
            state_dict.pop("mlp_head.1.weight", None)
            state_dict.pop("mlp_head.1.bias", None)

            self.model.load_state_dict(state_dict, strict=False)

        # Measure downsample factor
        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.model, self.num_channels, self.img_size)

        # Add segmentation head
        self.seg_head = segmentation_head(
            self.embed_dim,
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
        )

        if context_token >= 1:
            self.context_model = create_context_network(
                self.num_channels_token,
                self.embed_dim,
                context_token,
                context_network_shape,
            )
            self.auxiliary_head = nn.Sequential(
                Rearrange(
                    "b (h w) c -> b c h w", h=context_token, w=context_token
                ),
                nn.Conv2d(self.embed_dim, self.embed_dim // 2, 1),
                nn.BatchNorm2d(self.embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(self.embed_dim // 2, self.embed_dim // 4, 1),
                nn.BatchNorm2d(self.embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(self.embed_dim // 4, self.num_classes, 1),
            )

    def forward(self, x, metas=None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the output tensor.
        """
        if isinstance(x, list):
            assert (
                len(x) == 2
            ), "only support main image and one additional context image"
            context = x[1]
            x = x[0]
        x = self.model.forward_features(x, context=context)

        x = self.seg_head(x)
        context_aux = (
            self.auxiliary_head(context) if context is not None else None
        )
        return {"out": x, "context_aux": context_aux}


class ViTRoPE_MAE(nn.Module):
    """ViTRoPE_MAE is a modified version of the ViTRoPE model that includes a Masked Autoencoder
    (MAE) decoder.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 4.
        num_channels (int, optional): Number of input channels. Defaults to 1.
        segmentation_head (nn.Module, optional): Segmentation head module(not initialized). Defaults to SimpleSegmentationHead.
        pretrained (bool, optional): Whether to use a pretrained model. Defaults to True.
        pretrained_path (str, optional): Path to the pretrained model. Defaults to None.
        img_size (int, optional): Size of the input image. Defaults to 512.
        patch_size (int, optional): Size of the ViT patches. Defaults to 16.
        embed_dim (int, optional): Dimension of the ViT embeddings. Defaults to 384.
        depth (int, optional): Depth of the ViT model. Defaults to 12.
        num_heads (int, optional): Number of heads in the ViT model. Defaults to 6.
        mlp_ratio (int, optional): Ratio of the MLP dimension to the ViT embedding dimension. Defaults to 4.
        MAE_depth (int, optional): Depth of the MAE decoder. Defaults to 3.
        MAE_drop_perc (float, optional): Percentage of patches to drop in the MAE decoder. Defaults to 0.75.
        rotary_position_emb (bool, optional): Whether to use rotary position embeddings. Defaults to True.
        conditionnal_posemb (bool, optional): Whether to use conditional position embeddings. Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_classes=4,
        num_channels=1,
        segmentation_head=SimpleSegmentationHead,
        pretrained=True,
        pretrained_path=None,
        img_size=512,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        MAE_drop_perc=0.75,
        rotary_position_emb=True,
        conditionnal_posemb=False,
        cond_kernel_size=3,
        learned_masked_emb=False,
        context_token=0,
        context_network_shape=[1, 1, 1, 1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.img_size = img_size
        self.MAE_drop_prob = MAE_drop_perc
        self.patch_size = patch_size

        self.model = ViTRoPE(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,  # dimensions at each stage
            depth=depth,
            heads=num_heads,  # heads at each stage
            mlp_dim=mlp_ratio * embed_dim,
            dropout=0.0,
            dim_head=32,
            channels=num_channels,
            rotary_position_emb=rotary_position_emb,
            conditionnal_posemb=conditionnal_posemb,
            cond_kernel_size=cond_kernel_size,
            learned_masked_emb=learned_masked_emb,
        )

        self.MAE_decoder = ViTRoPE(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=decoder_embed_dim,  # dimensions at each stage
            depth=decoder_depth,
            heads=decoder_num_heads,  # heads at each stage
            mlp_dim=mlp_ratio * embed_dim,
            dropout=0.0,
            dim_head=32,
            channels=num_channels,
            rotary_position_emb=rotary_position_emb,
            conditionnal_posemb=conditionnal_posemb,
            cond_kernel_size=cond_kernel_size,
            learned_masked_emb=learned_masked_emb,
        )

        # Measure downsample factor
        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.model, self.num_channels, self.img_size)

        # Add segmentation head
        self.seg_head = segmentation_head(
            self.embed_dim,
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
        )

        self.masked_token = nn.Parameter(torch.randn(1, self.embed_dim))

        if context_token >= 1:
            self.context_model = create_context_network(
                self.num_channels,
                self.embed_dim,
                context_token,
                context_network_shape,
            )

    def forward(self, x, metas=None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the output tensor.
        """
        if isinstance(x, list):
            assert (
                len(x) == 2
            ), "only support main image and one additional context image"
            context = x[1]
            x = x[0]

        x = self.model.to_patch_embedding(x)
        # drop patches for MAE
        # rand = torch.rand(x.shape[0], x.shape[1]).to(x.device)
        # ceiling = torch.kthvalue(rand, int(x.shape[1] * self.MAE_drop_prob), dim=1).values
        # pos_kept = rand > ceiling[:, None]
        perm = torch.stack(
            [torch.randperm(x.shape[1]) for _ in range(x.shape[0])], dim=0
        )
        pos_kept = perm > int(x.shape[1] * self.MAE_drop_prob)
        pos_kept = pos_kept.to(x.device)

        # isolate patches to keep
        x = (compress_mask(x, pos_kept), pos_kept)
        x = self.model.forward_features(x, embedded=True, context=context)

        # fill with patch to learn
        x_filled = self.masked_token.repeat(
            x.shape[0], pos_kept.shape[1], 1
        ).to(x.device)
        x_filled[pos_kept] = x.view(-1, x.shape[2])

        x_decoded = self.MAE_decoder.forward_features(x_filled, embedded=True)

        out = self.seg_head(x_decoded)
        return {"out": out, "mask": pos_kept.unsqueeze(-1), "context": context}


def create_context_network(
    input_dim, ouput_dim, context_token_size, shape=[1, 1, 1, 1]
):
    min_emb_dim = 16
    if shape[0] > 1:
        blocks = [
            nn.Conv2d(input_dim, min_emb_dim, 3, padding="same"),
            nn.GELU(),
        ]
    else:  # we downsample directly
        blocks = [
            nn.Conv2d(input_dim, min_emb_dim * 2, 3, padding="same"),
            nn.GELU(),
        ]
    for i, depth in enumerate(shape[:-1]):
        assert depth >= 1, "depth of context network layer must be at least 1"
        emb_dim = min_emb_dim * 2**i
        if i == 0:
            depth -= 1
            if depth == 0:  # we downsample directly
                layer = nn.Sequential(nn.MaxPool2d(2))
                blocks.append(layer)
                continue
        layer = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Conv2d(emb_dim, emb_dim, 3, padding="same"),
                        nn.GELU(),
                    )
                    for d in range(depth - 1)
                ]
                + [
                    nn.Sequential(
                        nn.Conv2d(emb_dim, emb_dim * 2, 3, padding="same"),
                        nn.GELU(),
                        nn.MaxPool2d(2),
                    )
                ]
            )
        )
        blocks.append(layer)
    last_layer = nn.Sequential(
        *(
            [
                nn.Sequential(
                    nn.Conv2d(emb_dim * 2, emb_dim * 2, 3, padding="same"),
                    nn.GELU(),
                )
                for d in range(shape[-1] - 1)
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(emb_dim * 2, ouput_dim, 3, padding="same")
                ),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((context_token_size, context_token_size)),
                Rearrange("b c h w -> b (h w) c"),
            ]
        )
    )
    blocks.append(last_layer)

    return nn.Sequential(*blocks)
