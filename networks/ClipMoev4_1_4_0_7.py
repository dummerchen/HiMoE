import re

import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from torch.distributions.normal import Normal
import clip
from networks.ViT_MoE import _load_weights, checkpoint_filter_fn, default_cfgs


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class HiMoE_Adapter(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, deep2idx={0: [0, 1, 2, 3], 1: [4, 5], 2: [6]},
                 adapter_dim: list = [8, 8, 8, 8, 8, 8, 8], adapter_name=['q', 'k', 'v'], adapter_type='mean',
                 link_group=2, link_stride=1, noisy_gating=True, k=1):
        super(HiMoE_Adapter, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(adapter_dim)
        self.deep2idx = deep2idx
        self.deep = []
        self.layer_num = []
        for depth, ids in self.deep2idx.items():
            self.deep.extend([depth + 1] * len(ids))
            self.layer_num.append(len(ids))
        self.link_group = link_group
        self.link_stride = link_stride
        self.k = k
        self.identity = nn.Identity()
        self.Lora_a_experts = nn.ModuleList()
        self.Lora_b_experts = nn.ModuleList()
        self.adapter_name = adapter_name
        self.adapter_type = adapter_type
        for qkv in range(len(self.adapter_name)):
            tmp_a = nn.ModuleList()
            tmp_b = nn.ModuleList()
            for d in adapter_dim:
                tmp_a.append(nn.Linear(in_channels, d, bias=False))
                nn.init.kaiming_uniform_(tmp_a[-1].weight, a=math.sqrt(5))
                tmp_b.append(nn.Linear(d, out_channels, bias=False))
                nn.init.zeros_(tmp_b[-1].weight)
            self.Lora_a_experts.append(tmp_a)
            self.Lora_b_experts.append(tmp_b)
        self.w_gate = nn.Parameter(torch.zeros(in_channels, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(in_channels, self.num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        """Args:
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        B, N, C = x.shape
        if self.adapter_type == 'mean':
            x_global = torch.mean(x, dim=1)
        else:
            x_global = x.reshape(B * N, C)
            x = x.reshape(B * N, C)

        gates, load = self.noisy_top_k_gating(x_global, self.training)

        self.importance = gates.sum(0)
        loss = self.cv_squared(self.importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        y = []
        for idx in range(len(self.adapter_name)):
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                if i in self.deep2idx[0]:
                    res = F.linear(F.linear(expert_inputs[i], self.Lora_a_experts[idx][i].weight),
                                   self.Lora_b_experts[idx][i].weight)
                elif i in self.deep2idx[1]:
                    exps = []
                    for j in self.deep2idx[0]:
                        exps.append(F.linear(F.linear(expert_inputs[i], self.Lora_a_experts[idx][j].weight),
                                             self.Lora_b_experts[idx][j].weight))
                    res = F.linear(
                        F.linear(exps[i - 3] + exps[i - 2] + expert_inputs[i], self.Lora_a_experts[idx][i].weight),
                        self.Lora_b_experts[idx][i].weight)
                elif i in self.deep2idx[2]:
                    exp0 = F.linear(F.linear(expert_inputs[i], self.Lora_a_experts[idx][0].weight),
                                    self.Lora_b_experts[idx][0].weight)
                    exp1 = F.linear(F.linear(expert_inputs[i], self.Lora_a_experts[idx][1].weight),
                                    self.Lora_b_experts[idx][1].weight)
                    exp2 = F.linear(F.linear(expert_inputs[i], self.Lora_a_experts[idx][2].weight),
                                    self.Lora_b_experts[idx][2].weight)
                    exp3 = F.linear(F.linear(exp0 + exp1, self.Lora_a_experts[idx][3].weight),
                                    self.Lora_b_experts[idx][3].weight)
                    exp4 = F.linear(F.linear(exp1 + exp2, self.Lora_a_experts[idx][4].weight),
                                    self.Lora_b_experts[idx][4].weight)
                    res = F.linear(F.linear(exp3 + exp4 + expert_inputs[i], self.Lora_a_experts[idx][5].weight),
                                   self.Lora_b_experts[idx][5].weight)

                expert_outputs.append(res.reshape(expert_inputs[i].shape[0], -1))
            y.append(dispatcher.combine(expert_outputs, multiply_by_gates=True))
        y = torch.stack(y, dim=-1).reshape(B, N, -1)
        return y, loss


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., lora_topk=1,
                 attn_adapter_deep2idx={0: [0, 1, 2, 3, 4, 5, 6, 7]}, attn_adapter_topk=1,
                 attn_adapter_dim=[8, 16, 32, 48, 64, 96, 128],
                 attn_adapter_name=['qkv'],
                 attn_adapter_type='reshape',
                 attn_link_group=1,
                 attn_link_stride=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_adapter_topk = attn_adapter_topk
        self.adapter_name = attn_adapter_name
        if self.attn_adapter_topk > 0:
            self.attn_adapter_moe = HiMoE_Adapter(in_channels=dim, out_channels=dim * 3 // len(attn_adapter_name),
                                                  k=attn_adapter_topk, adapter_dim=attn_adapter_dim,
                                                  adapter_name=attn_adapter_name, deep2idx=attn_adapter_deep2idx, adapter_type=attn_adapter_type)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.attn_adapter_topk > 0:
            qkv_delta, lora_loss = self.attn_adapter_moe(x)
            qkv_delta = qkv_delta.reshape(B, N, 3//len(self.adapter_name), self.num_heads, C // self.num_heads).permute(2,
                                                                                                                     0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     4)
            qkv_delta = qkv_delta.unbind(0)
            for name in self.adapter_name:
                if name == 'qkv':
                    q = q + qkv_delta[0]
                    k = k + qkv_delta[1]
                    v = v + qkv_delta[2]
                elif name == 'q':
                    q = q + qkv_delta[0]
                elif name == 'k':
                    k = k + qkv_delta[1]
                elif name == 'v':
                    v = v + qkv_delta[2]

        else:
            lora_loss = torch.zeros(1).to(x.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, lora_loss  # ,adapter_loss


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_adapter_topk=1, mlp_adapter_topk=1,
                 attn_adapter_dim=[8, 16, 32, 48, 64, 96, 128], mlp_adapter_dim=[8, 16, 32, 48, 64, 96, 128],
                 mlp_adapter_deep2idx={0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7]},
                 mlp_adapter_link_group=2, mlp_adapter_link_stride=1, mlp_adapter_name=['mlp'], mlp_adapter_type='mean',
                 attn_adapter_link_group=1, attn_adapter_link_stride=1, attn_adapter_name=['qkv'], attn_adapter_type='reshape',
                 attn_adapter_deep2idx={0: [0, 1, 2, 3, 4, 5, 6, 7]},
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              attn_adapter_topk=attn_adapter_topk, attn_adapter_dim=attn_adapter_dim,
                              attn_adapter_deep2idx=attn_adapter_deep2idx, attn_link_group=attn_adapter_link_group,
                              attn_link_stride=attn_adapter_link_stride, attn_adapter_name=attn_adapter_name, attn_adapter_type=attn_adapter_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_adapter_topk = mlp_adapter_topk
        if self.mlp_adapter_topk > 0:
            self.mlp_adapter_moe = HiMoE_Adapter(in_channels=dim, out_channels=dim, adapter_name=mlp_adapter_name,
                                                 adapter_dim=mlp_adapter_dim, k=self.mlp_adapter_topk,
                                                 deep2idx=mlp_adapter_deep2idx,
                                                 link_group=mlp_adapter_link_group,
                                                 link_stride=mlp_adapter_link_stride,
                                                 adapter_type=mlp_adapter_type
                                                 )

    def forward(self, x):
        x1, lora_loss = self.attn(self.norm1(x))
        x = x + self.drop_path(x1)
        if self.mlp_adapter_topk > 0:
            x_adapter, adapter_loss = self.drop_path(self.mlp_adapter_moe(self.norm2(x)))
            x = x + x_adapter + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            adapter_loss = torch.zeros(1).to(x.device)

        return x, lora_loss, adapter_loss


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,
                 attn_adapter_dim=[8, 16, 32, 48, 64, 96, 128],
                 attn_adapter_deep2idx={0: [0, 1, 2, 3, 4, 5, 6, 7]},
                 attn_adapter_link_group=1, attn_adapter_link_stride=1, attn_adapter_topk=1,
                 attn_adapter_name=['qkv'], attn_adapter_type='reshape',

                 mlp_adapter_dim=[8, 16, 32, 48, 64, 96, 128],
                 mlp_adapter_link_group=2, mlp_adapter_link_stride=1,
                 mlp_adapter_deep2idx={0: [0, 1, 2, 3], 1: [4, 5], 2: [6]}, mlp_adapter_topk=1,
                 mlp_adapter_name=['mlp'], mlp_adapter_type='mean'
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,

                attn_adapter_dim=attn_adapter_dim, attn_adapter_deep2idx=attn_adapter_deep2idx,
                attn_adapter_link_group=attn_adapter_link_group, attn_adapter_link_stride=attn_adapter_link_stride,
                attn_adapter_topk=attn_adapter_topk, attn_adapter_name=attn_adapter_name, attn_adapter_type=attn_adapter_type,

                mlp_adapter_dim=mlp_adapter_dim, mlp_adapter_deep2idx=mlp_adapter_deep2idx,
                mlp_adapter_link_group=mlp_adapter_link_group, mlp_adapter_link_stride=mlp_adapter_link_stride,
                mlp_adapter_topk=mlp_adapter_topk, mlp_adapter_name=mlp_adapter_name, mlp_adapter_type=mlp_adapter_type
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.attn_adapter_topk = attn_adapter_topk
        self.mlp_adapter_topk = mlp_adapter_topk

        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.freeze_stages()

    def freeze_stages(self):

        self.pos_drop.eval()
        self.patch_embed.eval()

        for block in self.blocks:
            block.eval()
            if self.attn_adapter_topk > 0:
                block.attn.attn_adapter_moe.train()
            if self.mlp_adapter_topk > 0:
                block.mlp_adapter_moe.train()

        for name, param in self.named_parameters():
            if 'LoRA' not in name and 'adapter' not in name and 'head' not in name and 'norm1' not in name:
                param.requires_grad = False

        total_para_nums = 0
        LoRA_para_nums = 0
        adapter_para_nums = 0
        head_para_nums = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_para_nums += param.numel()
                if 'LoRA' in name:
                    LoRA_para_nums += param.numel()
                elif 'head' in name:
                    head_para_nums += param.numel()
                elif 'adapter' in name:
                    adapter_para_nums += param.numel()

        print('total train parameters:', total_para_nums, 'LoRA', LoRA_para_nums, 'adapter', adapter_para_nums, 'head',
              head_para_nums)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        # build model -> load_custom_pretrained -> load_pretrained
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        lora_loss_list = []
        adapter_loss_list = []
        for block in self.blocks:
            x, cur_lora_loss, cur_adapter_loss = block(x)
            lora_loss_list.append(cur_lora_loss)
            adapter_loss_list.append(cur_adapter_loss)
        lora_loss = torch.mean(torch.stack(lora_loss_list))
        adapter_loss = torch.mean(torch.stack(adapter_loss_list))
        moe_loss = lora_loss * 200 + adapter_loss * 1
        x = self.norm(x)
        return self.pre_logits(x[:, 0]), moe_loss

    def forward(self, x):
        x, moe_loss = self.forward_features(x)
        x = self.head(x)
        return x, moe_loss


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.transformer.eval()
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        if (hasattr(clip_model, 'attn_mask')):
            self.attn_mask = clip_model.attn_mask
        else:
            self.attn_mask = None
        self.freeze_stages()

    def freeze_stages(self):
        for name, param in self.named_parameters():
            if 'LoRA' not in name and 'adapter' not in name and 'head' not in name and 'norm1' not in name:
                param.requires_grad = False

    def forward(self, prompts, tokenized_prompts):
        if len(prompts.shape) == 4:
            prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (n_class*(1+n_neg)) * n_ctx * dim
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(x.device)
            x = self.transformer(x, self.attn_mask)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, ctx_init, classnames, clip_model, device='cuda:0'):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("\"{}\"", "")
        ctx_init = ctx_init.replace(".", "")
        ctx_init = ctx_init.replace("_", " ")
        words = re.findall(r'\b\w+\b', ctx_init)
        n_ctx = len(words)
        prompt = clip.tokenize(ctx_init)
        prompt = prompt.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx_vectors = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # positive prompt CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self, label=None):
        # 3 (len_ctx) , d -> n cls, len_ctx, d
        ctx = self.ctx_vectors.unsqueeze(0).expand(self.n_cls, -1, -1)
        if label is not None:
            prefix = self.token_prefix[label]
            suffix = self.token_suffix[label]
        else:
            # 5(n cls) 1 d
            prefix = self.token_prefix
            # 5(n cls) 73 d
            suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls,n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class MergePrompt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.q_proj.weight)

    def forward(self, query, key=None, value=None, pos=1):
        key_real = -key
        if key is None:
            key = query
        if value is None:
            value = key
        q_real = self.q_proj(query[:pos])
        q_fake = self.q_proj(query[pos:])
        k_real = self.k_proj(key_real)
        k_fake = self.k_proj(key)
        v_real = key_real
        v_fake = value

        # 1 d @ cls d -> 1 cls
        attn_real = (q_real @ k_real.transpose(-2, -1)) * self.scale
        attn_fake = (q_fake @ k_fake.transpose(-2, -1)) * self.scale
        attn_real = torch.softmax(attn_real, dim=-1)
        attn_fake = torch.softmax(attn_fake, dim=-1)
        out_real = attn_real @ v_real
        out_fake = attn_fake @ v_fake
        out_real = query[:pos] + out_real
        out_fake = query[pos:] + out_fake
        out = torch.cat([out_real, out_fake], dim=0)
        return out


class ClipMoev4_1_4_0_7(nn.Module):
    def __init__(self, config, embed_dim=768, clip_model=None):
        super().__init__()
        self.config = config
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        device = clip_model.text_projection.device
        self.prompt_learner = PromptLearner(config['prompt_init'], config['prompt_classnames'], clip_model,
                                            device=device).to(device)
        self.text_encoder = TextEncoder(clip_model)
        model_cfg = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                         num_classes=config['backbone_config']['num_classes'],
                         attn_adapter_topk=config['backbone_config']['attn_adapter_topk'],
                         attn_adapter_dim=config['backbone_config']['attn_adapter_dim'],
                         attn_adapter_link_group=config['backbone_config']['attn_adapter_link_group'],
                         attn_adapter_link_stride=config['backbone_config']['attn_adapter_link_stride'],
                         attn_adapter_deep2idx=config['backbone_config']['attn_adapter_deep2idx'],
                         attn_adapter_name=config['backbone_config']['attn_adapter_name'],
                         attn_adapter_type=config['backbone_config']['attn_adapter_type'],

                         mlp_adapter_topk=config['backbone_config']['mlp_adapter_topk'],
                         mlp_adapter_dim=config['backbone_config']['mlp_adapter_dim'],
                         mlp_adapter_link_group=config['backbone_config']['mlp_adapter_link_group'],
                         mlp_adapter_link_stride=config['backbone_config']['mlp_adapter_link_stride'],
                         mlp_adapter_deep2idx=config['backbone_config']['mlp_adapter_deep2idx'],
                         mlp_adapter_name=config['backbone_config']['mlp_adapter_name'],
                         mlp_adapter_type=config['backbone_config']['mlp_adapter_type'],
                         )
        self.vision_encoder = build_model_with_cfg(
            VisionTransformer, 'vit_base_patch16_224', True,
            default_cfg=default_cfgs['vit_base_patch16_224'],
            pretrained_filter_fn=checkpoint_filter_fn,
            pretrained_custom_load=True,
            **model_cfg)
        self.len = config['backbone_config']['token_dim']
        token = torch.empty(config['backbone_config']['token_dim'], clip_model.ln_final.weight.shape[0])
        self.learnable_token = nn.Parameter(token)
        nn.init.normal_(self.learnable_token, std=0.02)
        self.attn = MergePrompt(dim=clip_model.ln_final.weight.shape[0])

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, data):
        if isinstance(data, dict):
            image = data['aug_image']
        else:
            image = data
        image_embeds, moe_loss = self.vision_encoder(image)
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        prompt = self.prompt_learner()
        text_embeds = self.text_encoder(prompt, tokenized_prompts=self.prompt_learner.tokenized_prompts)
        self.text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds_norm = self.attn(self.learnable_token, self.text_embeds_norm, pos=self.len//2)
        self.text_embeds_atten_norm = text_embeds_norm
        logits_per_images = self.logit_scale.exp() * image_embeds_norm @ text_embeds_norm.t()
        if self.config['use_distillation_loss']:
            dl = F.mse_loss(text_embeds_norm[:self.len//2, :], text_embeds_norm[self.len//2:, :])
            moe_loss -= dl
        # b,5 -> b,2
        logits_per_images = torch.stack(
            [torch.logsumexp(logits_per_images[:, :self.len//2], dim=1), torch.logsumexp(logits_per_images[:, self.len//2:], dim=1)], dim=1)
        return logits_per_images, moe_loss


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


if __name__ == '__main__':
    device = 'cuda:2'
    config = {'backbone_config': {
        "embed_dim": 768,
        "num_classes": 512,
        "token_dim": 5,

        "attn_adapter_topk": 1,
        "attn_adapter_dim": [8, 16, 32, 48, 64, 96, 128],
        "attn_adapter_deep2idx": {0: [0, 1, 2, 3, 4, 5, 6]},
        "attn_adapter_link_group": 7,
        "attn_adapter_link_stride": 1,
        "attn_adapter_name": ['qkv'],
        "attn_adapter_type": 'mean',

        "mlp_adapter_topk": 2,
        "mlp_adapter_dim": [8, 8, 8, 16, 16, 32],
        "mlp_adapter_deep2idx": {0: [0, 1, 2], 1: [3, 4], 2: [5]},
        "mlp_adapter_link_group": 2,
        "mlp_adapter_link_stride": 1,
        "mlp_adapter_name": ['mlp'],
        "mlp_adapter_type": 'mean'
    },
        "use_distillation_loss": False,
        "prompt_init": 'this_face_photo_is',
        "prompt_classnames": [
            'fake', 'face-swapped', 'edited in facial expressions', 'manipulated in facial attributes']
    }
    clip_model, preprocess = clip.load('ViT-B/16', device=device)
    encode_text_func = clip_model.encode_text

    model = ClipMoev4_1_4_0_7(config, clip_model=clip_model)
    model.to(device)
    image = torch.randn(2, 3, 224, 224).to(device)
    res, _ = model(image)
    print(res.shape)
