import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger
from typing import Optional, Tuple


@cache
def log_glora_extended_drop():
    return logger.warning(
        "Using GLoRAExtended with bypass_mode=False will result in network or LoRA dropout "
        "being applied to the forward input instead of the layers. Requiring much lower values for dropout. "
        "Note: Bypass mode may not behave the same, so test and compare if desired."
    )


class GLoRAExtendedModule(LycorisBaseModule):
    name = "glora-ex"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    # Order matters for Lycoris framework's extract_state_dict and make_module_from_state_dict
    weight_list = [
        "alpha",      # 0
        "scalar",     # 1 (Optional, will be None if not use_scalar)
        "a1.weight",  # 2
        "a2.weight",  # 3
        "b1.weight",  # 4
        "b2.weight",  # 5
        "c1.weight",  # 6
        "c2.weight",  # 7
        "d_param",    # 8 (Optional)
        "e_param",    # 9 (Optional)
    ]
    weight_list_det = ["a1.weight"] # Used for algo_check

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        lora_dropout: float = 0.0,
        apply_d: bool = True,
        apply_e: bool = True,
        rs_lora: bool = False,
        use_scalar: bool = False,
        **kwargs,
    ):
        super().__init__(
            lora_name = lora_name,
            org_module = org_module,
            multiplier = multiplier,
            dropout = dropout,
            rank_dropout = rank_dropout,
            module_dropout = module_dropout,
            lora_dropout = lora_dropout,
            rank_dropout_scale = False,
            bypass_mode = False, # Must be False as bypass_forward is not implemented
            ggpo_beta = None,
            ggpo_sigma = None,
            ggpo_conv = False,
            ggpo_conv_weight_sample_size = 0,
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in GLoRAExtended.")

        if (dropout > 0 or lora_dropout > 0):
            log_glora_extended_drop()

        self.lora_dim = lora_dim
        self.alpha_val = alpha
        self.apply_d_param = apply_d
        self.apply_e_param = apply_e
        self.rs_lora = rs_lora

        if self.module_type.startswith("conv"):
            self.isconv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            conv_module_type = self.module # type(org_module) e.g. nn.Conv2d
        else: # Linear
            self.isconv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            linear_module_type = self.module # type(org_module) e.g. nn.Linear

        # A: W₀A (A_eff is in_dim x in_dim)
        # a1: in_dim -> lora_dim (effectively)
        # a2: lora_dim -> in_dim (effectively)
        if self.isconv:
            self.a1 = conv_module_type(in_dim, lora_dim, kernel_size=1, bias=False) # output (lora_dim, in_dim,1,1)
            self.a2 = conv_module_type(lora_dim, in_dim, kernel_size=1, bias=False) # output (in_dim, lora_dim,1,1)
        else:
            self.a1 = linear_module_type(in_dim, lora_dim, bias=False) # (lora_dim, in_dim)
            self.a2 = linear_module_type(lora_dim, in_dim, bias=False) # (in_dim, lora_dim)

        # B: Bx (B_eff is out_dim x in_dim)
        # b1: in_dim -> lora_dim
        # b2: lora_dim -> out_dim
        if self.isconv:
            self.b1 = conv_module_type(
                in_dim, lora_dim, kernel_size=k_size, # Use stored kernel_size
                stride=stride, padding=padding, bias=False
            )
            self.b2 = conv_module_type(lora_dim, out_dim, kernel_size=1, bias=False)
        else: # Linear
            self.b1 = linear_module_type(in_dim, lora_dim, bias=False)
            self.b2 = linear_module_type(lora_dim, out_dim, bias=False)

        # C: CW₀ (C_eff is in_dim x 1 vector, result is out_dim bias term)
        # c1: in_dim -> lora_dim
        # c2: lora_dim -> 1
        if self.isconv:
            self.c1 = conv_module_type(in_dim, lora_dim, kernel_size=1, bias=False)
            self.c2 = conv_module_type(lora_dim, 1, kernel_size=1, bias=False)
        else:
            self.c1 = linear_module_type(in_dim, lora_dim, bias=False)
            self.c2 = linear_module_type(lora_dim, 1, bias=False)

        # D: Db₀
        if self.apply_d_param and self.org_module[0].bias is not None:
            self.d_param = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_buffer("d_param", torch.zeros(0), persistent=False)

        # E: Additive bias E
        if self.apply_e_param:
            self.e_param = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_buffer("e_param", torch.zeros(0), persistent=False)

        if isinstance(self.alpha_val, torch.Tensor):
            self.alpha_val = self.alpha_val.detach().float().item()

        r_factor = float(lora_dim)
        if self.rs_lora and r_factor > 0:
            r_factor = math.sqrt(r_factor)

        self.lora_scaling = self.alpha_val / r_factor if r_factor > 0 else 0.0
        self.register_buffer("alpha", torch.tensor(self.alpha_val))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
            init_kaiming = lambda w: torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            init_kaiming(self.a1.weight); init_kaiming(self.a2.weight)
            init_kaiming(self.b1.weight); init_kaiming(self.b2.weight)
            init_kaiming(self.c1.weight); init_kaiming(self.c2.weight)
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
            # W₀(A₂A₁), (B₂B₁), (C₂C₁)W₀
            # To make delta zero, init A₁ or A₂ to zero (etc.)
            torch.nn.init.kaiming_uniform_(self.a1.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.a2.weight)
            torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.b2.weight)
            torch.nn.init.kaiming_uniform_(self.c1.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.c2.weight)

        if isinstance(self.d_param, nn.Parameter): torch.nn.init.zeros_(self.d_param)
        if isinstance(self.e_param, nn.Parameter): torch.nn.init.zeros_(self.e_param)

    def _get_current_scalar_value(self) -> float:
        return self.scalar.item()

    def _calculate_raw_delta_weight_and_bias(self, device=None, dtype=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype

        org_w = self.org_module[0].weight.to(device=device, dtype=dtype)

        # A: W₀A term, where A = A₂A₁
        A1_w = self.a1.weight.to(device=device, dtype=dtype)
        A2_w = self.a2.weight.to(device=device, dtype=dtype)

        if self.isconv:
            A1_matrix = A1_w.view(A1_w.size(0), -1) # (lora_dim, in_c)
            A2_matrix = A2_w.view(A2_w.size(0), -1) # (in_c, lora_dim)
            w_temp = torch.einsum("o i ..., i j -> o j ...", org_w, A2_matrix) # W₀ @ A₂
            W_delta_A = torch.einsum("o i ..., i j -> o j ...", w_temp, A1_matrix) # (W₀A₂) @ A₁ = W₀(A₂A₁)
        else:
            W_delta_A = (org_w @ A2_w) @ A1_w # W₀(A₂A₁)

        # B: Bx term, where B = B₂B₁
        B1_w = self.b1.weight.to(device=device, dtype=dtype)
        B2_w = self.b2.weight.to(device=device, dtype=dtype)
        if self.isconv:
            B2_w_matrix = B2_w.view(B2_w.size(0), B2_w.size(1)) # (out_dim, lora_dim) from (out_dim, lora_dim,1,1)
            W_delta_B = torch.einsum('or,ri...->oi...', B2_w_matrix, B1_w) # (out_dim, lora_dim) @ (lora_dim, in_dim, Kh, Kw)
        else:
            W_delta_B = B2_w @ B1_w
        raw_delta_W = W_delta_A + W_delta_B

        # C: CW₀ term (results in a bias), where C_eff = (C₂C₁)^T
        C1_w = self.c1.weight.to(device=device, dtype=dtype)
        C2_w = self.c2.weight.to(device=device, dtype=dtype)
        if self.isconv:
            C1_matrix = C1_w.view(C1_w.size(0), -1) # (lora_dim, in_c)
            C2_matrix = C2_w.view(C2_w.size(0), -1) # (1, lora_dim)
        else:
            C1_matrix, C2_matrix = C1_w, C2_w # (lora_dim, in_dim), (1, lora_dim)

        C_eff_vec = (C2_matrix @ C1_matrix).T # (in_dim, 1) or (in_c, 1)

        if self.isconv:
            bias_C_contrib_raw = torch.einsum("oi...,i->o...", org_w, C_eff_vec.squeeze())
            bias_C_contrib = bias_C_contrib_raw.mean(dim=list(range(1, bias_C_contrib_raw.dim()))) if bias_C_contrib_raw.dim() > 1 else bias_C_contrib_raw
        else:
            bias_C_contrib = (org_w @ C_eff_vec).squeeze(-1)

        raw_bias_terms = [bias_C_contrib]
        if isinstance(self.d_param, nn.Parameter) and self.org_module[0].bias is not None:
            org_b = self.org_module[0].bias.to(device=device, dtype=dtype)
            D_scaler = self.d_param.to(device=device, dtype=dtype)
            raw_bias_terms.append(D_scaler * org_b)
        if isinstance(self.e_param, nn.Parameter):
            raw_bias_terms.append(self.e_param.to(device=device, dtype=dtype))

        valid_bias_terms = [t for t in raw_bias_terms if t is not None and t.numel() > 0]
        raw_delta_Bias = sum(valid_bias_terms) if valid_bias_terms else None

        return raw_delta_W, raw_delta_Bias

    def get_diff_weight(self, multiplier:float=1.0, shape=None, device=None, dtype=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if dtype is None: dtype = self.dtype # Use module's default dtype if not specified
        raw_delta_W, raw_delta_B = self._calculate_raw_delta_weight_and_bias(device=device, dtype=dtype)

        current_scalar = self._get_current_scalar_value()
        effective_scale = self.lora_scaling * current_scalar * multiplier

        delta_W = raw_delta_W * effective_scale
        delta_B = raw_delta_B * effective_scale if raw_delta_B is not None else None

        if shape is not None and delta_W.shape != shape: delta_W = delta_W.view(shape)
        return delta_W, delta_B

    def get_merged_weight(self, multiplier:float=1.0, shape=None, device=None, dtype=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if dtype is None: dtype = self.dtype
        delta_W, delta_B = self.get_diff_weight(multiplier, shape, device, dtype)
        org_w = self.org_module[0].weight.to(device=device, dtype=dtype)
        merged_W = org_w + delta_W

        org_b = self.org_module[0].bias
        merged_B = None
        if org_b is not None:
            merged_B = org_b.to(device=device, dtype=dtype)
            if delta_B is not None:
                merged_B = merged_B + delta_B
        elif delta_B is not None:
            merged_B = delta_B
        return merged_W, merged_B

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.module_dropout > 0 and self.training and torch.rand(1) < self.module_dropout:
            return self.org_forward(x)

        merged_W, merged_B = self.get_merged_weight(multiplier=self.multiplier, device=x.device, dtype=x.dtype)

        x_eff = x
        if self.dropout > 0 and self.training:
            x_eff = self.drop(x_eff)

        if self.lora_dropout > 0 and self.training:
            input_mask_shape = [1] * x.dim()
            last_dim_size = x.shape[1] if self.isconv and x.dim() > 2 else x.shape[-1]
            if self.isconv and x.dim() > 2: input_mask_shape[1] = -1
            else: input_mask_shape[-1] = -1

            if last_dim_size > 0 :
                input_mask = torch.bernoulli(
                    torch.ones(last_dim_size, device=x.device, dtype=x.dtype) * (1 - self.lora_dropout)
                ).view(*input_mask_shape)
                x_eff = x_eff * input_mask

        return self.op(x_eff, merged_W, merged_B, **self.kw_dict)

    def custom_state_dict(self):
        sd = {
            "alpha": self.alpha,
            "a1.weight": self.a1.weight, "a2.weight": self.a2.weight,
            "b1.weight": self.b1.weight, "b2.weight": self.b2.weight,
            "c1.weight": self.c1.weight, "c2.weight": self.c2.weight,
        }
        if isinstance(self.scalar, nn.Parameter): sd["scalar"] = self.scalar.data
        if isinstance(self.d_param, nn.Parameter): sd["d_param"] = self.d_param
        if isinstance(self.e_param, nn.Parameter): sd["e_param"] = self.e_param
        return sd

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        alpha: torch.Tensor,
        scalar: Optional[torch.Tensor] = None,
        a1_weight: Optional[torch.Tensor] = None, a2_weight: Optional[torch.Tensor] = None,
        b1_weight: Optional[torch.Tensor] = None, b2_weight: Optional[torch.Tensor] = None,
        c1_weight: Optional[torch.Tensor] = None, c2_weight: Optional[torch.Tensor] = None,
        d_param_tensor: Optional[torch.Tensor] = None, e_param_tensor: Optional[torch.Tensor] = None,
    ):
        if a1_weight is None:
            raise ValueError("a1.weight is required for GLoRAExtendedModule.")

        lora_dim = a1_weight.size(0)

        module = cls(
            lora_name=lora_name, org_module=orig_module, lora_dim=lora_dim,
            alpha=alpha.item(),
            apply_d=(d_param_tensor is not None), apply_e=(e_param_tensor is not None),
            use_scalar=(scalar is not None)
        )

        def _copy_param_or_buffer(target, source_tensor, name_for_log):
            if source_tensor is not None:
                if target.shape == source_tensor.shape:
                    target.copy_(source_tensor)
                else:
                    logger.error(f"Shape mismatch for {name_for_log}: target{target.shape}, source{source_tensor.shape}")

        _copy_param_or_buffer(module.a1.weight.data, a1_weight, "a1.weight")
        _copy_param_or_buffer(module.a2.weight.data, a2_weight, "a2.weight")
        _copy_param_or_buffer(module.b1.weight.data, b1_weight, "b1.weight")
        _copy_param_or_buffer(module.b2.weight.data, b2_weight, "b2.weight")
        _copy_param_or_buffer(module.c1.weight.data, c1_weight, "c1.weight")
        _copy_param_or_buffer(module.c2.weight.data, c2_weight, "c2.weight")

        if scalar is not None: _copy_param_or_buffer(module.scalar.data if isinstance(module.scalar, nn.Parameter) else module.scalar, scalar, "scalar")
        if d_param_tensor is not None: _copy_param_or_buffer(module.d_param.data, d_param_tensor, "d_param")
        if e_param_tensor is not None: _copy_param_or_buffer(module.e_param.data, e_param_tensor, "e_param")

        return module

    @torch.no_grad()
    def get_norm(self, device=None, dtype=None) -> Optional[Tuple[float, float]]:
        if dtype is None: dtype = self.dtype
        raw_delta_w, _ = self._calculate_raw_delta_weight_and_bias(device, dtype)
        if raw_delta_w is None: return None

        current_scalar = self._get_current_scalar_value()
        unscaled_norm = raw_delta_w.norm()
        fully_scaled_delta_norm = unscaled_norm * self.lora_scaling * current_scalar
        return unscaled_norm.item(), fully_scaled_delta_norm.item()
