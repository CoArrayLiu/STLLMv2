"""Transformer ablations for QK attention and adaptive-graph mixing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_MODES = ("qk", "graph", "qk_graph")


class TemporalEmbedding(nn.Module):
    def __init__(self, time_slots: int, features: int) -> None:
        super().__init__()
        self.time_slots = time_slots
        self.time_day = nn.Parameter(torch.empty(time_slots, features))
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        day_index = (x[:, -1, :, 1] * self.time_slots).long()
        day_index = day_index.clamp_(0, self.time_slots - 1)
        week_index = x[:, -1, :, 2].long().clamp_(0, 6)
        time_day = self.time_day[day_index].transpose(1, 2).unsqueeze(-1)
        time_week = self.time_week[week_index].transpose(1, 2).unsqueeze(-1)
        return time_day + time_week


def prepare_adaptive_graph(
    adaptive_graph: torch.Tensor,
    num_nodes: int,
    epsilon: float,
) -> torch.Tensor:
    """Validate and row-normalize an already weighted adaptive graph."""

    graph = torch.as_tensor(adaptive_graph, dtype=torch.float32).detach().clone()
    expected_shape = (num_nodes, num_nodes)
    if tuple(graph.shape) != expected_shape:
        raise ValueError(
            f"Adaptive graph shape mismatch: expected {expected_shape}, "
            f"got {tuple(graph.shape)}"
        )
    if not torch.isfinite(graph).all():
        raise ValueError("Adaptive graph contains NaN or infinite values")
    if torch.any(graph < 0):
        minimum = graph.min().item()
        raise ValueError(
            "Adaptive graph must contain non-negative allocation weights; "
            f"minimum value is {minimum}"
        )
    row_sum = graph.sum(dim=-1, keepdim=True)
    if torch.any(row_sum <= epsilon):
        bad_rows = (row_sum.squeeze(-1) <= epsilon).nonzero().flatten().tolist()
        raise ValueError(f"Adaptive graph has zero-sum rows: {bad_rows[:10]}")
    return graph / row_sum


class SpatialTokenMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_mode: str,
        adaptive_graph: Optional[torch.Tensor],
        attention_dropout: float = 0.1,
        graph_alpha: float = 1.0,
        graph_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"Unknown attention_mode={attention_mode!r}; "
                f"choose one of {ATTENTION_MODES}"
            )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if graph_alpha <= 0:
            raise ValueError("graph_alpha must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_mode = attention_mode
        self.graph_epsilon = graph_epsilon

        if attention_mode in {"qk", "qk_graph"}:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
        else:
            self.q_proj = None
            self.k_proj = None
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(attention_dropout)

        if attention_mode in {"graph", "qk_graph"}:
            if adaptive_graph is None:
                raise ValueError(
                    f"attention_mode={attention_mode!r} requires an adaptive graph"
                )
            self.register_buffer("adaptive_graph", adaptive_graph.clone())
        else:
            self.register_buffer("adaptive_graph", None)

        if attention_mode == "qk_graph":
            raw_alpha = math.log(math.expm1(graph_alpha))
            self.raw_graph_alpha = nn.Parameter(torch.tensor(raw_alpha))
        else:
            self.register_parameter("raw_graph_alpha", None)

    @property
    def graph_alpha(self) -> Optional[torch.Tensor]:
        if self.raw_graph_alpha is None:
            return None
        return F.softplus(self.raw_graph_alpha)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        return x.view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        value = self._split_heads(self.v_proj(x))

        if self.attention_mode == "graph":
            allocation = self.adaptive_graph.view(1, 1, num_nodes, num_nodes)
            allocation = allocation.expand(
                batch_size, self.num_heads, num_nodes, num_nodes
            )
            allocation = self.attention_dropout(allocation)
        else:
            query = self._split_heads(self.q_proj(x))
            key = self._split_heads(self.k_proj(x))
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            if self.attention_mode == "qk_graph":
                graph_bias = torch.log(
                    self.adaptive_graph.clamp_min(self.graph_epsilon)
                )
                scores = scores + self.graph_alpha * graph_bias.view(
                    1, 1, num_nodes, num_nodes
                )
            allocation = torch.softmax(scores, dim=-1)
            allocation = self.attention_dropout(allocation)

        mixed = torch.matmul(allocation, value)
        mixed = mixed.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, self.d_model
        )
        return self.out_proj(mixed)


class AdaptiveGraphTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        attention_mode: str,
        adaptive_graph: Optional[torch.Tensor],
        dropout: float,
        graph_alpha: float,
        graph_epsilon: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = SpatialTokenMixer(
            d_model=d_model,
            num_heads=num_heads,
            attention_mode=attention_mode,
            adaptive_graph=adaptive_graph,
            attention_dropout=dropout,
            graph_alpha=graph_alpha,
            graph_epsilon=graph_epsilon,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.mixer(self.norm1(x)))
        x = x + self.residual_dropout(self.ffn(self.norm2(x)))
        return x


class STTransformerAdaptive(nn.Module):
    def __init__(
        self,
        adaptive_graph: Optional[torch.Tensor],
        time_slots: int,
        input_dim: int = 3,
        num_nodes: int = 250,
        input_len: int = 12,
        output_len: int = 12,
        attention_mode: str = "qk",
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        ffn_dim: int = 3072,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        graph_alpha: float = 1.0,
        graph_epsilon: float = 1e-8,
        learn_node_embedding: bool = True,
        learn_prediction_head: bool = True,
    ) -> None:
        super().__init__()
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"Unknown attention_mode={attention_mode!r}; "
                f"choose one of {ATTENTION_MODES}"
            )
        if input_dim < 3:
            raise ValueError(
                "input_dim must be at least 3: value, time-of-day, day-of-week"
            )
        if time_slots <= 0:
            raise ValueError("time_slots must be positive")

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.time_slots = time_slots
        self.attention_mode = attention_mode
        self.embedding_dim = embedding_dim

        graph = None
        if attention_mode in {"graph", "qk_graph"}:
            if adaptive_graph is None:
                raise ValueError(
                    f"attention_mode={attention_mode!r} requires adaptive_graph"
                )
            graph = prepare_adaptive_graph(
                adaptive_graph=adaptive_graph,
                num_nodes=num_nodes,
                epsilon=graph_epsilon,
            )

        self.start_conv = nn.Conv2d(
            input_dim * input_len, embedding_dim, kernel_size=(1, 1)
        )
        self.temporal_embedding = TemporalEmbedding(time_slots, embedding_dim)
        if learn_node_embedding:
            self.node_embedding = nn.Parameter(
                torch.empty(num_nodes, embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_embedding)
        else:
            self.register_parameter("node_embedding", None)
        self.input_projection = nn.Conv2d(
            embedding_dim * 3, d_model, kernel_size=(1, 1)
        )
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                AdaptiveGraphTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    attention_mode=attention_mode,
                    adaptive_graph=graph,
                    dropout=dropout,
                    graph_alpha=graph_alpha,
                    graph_epsilon=graph_epsilon,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        if learn_prediction_head:
            self.regression_layer = nn.Conv2d(
                d_model, output_len, kernel_size=(1, 1)
            )
        else:
            self.regression_layer = None

    def param_num(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def graph_alphas(self) -> list[float]:
        alphas = []
        for layer in self.layers:
            if layer.mixer.graph_alpha is not None:
                alphas.append(layer.mixer.graph_alpha.detach().item())
        return alphas

    def forward(
        self,
        history_data: torch.Tensor,
        node_embedding_override: Optional[torch.Tensor] = None,
        prediction_head_override: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if history_data.ndim != 4:
            raise ValueError(
                "history_data must have shape [batch, features, nodes, history]"
            )
        batch_size, input_dim, num_nodes, input_len = history_data.shape
        if (input_dim, input_len) != (self.input_dim, self.input_len):
            raise ValueError(
                "Input shape mismatch after batch dimension: expected "
                f"input_dim/input_len={(self.input_dim, self.input_len)}, got "
                f"{(input_dim, input_len)}"
            )
        if self.attention_mode != "qk" and num_nodes != self.num_nodes:
            raise ValueError(
                "Graph attention requires the configured number of nodes: "
                f"expected {self.num_nodes}, got {num_nodes}"
            )

        if node_embedding_override is not None:
            expected_shape = (num_nodes, self.embedding_dim)
            if tuple(node_embedding_override.shape) != expected_shape:
                raise ValueError(
                    "Node embedding override shape mismatch: expected "
                    f"{expected_shape}, got {tuple(node_embedding_override.shape)}"
                )
            node_embedding = node_embedding_override
        elif self.node_embedding is not None:
            if num_nodes != self.num_nodes:
                raise ValueError(
                    "Learned node embedding requires the configured number of "
                    f"nodes: expected {self.num_nodes}, got {num_nodes}"
                )
            node_embedding = self.node_embedding
        else:
            raise ValueError(
                "This model requires node_embedding_override because its internal "
                "node embedding is disabled"
            )

        data = history_data.permute(0, 3, 2, 1)
        temporal = self.temporal_embedding(data)
        node = node_embedding.transpose(0, 1).view(
            1, -1, num_nodes, 1
        )
        node = node.expand(batch_size, -1, -1, -1)

        flattened = history_data.transpose(1, 2).contiguous()
        flattened = flattened.view(batch_size, num_nodes, -1)
        flattened = flattened.transpose(1, 2).unsqueeze(-1)
        encoded = self.start_conv(flattened)

        tokens = torch.cat((encoded, temporal, node), dim=1)
        tokens = F.leaky_relu(self.input_projection(tokens))
        tokens = tokens.permute(0, 2, 1, 3).squeeze(-1)
        tokens = self.input_dropout(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)
        outputs = tokens.permute(0, 2, 1).unsqueeze(-1)
        prediction_head = prediction_head_override
        if prediction_head is None:
            prediction_head = self.regression_layer
        if prediction_head is None:
            raise ValueError(
                "This model requires prediction_head_override because its internal "
                "prediction head is disabled"
            )
        return prediction_head(outputs)
