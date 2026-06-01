from argparse import Namespace
from copy import deepcopy

import torch
from huggingface_hub import PyTorchModelHubMixin 
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model
import numpy as np
from utils.common import TASKS
from data_provider.base import TimeseriesOutputs
from layers.embed import PatchEmbedding, Patching,CircaDataEmbedding
from layers.revin import RevIN
from utils.masking import Masking
from layers.FANLayer import FANLayer
from scipy.interpolate import CubicSpline

from utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout) 
        if reduction == "mean":
            self.linear = nn.Linear(d_model, n_classes)
        elif reduction == "concat":
            self.linear = nn.Sequential(
                nn.Linear(n_channels * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model // 2, n_classes-1)
            )
        else:
            raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")
    def forward(self, x, input_mask: torch.Tensor = None):
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        y = self.linear(x)
        return y
    
    
class DiffRhythmHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 2,
        d_model: int = 512,
        n_classes_head1: int = 2,
        n_classes_head2: int = 2,
        head_dropout: int = 0.1,
        reduction: str = "concat",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        
        self.linear_head1 = nn.Sequential(
            nn.Linear(n_channels * d_model + 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes_head1)
        )
        
        self.linear_head2 = nn.Sequential(
            nn.Linear(n_channels * d_model + 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes_head2)
        )

    def forward(self, x, feature):
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = torch.cat((x, feature), dim=1)
        
        y_head1 = self.linear_head1(x)
        y_head2 = self.linear_head2(x)
        
        combined_logits = torch.cat((y_head1, y_head2), dim=1)
        
        return combined_logits, y_head1.shape[1]


class CIRCALLM(nn.Module):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len
        self.normalizer = RevIN(num_features=1, affine=config.getattr("revin_affine", False))

        self.data_embedding = CircaDataEmbedding(
            d_model=config.d_model,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )

        self.encoder = self._get_transformer_backbone(config)

        self.head = self._get_head(self.task_name)

        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.data_embedding = freeze_parameters(self.data_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _update_inputs(
        self, config: Namespace | dict, **kwargs: dict
    ) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            config.d_model = config.t5_config['d_model']
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            # warning removed as per cleanup request
            pass
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction=self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.DIFFRHYTHM:
            return DiffRhythmHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.getattr("n_classes_head1", 2),
                self.config.getattr("n_classes_head2", 2),
                self.config.getattr("head_dropout", 0.1),
                reduction=self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        model_config = T5Config.from_dict(config.t5_config)

        if config.getattr("randomly_initialize_backbone", False):
            transformer_backbone = T5Model(model_config)
        else:
            transformer_backbone = T5EncoderModel(model_config)

        transformer_backbone = transformer_backbone.get_encoder()

        if config.getattr("enable_FAN", False):
            if model_config.dense_act_fn == "gelu":
                for block in transformer_backbone.block:
                    MLPlayer = block.layer[1]
                    MLPlayer.DenseReluDense.wi = FANLayer(
                        input_dim=MLPlayer.DenseReluDense.wi.in_features,
                        output_dim=MLPlayer.DenseReluDense.wi.out_features,
                        with_gate=config.getattr("enable_FAN_gate", True)
                    )
                    MLPlayer.DenseReluDense.wo = FANLayer(
                        input_dim=MLPlayer.DenseReluDense.wo.in_features,
                        output_dim=MLPlayer.DenseReluDense.wo.out_features,
                        with_gate=config.getattr("enable_FAN_gate", True)
                    )
            elif model_config.dense_act_fn == "gelu_new":
                for block in transformer_backbone.block:
                    MLPlayer = block.layer[1]
                    MLPlayer.DenseReluDense.wi_0 = FANLayer(
                        input_dim=MLPlayer.DenseReluDense.wi_0.in_features,
                        output_dim=MLPlayer.DenseReluDense.wi_0.out_features,
                        with_gate=config.getattr("enable_FAN_gate", True)
                    )
                    MLPlayer.DenseReluDense.wi_1 = FANLayer(
                        input_dim=MLPlayer.DenseReluDense.wi_1.in_features,
                        output_dim=MLPlayer.DenseReluDense.wi_1.out_features,
                        with_gate=config.getattr("enable_FAN_gate", True)
                    )
                    MLPlayer.DenseReluDense.wo = FANLayer(
                        input_dim=MLPlayer.DenseReluDense.wo.in_features,
                        output_dim=MLPlayer.DenseReluDense.wo.out_features,
                        with_gate=config.getattr("enable_FAN_gate", True)
                    )

        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def classify(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        x_mark: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        enc_in = self.data_embedding(x_enc, mask=input_mask, x_mark=x_mark)
        enc_in = enc_in.reshape(
            (batch_size * n_channels, seq_len, self.config.d_model)
        )

        attention_mask = input_mask.repeat_interleave(n_channels, dim=0)

        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, seq_len, self.config.d_model))

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)
        elif reduction == "concat":
            enc_out = enc_out.permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.config.d_model * n_channels)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        logits = self.head(enc_out, input_mask=input_mask)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)

    def diffthyrhm_classify(
        self,
        *,
        x_enc1: torch.Tensor,
        input_mask1: torch.Tensor = None,
        x_mark1: torch.Tensor = None,
        
        x_enc2: torch.Tensor,
        input_mask2: torch.Tensor = None,
        x_mark2: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size1, n_channels1, seq_len1 = x_enc1.shape
        batch_size2, n_channels2, seq_len2 = x_enc2.shape

        if input_mask1 is None:
            input_mask1 = torch.ones((batch_size1, seq_len1)).to(x_enc1.device)
        if input_mask2 is None:
            input_mask2 = torch.ones((batch_size2, seq_len2)).to(x_enc2.device)

        x_enc1 = self.normalizer(x=x_enc1, mask=input_mask1, mode="norm")
        mean1 = self.normalizer.mean.mean(dim=1)
        stdev1 = self.normalizer.stdev.mean(dim=1)
        
        x_enc2 = self.normalizer(x=x_enc2, mask=input_mask2, mode="norm")
        mean2 = self.normalizer.mean.mean(dim=1)
        stdev2 = self.normalizer.stdev.mean(dim=1)
        
        feature = torch.cat((mean1, stdev1, mean2, stdev2), dim=1)

        x_enc1, x_enc2 = torch.nan_to_num(x_enc1, nan=0, posinf=0, neginf=0), torch.nan_to_num(x_enc2, nan=0, posinf=0, neginf=0)

        enc_in1 = self.data_embedding(x_enc1, mask=input_mask1, x_mark=x_mark1)
        enc_in2 = self.data_embedding(x_enc2, mask=input_mask2, x_mark=x_mark2)
        enc_in1 = enc_in1.reshape((batch_size1 * n_channels1, seq_len1, self.config.d_model))
        enc_in2 = enc_in2.reshape((batch_size2 * n_channels2, seq_len2, self.config.d_model))

        attention_mask1 = input_mask1.repeat_interleave(n_channels1, dim=0)
        attention_mask2 = input_mask2.repeat_interleave(n_channels2, dim=0)

        outputs1 = self.encoder(inputs_embeds=enc_in1, attention_mask=attention_mask1)
        outputs2 = self.encoder(inputs_embeds=enc_in2, attention_mask=attention_mask2)
        enc_out1 = outputs1.last_hidden_state
        enc_out2 = outputs2.last_hidden_state

        enc_out1 = enc_out1.reshape((-1, n_channels1, seq_len1, self.config.d_model))
        enc_out2 = enc_out2.reshape((-1, n_channels2, seq_len2, self.config.d_model))

        enc_out1 = enc_out1.mean(dim=1, keepdim=False)
        enc_out2 = enc_out2.mean(dim=1, keepdim=False)
        
        enc_out = torch.cat((enc_out1, enc_out2), dim=-1)

        combined_logits, head1_classes = self.head(enc_out, feature)
        
        metadata = {
            'reduction': reduction,
            'head1_classes': head1_classes,
            'head2_classes': combined_logits.shape[1] - head1_classes,
            'head1_logits_shape': (batch_size1, head1_classes),
            'head2_logits_shape': (batch_size1, combined_logits.shape[1] - head1_classes)
        }

        return TimeseriesOutputs(
            embeddings=enc_out, 
            logits=combined_logits,
            metadata=metadata,
            embeddings1=enc_out1,
            embeddings2=enc_out2
        )
    
    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        x_mark: torch.Tensor = None,

        x_enc2: torch.Tensor = None,
        input_mask2: torch.Tensor = None,
        x_mark2: torch.Tensor = None,

        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, x_mark=x_mark, **kwargs)
        elif self.task_name == TASKS.DIFFRHYTHM:
            return self.diffthyrhm_classify(x_enc1=x_enc, input_mask1=input_mask, x_mark1=x_mark,
                                            x_enc2=x_enc2, input_mask2=input_mask2, x_mark2=x_mark2, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class CIRCALLMPipeline(CIRCALLM, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.CLASSIFICATION
        )
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.CLASSIFICATION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.CLASSIFICATION)
        config = Namespace(**kwargs["model_kwargs"])
        
        if config.task_name == TASKS.CLASSIFICATION or config.task_name == TASKS.DIFFRHYTHM:
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")
    
    def init(self) -> None:
        if self.new_task_name != TASKS.CLASSIFICATION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)


def freeze_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model