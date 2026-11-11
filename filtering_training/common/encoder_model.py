"""용도: 기존 공동 Encoder와 관련도 전용 Encoder의 백본·풀링·분류 헤드를 함께 구성한다.
생성일: 2026-11-11
"""


def build_encoder(*, base, config=None, pretrained=True, head_sizes=(5, 5),
                  initialize_head_sizes=None, attention='eager', stack_outputs=False,
                  required_model_type=None):
    import torch
    from transformers import AutoConfig, AutoModel

    if not head_sizes or any(type(size) is not int or size < 2 for size in head_sizes):
        raise ValueError('classification heads require at least two classes')
    initial = head_sizes if initialize_head_sizes is None else initialize_head_sizes
    if len(initial) != len(head_sizes) or any(a < b for a, b in zip(initial, head_sizes)):
        raise ValueError('initial heads must cover final head sizes')
    if stack_outputs and len(set(head_sizes)) != 1:
        raise ValueError('stacked heads must have matching sizes')

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            model_config = config if config is not None else AutoConfig.from_pretrained(str(base), local_files_only=True)
            if required_model_type and model_config.model_type != required_model_type:
                raise ValueError('unexpected encoder model type')
            kwargs = {'attn_implementation': attention}
            if model_config.model_type == 'roberta':
                kwargs['add_pooling_layer'] = False
            self.encoder = (AutoModel.from_pretrained(str(base), local_files_only=True, **kwargs) if pretrained
                            else AutoModel.from_config(model_config, **kwargs))
            self.dropout = torch.nn.Dropout(.1)
            self.heads = torch.nn.ModuleList([torch.nn.Linear(model_config.hidden_size, size) for size in initial])
            for index, (old_size, size) in enumerate(zip(initial, head_sizes)):
                if old_size == size:
                    continue
                # Preserve the historical joint model's initialization and RNG state.
                old = self.heads[index]
                with torch.random.fork_rng(devices=[]):
                    replacement = torch.nn.Linear(model_config.hidden_size, size)
                with torch.no_grad():
                    replacement.weight.copy_(old.weight[:size])
                    replacement.bias.copy_(old.bias[:size])
                self.heads[index] = replacement

        def forward(self, input_ids, attention_mask):
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if self.encoder.config.model_type == 'modernbert':
                weights = attention_mask.unsqueeze(-1).to(states.dtype)
                features = (states * weights).sum(1) / weights.sum(1).clamp_min(1)
            else:
                features = states[:, 0]
            # One shared dropout mask across heads, matching archived models.
            features = self.dropout(features)
            outputs = tuple(head(features) for head in self.heads)
            return torch.stack(outputs, dim=1) if stack_outputs else outputs

    return Encoder()
