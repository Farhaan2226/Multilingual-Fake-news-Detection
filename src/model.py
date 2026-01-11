import torch.nn as nn
from transformers import AutoModel


class XLMRFakeNewsClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        return self.classifier(cls_output)
