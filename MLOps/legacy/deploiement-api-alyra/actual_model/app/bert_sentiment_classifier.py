import json
from torch import nn
from transformers import BertModel

config = {
    "BERT_MODEL": "bert-base-cased",
    "PRE_TRAINED_MODEL": "model/model_state_dict.bin",
    "CLASS_NAMES": ["negative", "neutral","positive"],
    "MAX_SEQUENCE_LEN": 160
}

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"],return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)