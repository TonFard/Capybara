import os
import requests
import numpy as np
import sentencepiece as spm
model_path = "tokenizer.model"

class CapybaraTokenizer:
    def __init__(self, model_path: str):
        self.spm_model = spm.SentencePieceProcessor(model_path)

    def encode(self, text):
        return self.spm_model.EncodeAsIds(text)

    def decode(self, ids):
        return self.spm_model.DecodeIds(ids)

input_file_path = "/home/gsq/tinyllm/train_100w.txt"

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]

val_data = data[int(n*0.9):]
tokenizer = CapybaraTokenizer(model_path="/home/gsq/tinyllm/tokenizer.model")
print(tokenizer.encode("这是一个测试的句子。"))

if __name__ == "__main__":
    pass
    # # encode with tiktoken gpt2 bpe
    # train_ids = tokenizer.encode(train_data)
    # val_ids = tokenizer.encode(val_data)
    # print(len(train_ids))
    # print(len(val_ids))

    # print(f"train has {len(train_ids):,} tokens")
    # print(f"val has {len(val_ids):,} tokens")

    # # export to bin files
    # train_ids = np.array(train_ids, dtype=np.uint16)
    # val_ids = np.array(val_ids, dtype=np.uint16)
    # train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    # val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
