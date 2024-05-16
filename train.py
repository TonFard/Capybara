
import os
import torch
import numpy as np



def get_batch(data_dir, split, block_size=1024, batch_size=32, device_type="cuda", device="cuda"):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([torch.from_numpy((data[i: i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out





