import torch
from torch.utils.data import Dataset, DataLoader

from models.net import DMOREdgeNet


class DummyEdgeDataset(Dataset):
    """最小可训练闭环：随机图像 + 稀疏边缘 mask"""
    def __len__(self):
        return 128

    def __getitem__(self, idx):
        x = torch.randn(3, 128, 128)
        y = (torch.rand(1, 128, 128) > 0.9).float()
        return x, y


def balanced_bce_with_logits(logits, gt):
    # logits: [B,1,H,W], gt: [B,1,H,W]
    pred = torch.sigmoid(logits)
    pos = gt.sum().clamp_min(1.0)
    neg = (1 - gt).sum().clamp_min(1.0)
    beta = neg / (pos + neg)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = -beta * (1 - gt) * torch.log(1 - pred) - (1 - beta) * gt * torch.log(pred)
    return loss.mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    ds = DummyEdgeDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    # proposal 对齐：先关 topk 跑通，再做 topk 消融
    model = DMOREdgeNet(channels=32, topk=0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for it, (x, y) in enumerate(dl, start=1):
        x, y = x.to(device), y.to(device)

        logits = model(x)  # [B,1,H,W]
        loss = balanced_bce_with_logits(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 10 == 0:
            print(f"iter {it:03d} | loss {loss.item():.4f}")

        if it >= 50:
            break

    print("✅ minimal end-to-end training finished")


if __name__ == "__main__":
    main()
