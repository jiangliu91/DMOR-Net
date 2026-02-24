import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import DMOREdge 

class EdgeDetectionLoss(nn.Module):
    def __init__(self):
        super(EdgeDetectionLoss, self).__init__()

    def forward(self, preds, targets):
        total_loss = 0.0
        for pred in preds:
            # 动态加权 BCE
            mask = targets > 0.5
            pos_num = mask.sum().float()
            neg_num = (targets <= 0.5).sum().float()
            total_num = pos_num + neg_num
            weight_pos = neg_num / (total_num + 1e-6)
            weight_neg = pos_num / (total_num + 1e-6)
            
            bce_loss = nn.functional.binary_cross_entropy(pred, targets, reduction='none')
            weighted_bce = torch.where(mask, weight_pos * bce_loss, weight_neg * bce_loss)
            
            # Ranking Loss 抑制冗余背景噪声 
            pos_preds = pred[mask]
            neg_preds = pred[~mask]
            rank_loss = 0.0
            if len(pos_preds) > 0 and len(neg_preds) > 0:
                rank_loss = torch.mean(torch.relu(neg_preds.view(-1, 1) - pos_preds.view(1, -1) + 0.3))
                
            total_loss += (weighted_bce.mean() + 0.5 * rank_loss)
        return total_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_nyud = (args.dataset == 'NYUDv2')
    
    model = DMOREdge(in_channels_rgb=3, in_channels_hha=3 if is_nyud else 0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = EdgeDetectionLoss()

    # 伪代码：根据 dataset 参数初始化 DataLoader
    train_loader = create_dataloader(args.dataset, args.batch_size, is_train=True)

    model.train()
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(train_loader):
            if is_nyud:
                img_rgb, img_hha, targets = data
                preds = model(img_rgb.to(device), img_hha.to(device))
            else:
                img_rgb, targets = data
                preds = model(img_rgb.to(device), None)
            
            loss = criterion(preds, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"dmor_edge_{args.dataset}_ep{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    train(parser.parse_args())