import torch
import torch.nn as nn
import time, os, argparse, csv, math
from tqdm import tqdm
from datasets import load_dataset
from src.model import Transformer
from src.dataset import TranslationDataset, get_or_build_tokenizer
from src.utils import get_std_opt

def get_lr(step, d_model, factor, warmup):
    """手动实现 Noam 学习率公式"""
    if step == 0: step = 1
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, label = batch['encoder_input'].to(device).long(), batch['decoder_input'].to(device).long(), batch['label'].to(device).long()
            src_mask, tgt_mask = batch['src_mask'].to(device), batch['tgt_mask'].to(device)
            with torch.amp.autocast('cuda'):
                logits = model(src, tgt, src_mask, tgt_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="base_h4_l2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_pe", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    # 初始化
    torch.manual_seed(42)
    exp_dir = f"results/{args.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    raw_ds = load_dataset("bentrevett/multi30k")
    t_en = get_or_build_tokenizer("en", raw_ds['train'])
    t_de = get_or_build_tokenizer("de", raw_ds['train'])
    
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(raw_ds['train'], "en", "de", t_en, t_de, 32), 
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        TranslationDataset(raw_ds['validation'], "en", "de", t_en, t_de, 32), 
        batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    model = Transformer(t_en.get_vocab_size(), t_de.get_vocab_size(), args.n_layers, args.d_model, 512, args.n_heads, use_pe=not args.no_pe).to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=t_de.token_to_id("[PAD]"), label_smoothing=0.1)
    
    # 获取原始 Adam
    optimizer = get_std_opt(model, args.d_model)
    scaler = torch.amp.GradScaler('cuda')

    # 日志记录
    log_file = open(f"{exp_dir}/train_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val_loss = float('inf')
    no_improve_count = 0
    global_step = 0 # 全局步数计数器

    print(f"┌──────────────────────────────────────────────────────────┐")
    print(f"│ START TRAINING: {args.exp_name:<41} │")
    print(f"└──────────────────────────────────────────────────────────┘")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}", leave=False)
        
        for batch in pbar:
            global_step += 1
            
            # 1. 手动计算并更新学习率
            lr = get_lr(global_step, args.d_model, factor=1.5, warmup=800)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # 2. 正常训练流程
            src, tgt, label = batch['encoder_input'].to(args.device), batch['decoder_input'].to(args.device), batch['label'].to(args.device)
            src_mask, tgt_mask = batch['src_mask'].to(args.device), batch['tgt_mask'].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(src, tgt, src_mask, tgt_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        # 3. 验证
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, loss_fn, args.device)
        
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{exp_dir}/model_best.pt")
            no_improve_count = 0
            tag = "[BEST]"
        else:
            no_improve_count += 1
            tag = f"[{no_improve_count}/{args.patience}]"
        
        print(f"Ep {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {lr:.2e} {tag}")
        log_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, lr])
        log_file.flush()

        if no_improve_count >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    log_file.close()

if __name__ == "__main__":
    main()