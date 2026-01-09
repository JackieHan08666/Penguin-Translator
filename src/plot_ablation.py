import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from src.model import Transformer
from src.dataset import get_or_build_tokenizer
from datasets import load_dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_and_stats():
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    
    # 实验配置
    configs = {
        "base_h4_l2":    {"label": "Base (4H-2L)",    "color": "#1f77b4", "ls": "-",  "marker": "o"},
        "ablation_h1_l2": {"label": "Ablation: 1-Head", "color": "#ff7f0e", "ls": "--", "marker": "s"},
        "ablation_h4_l1": {"label": "Ablation: 1-Layer","color": "#2ca02c", "ls": "-.", "marker": "^"},
        "ablation_no_pe": {"label": "Ablation: No PE",  "color": "#d62728", "ls": ":",  "marker": "x"}
    }

    results_dir = "results"
    stats_data = []

    # 加载分词器用于初始化模型
    raw_ds = load_dataset("bentrevett/multi30k")
    t_en = get_or_build_tokenizer("en", raw_ds['train'])
    t_de = get_or_build_tokenizer("de", raw_ds['train'])

    print(f"\n{'Experiment':<20} | {'Params':<12} | {'Best Val Loss':<15}")
    print("-" * 55)

    all_val_losses = []

    for folder, cfg in configs.items():
        csv_path = os.path.join(results_dir, folder, "train_log.csv")
        pt_path = os.path.join(results_dir, folder, "model_best.pt")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 抽样显示标记，避免太密集
            ax.plot(df['epoch'], df['val_loss'], 
                    label=cfg['label'], color=cfg['color'], 
                    linestyle=cfg['ls'], marker=cfg['marker'], 
                    markevery=5, markersize=5, linewidth=1.5)
            
            best_loss = df['val_loss'].min()
            all_val_losses.extend(df['val_loss'].tolist())

            # 统计参数
            n_heads = 1 if "h1" in folder else 4
            n_layers = 1 if "l1" in folder else 2
            use_pe = False if "no_pe" in folder else True
            
            model = Transformer(t_en.get_vocab_size(), t_de.get_vocab_size(), 
                                n_layers=n_layers, n_heads=n_heads, use_pe=use_pe)
            params = count_parameters(model)
            
            print(f"{folder:<20} | {params:<12,} | {best_loss:<15.4f}")

    # 动态优化坐标轴
    # 过滤掉 No PE 组可能产生的极大值，以便观察细节
    valid_losses = [l for l in all_val_losses if l < 6.0]
    if valid_losses:
        y_min = min(valid_losses) * 0.95
        y_max = max(valid_losses) * 1.05
        ax.set_ylim(y_min, y_max)

    ax.set_title("Transformer Ablation: Validation Loss Detail", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/ablation_comparison.png")
    print(f"\n>>> 优化后的对比图已保存至: results/ablation_comparison.png")

if __name__ == "__main__":
    plot_and_stats()