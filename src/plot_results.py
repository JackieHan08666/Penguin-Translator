import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_combined_results():
    # 设置学术绘图风格
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 6), dpi=150)
    
    results_dir = "results"
    # 获取所有实验文件夹
    exps = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    # 颜色和线型配置，确保区分度
    styles = {
        'base_h4_l2': {'color': '#1f77b4', 'ls': '-', 'label': 'Base (4-Head, 2-Layer)'},
        'ablation_h1_l2': {'color': '#ff7f0e', 'ls': '--', 'label': 'Ablation: 1-Head'},
        'ablation_h4_l1': {'color': '#2ca02c', 'ls': '-.', 'label': 'Ablation: 1-Layer'},
        'ablation_no_pe': {'color': '#d62728', 'ls': ':', 'label': 'Ablation: No Positional Encoding'}
    }

    print("正在汇总实验数据...")
    for exp in exps:
        csv_path = os.path.join(results_dir, exp, "train_log.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            config = styles.get(exp, {'color': None, 'ls': '-', 'label': exp})
            
            # 绘制验证集 Loss 曲线
            plt.plot(df['epoch'], df['val_loss'], 
                     color=config['color'], 
                     linestyle=config['ls'], 
                     label=config['label'], 
                     linewidth=2)
            print(f"OK: 已加载 {exp}")

    plt.title("Transformer Ablation Study: Validation Loss Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=12)
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存结果
    save_path = "results/ablation_comparison_plot.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"\n>>> 绘图完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    plot_combined_results()