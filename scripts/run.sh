#!/bin/bash

# 环境变量设置
export PYTHONPATH=$PYTHONPATH:.

# 颜色定义
BLUE='\033[0;34m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

show_banner() {
    clear
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}        Transformer Midterm Project: Management System          ${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

run_parallel_ablation() {
    echo -e "${YELLOW}>>> 启动 4 组消融实验并行训练...${NC}"
    echo -n "请输入使用的 GPU 编号 (默认 0): "
    read gpu_id
    export CUDA_VISIBLE_DEVICES=${gpu_id:-0}

    # 实验矩阵: 实验名 n_heads n_layers use_pe
    experiments=(
        "base_h4_l2 4 2 1"
        "ablation_h1_l2 1 2 1"
        "ablation_h4_l1 4 1 1"
        "ablation_no_pe 4 2 0"
    )

    pids=()
    for exp in "${experiments[@]}"; do
        read -r name heads layers use_pe <<< "$exp"
        EXP_DIR="results/${name}"
        mkdir -p "$EXP_DIR"
        
        echo -e "${CYAN}[启动] ${name}${NC} -> 日志见 ${EXP_DIR}/train_process.log"
        
        ARGS="--exp_name $name --n_heads $heads --n_layers $layers --epochs 100 --batch_size 256"
        [ "$use_pe" == "0" ] && ARGS="$ARGS --no_pe"
        
        python src/train.py $ARGS > "${EXP_DIR}/train_process.log" 2>&1 &
        pids+=($!)
    done

    echo -e "${YELLOW}>>> 所有实验已转入后台。运行 'tail -f results/[实验名]/train_process.log' 监控。${NC}"
    for pid in "${pids[@]}"; do wait $pid; done
    echo -e "${GREEN}>>> 消融实验全部完成！${NC}"
}

interactive_translate() {
    echo -e "${YELLOW}>>> 请选择要加载的模型权重：${NC}"
    options=($(find results -maxdepth 1 -type d ! -name "results" ! -name "tokenizers" -exec basename {} \;))
    
    if [ ${#options[@]} -eq 0 ]; then
        echo -e "${RED}错误：未发现模型。${NC}"; return
    fi

    for i in "${!options[@]}"; do echo -e "  $i) ${options[$i]}"; done
    echo -ne "${YELLOW}请输入编号: ${NC}"
    read opt_idx

    selected_model=${options[$opt_idx]}

    echo -e "${GREEN}>>> 模型 [${selected_model}] 加载成功！${NC}"
    echo -e "${CYAN}提示：请输入您想翻译的英文句子。直接按 [回车] 将触发系统预设。输入 'q' 退出。${NC}"
    
    while true; do
        echo -ne "${BLUE}EN > ${NC}"
        read sentence
        if [ "$sentence" == "q" ]; then break; fi
        
        # 无论是否为空，都统一调用 translate.py
        # Python 内部会判断是否为空并随机选择
        python src/translate.py --exp "$selected_model" --text "$sentence"
    done
}

while true; do
    show_banner
    echo -e " 1) 并行消融实验 (Train Parallel)"
    echo -e " 2) 生成对比图与参数统计 (Plot & Stats)"
    echo -e " 3) 交互式翻译测试 (Translate)"
    echo -e " 4) 退出 (Exit)"
    echo -e "----------------------------------------------------------------"
    read -p "请选择操作 [1-4]: " main_opt

    case $main_opt in
        1) run_parallel_ablation ;;
        2) python -m src.plot_ablation ;;
        3) interactive_translate ;;
        4) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}无效选项，请重新输入${NC}" ; sleep 1 ;;
    esac
    echo -e "\n按任意键继续..."
    read -n 1
done