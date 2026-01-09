import torch
import sys
import os
import argparse
import random  # 导入随机库
from tokenizers import Tokenizer
from datasets import load_dataset

sys.path.append(os.getcwd())
from src.model import Transformer
from src.dataset import get_or_build_tokenizer

def greedy_decode(model, src, src_mask, max_len, sos_id, eos_id, device):
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            src_emb = model.src_embed(src)
            memory = model.pe(src_emb) if model.use_pe else src_emb
            for layer in model.encoder_layers:
                memory = layer(memory, src_mask)
            memory = model.encoder_norm(memory)

    ys = torch.ones(1, 1).fill_(sos_id).long().to(device)
    for i in range(max_len - 1):
        size = ys.size(1)
        tgt_mask = (torch.triu(torch.ones((size, size), device=device), diagonal=1).bool() == False).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out_emb = model.tgt_embed(ys)
                out = model.pe(out_emb) if model.use_pe else out_emb
                for layer in model.decoder_layers:
                    out = layer(out, memory, src_mask, tgt_mask)
                prob = model.output_linear(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).long().to(device)], dim=1)
        if next_word == eos_id:
            break
    return ys

def translate_sentence(sentence, model, t_en, t_de, device):
    tokens = t_en.encode(sentence).ids
    src_ids = [t_en.token_to_id("[SOS]")] + tokens + [t_en.token_to_id("[EOS]")]
    src = torch.tensor(src_ids).unsqueeze(0).to(device).long()
    src_mask = (src != t_en.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).to(device)
    out_ids = greedy_decode(model, src, src_mask, 64, t_de.token_to_id("[SOS]"), t_de.token_to_id("[EOS]"), device)
    output_ids = out_ids[0].tolist()
    special_ids = {t_de.token_to_id(s) for s in ["[SOS]", "[EOS]", "[PAD]"]}
    filtered_ids = [id for id in output_ids if id not in special_ids]
    return t_de.decode(filtered_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_ds = load_dataset("bentrevett/multi30k")
    t_en = get_or_build_tokenizer("en", raw_ds['train'])
    t_de = get_or_build_tokenizer("de", raw_ds['train'])
    
    use_pe = False if "no_pe" in args.exp else True
    n_heads = 1 if "h1" in args.exp else 4
    n_layers = 1 if "l1" in args.exp else 2
    
    model = Transformer(t_en.get_vocab_size(), t_de.get_vocab_size(), 
                        n_layers=n_layers, d_model=128, d_ff=512, 
                        n_heads=n_heads, use_pe=use_pe).to(device)
    model.load_state_dict(torch.load(f"results/{args.exp}/model_best.pt", map_location=device))

    # 彩蛋例句列表
    easter_eggs = [
        "A small penguin is looking for its love.",
        "I love you more than all the fish in the ocean.",
        "Two penguins are standing on the blue ice.",
        "A group of penguins are dancing together.",
        "The heart of a penguin belongs to its partner.",
        "A penguin and a polar bear are friends.",
        "The snow is falling on the little penguin."
    ]

    # 如果输入为空，随机选一个
    input_text = args.text if args.text.strip() else random.choice(easter_eggs)
    
    # 如果是自动触发的，多打印一行提示
    if not args.text.strip():
        print(f"\033[0;35m[Surprise!] EN: {input_text}\033[0m")
    
    res = translate_sentence(input_text, model, t_en, t_de, device)
    print(f"DE: {res}")

if __name__ == "__main__":
    main()