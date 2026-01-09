import torch
import os
from torch.utils.data import Dataset
from tokenizers import Tokenizer, decoders, models, trainers, pre_tokenizers

def get_or_build_tokenizer(lang, ds, vocab_size=5000):
    """构建或加载 ByteLevel BPE 分词器"""
    os.makedirs("results/tokenizers", exist_ok=True)
    tokenizer_path = f"results/tokenizers/tokenizer_{lang}.json"
    
    if os.path.exists(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)
    
    # 使用 ByteLevel BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # 预分词器：ByteLevel 能完美处理空格，不会报 Metaspace 的参数错误
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # 训练器：加入 ByteLevel 字母表
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    def get_training_corpus():
        for item in ds:
            yield item[lang]
            
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # 解码器：关键！确保生成的句子带正常空格
    tokenizer.decoder = decoders.ByteLevel()
    
    tokenizer.save(tokenizer_path)
    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, ds, lang_src, lang_tgt, tokenizer_src, tokenizer_tgt, max_len):
        super().__init__()
        self.ds = ds
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len
        self.pad_id = tokenizer_src.token_to_id("[PAD]")
        self.sos_id = tokenizer_src.token_to_id("[SOS]")
        self.eos_id = tokenizer_src.token_to_id("[EOS]")

    def __len__(self):
        # 修复点：必须返回数据集长度
        return len(self.ds)

    def make_tgt_mask(self, tgt):
        # 显式转换为 bool
        pad_mask = (tgt != self.pad_id).unsqueeze(0).bool() 
        size = tgt.size(-1)
        sub_mask = torch.triu(torch.ones((size, size)), diagonal=1).bool() == False
        # 返回 bool 类型的 mask
        return pad_mask & sub_mask

    def __getitem__(self, idx):
        item = self.ds[idx]
        
        enc_src = self.tokenizer_src.encode(item[self.lang_src]).ids[:self.max_len - 2]
        enc_tgt = self.tokenizer_tgt.encode(item[self.lang_tgt]).ids[:self.max_len - 2]

        # 确保所有拼接的 tensor 类型一致 (Long)
        enc_input = torch.cat([
            torch.tensor([self.sos_id], dtype=torch.long), 
            torch.tensor(enc_src, dtype=torch.long), 
            torch.tensor([self.eos_id], dtype=torch.long),
            torch.tensor([self.pad_id] * (self.max_len - len(enc_src) - 2), dtype=torch.long)
        ])
        
        dec_input = torch.cat([
            torch.tensor([self.sos_id], dtype=torch.long), 
            torch.tensor(enc_tgt, dtype=torch.long),
            torch.tensor([self.pad_id] * (self.max_len - len(enc_tgt) - 1), dtype=torch.long)
        ])
        
        label = torch.cat([
            torch.tensor(enc_tgt, dtype=torch.long), 
            torch.tensor([self.eos_id], dtype=torch.long),
            torch.tensor([self.pad_id] * (self.max_len - len(enc_tgt) - 1), dtype=torch.long)
        ])

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "label": label,
            # 显式转换为 bool，避免 collate 报错
            "src_mask": (enc_input != self.pad_id).unsqueeze(0).bool(),
            "tgt_mask": self.make_tgt_mask(dec_input).bool(),
        }