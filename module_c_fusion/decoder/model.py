"""Decoder：Z_fused 當 soft-prompt 前綴，加上 RAG-2 候選文字，接凍結的 LLaMA-2 + LoRA
（計畫書 3.4/3.5；decisions.md #113、#117、#118）。

    context = [ 4 個 prefix ; RAG-2 候選短版文字（15–20 段）; y_belief ]

- ZFusedProjector：Z_fused [B,768] -> 4 個 LLM prefix token。
- ZFusedDecoder.forward：只對 y_belief 的 token 算 L_belief = −Σ log p(y_t | y_<t, prefix, 候選)；
  prefix 與候選文字的 label 都是 -100（條件輸入，不是預測目標）。只生成一次：Decoder 看 Z_fused 的
  同時看候選，由注意力自己挑要參考哪幾段。
- 不加 RAG-2 的 512 版本（選配，可在 Mac 跑）就是候選文字為空。
- Z_fused 由融合層即時算出（decoder/train.py 聯合訓練），L_belief 的梯度會回到融合層。
- load_tiny_random：用本機快取的 LLaMA-2 tokenizer 建一個隨機初始化的小 LLaMA（不用下載、不用 GPU），
  只用來 smoke test 整條聯合訓練流程。
"""
import torch
import torch.nn as nn


class ZFusedProjector(nn.Module):
    """Z_fused [B, z_dim] -> soft-prompt 前綴 [B, n_prefix_tokens, hidden_size]。"""

    def __init__(self, z_dim: int, hidden_size: int, n_prefix_tokens: int = 4):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.hidden_size = hidden_size
        self.proj = nn.Linear(z_dim, n_prefix_tokens * hidden_size)

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        B = z_fused.size(0)
        out = self.proj(z_fused)
        return out.view(B, self.n_prefix_tokens, self.hidden_size)


def load_backbone(base_model_name: str = "meta-llama/Llama-2-7b-hf",
                  lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """4-bit 量化載入 LLaMA-2 + 掛 LoRA adapter，backbone 凍結。

    前置：本機已 `hf auth login`，且該帳號已通過 meta-llama/Llama-2-7b-hf 的存取申請。
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA-2 沒有內建 pad token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map={"": 0},
        attn_implementation="sdpa",  # 明確指定用 PyTorch 內建的 scaled-dot-product-attention，
        # 不用另外裝 flash-attn（裝的過程容易在 Windows 上出包），比預設可能退回的 eager
        # attention 快、且省記憶體，是內建在 PyTorch 裡的功能，沒有額外依賴風險
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        # 原本只掛 q_proj/v_proj（LoRA 原始論文的做法），2026 社群/QLoRA 論文後續的共識是
        # 全部 linear 層（含 MLP 的 gate/up/down_proj）效果明顯更好（下游任務約 +1~3 個百分點），
        # 額外的可訓練參數量增加不多（約 0.2% -> 1.5%），VRAM 負擔也不大，所以改成全掛
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


def load_backbone_for_resume(checkpoint_dir, base_model_name: str = "meta-llama/Llama-2-7b-hf"):
    """跟 load_backbone() 幾乎一樣（4-bit 量化 + prepare_model_for_kbit_training），差別是
    LoRA adapter 不是重新隨機初始化，是讀 checkpoint_dir/lora_adapter 裡上次訓練存的權重
    接著練——train.py 的 `--resume` 用這個，不是從頭 get_peft_model()。

    `is_trainable=True` 是關鍵：PeftModel.from_pretrained() 預設載入的 adapter 是不可訓練
    的（給推論用，梯度不會流過去），要接著訓練一定要明確指定，否則 loss.backward() 後
    optimizer.step() 完全不會更新到任何參數，訓練會「看起來正常跑、實際上甚麼都沒學到」
    ——這是 2026-08 WebSearch 查證過的 PEFT 官方用法，不是憑印象猜的。
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, checkpoint_dir / "lora_adapter", is_trainable=True)
    model.print_trainable_parameters()
    return tokenizer, model


def load_base_only(base_model_name: str = "meta-llama/Llama-2-7b-hf"):
    """4-bit 量化載入 LLaMA-2，完全不掛 LoRA——給評估腳本當「微調前」的對照組用
    （見 evaluate.py，比較微調前後的差異，而不是只看微調後的絕對數字）。
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map={"": 0},
    )
    # 跟 load_backbone() 用同一個處理：把 lm_head 等關鍵層轉成 float32 做數值穩定，
    # 順便讓 dtype 跟我們手動組的 inputs_embeds（float32，見 ZFusedDecoder.forward）
    # 保持一致——不呼叫這個的話 lm_head 會停在原本讀進來的 fp16，跟注入的 float32
    # embeds 相乘時會報 dtype 不一致的錯（mat1 float != mat2 Half）。
    # 這裡不會真的拿去訓練（eval 都在 no_grad 底下跑），只是借用它的 dtype 處理。
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def load_finetuned_decoder(checkpoint_dir, base_model_name: str = "meta-llama/Llama-2-7b-hf",
                           z_dim: int = 768, n_prefix_tokens: int = 4):
    """讀 train.py 存下來的 checkpoint（projector.pt + lora_adapter/），組回一個可以用的
    ZFusedDecoder，給 evaluate.py 用。
    """
    from peft import PeftModel

    tokenizer, base_model = load_base_only(base_model_name)
    llm = PeftModel.from_pretrained(base_model, checkpoint_dir / "lora_adapter")
    decoder = ZFusedDecoder(llm, tokenizer, z_dim=z_dim, n_prefix_tokens=n_prefix_tokens)
    decoder.projector.load_state_dict(torch.load(checkpoint_dir / "projector.pt", map_location=llm.device))
    decoder.eval()
    return decoder


class ZFusedDecoder(nn.Module):
    """把 ZFusedProjector 的輸出接到 LLM 的 input embeddings 前面，forward 直接回傳 L_belief。"""

    def __init__(self, llm, tokenizer, z_dim: int, n_prefix_tokens: int = 4):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.n_prefix_tokens = n_prefix_tokens
        hidden_size = llm.config.hidden_size
        self.projector = ZFusedProjector(z_dim, hidden_size, n_prefix_tokens).to(llm.device)

    def forward(self, z_fused: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """z_fused [B,z_dim]；input_ids = 候選文字＋y_belief 串好、右側 padding 的 token。
        labels：候選文字與 padding 位置為 -100、y_belief 位置為原 token（decoder/train.py 的 collate 組好）；
        不給時視為沒有候選文字，整段 input_ids（padding 除外）都是 y_belief。回傳 L_belief。
        """
        prefix_embeds = self.projector(z_fused)                          # [B, P, H]
        token_embeds = self.llm.get_input_embeddings()(input_ids)        # [B, L, H]
        inputs_embeds = torch.cat([prefix_embeds.to(token_embeds.dtype), token_embeds], dim=1)
        prefix_mask = torch.ones(z_fused.size(0), self.n_prefix_tokens,
                                 device=attention_mask.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # padding 一定要蓋成 -100：pad_token 跟 eos 相同，不蓋掉的話模型會學到「盡快輸出 eos」，
        # 生成時第一步就結束（舊版遇過的空字串問題）。
        if labels is None:
            labels = input_ids.masked_fill(attention_mask == 0, -100)
        prefix_labels = torch.full((z_fused.size(0), self.n_prefix_tokens), -100,
                                   device=input_ids.device, dtype=labels.dtype)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask,
                       labels=torch.cat([prefix_labels, labels], dim=1))
        return out.loss

    @torch.no_grad()
    def generate(self, z_fused: torch.Tensor, context_ids: torch.Tensor | None = None,
                 max_new_tokens: int = 200) -> str:
        """單筆 Z_fused [1,z_dim]（＋候選文字的 token [1,Lc]）-> greedy 生成的文字（可重現）。
        方法內強制切 eval：LoRA dropout 在生成時不可啟用。"""
        self.eval()
        embeds = self.projector(z_fused)                                  # [1, P, H]
        if context_ids is not None and context_ids.numel():
            ctx = self.llm.get_input_embeddings()(context_ids)
            embeds = torch.cat([embeds.to(ctx.dtype), ctx], dim=1)
        mask = torch.ones(1, embeds.size(1), device=z_fused.device, dtype=torch.long)
        out_ids = self.llm.generate(
            inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)


def load_tiny_random(base_model_name: str = "meta-llama/Llama-2-7b-hf", lora_r: int = 4):
    """smoke test 用：LLaMA-2 tokenizer（本機快取）＋隨機初始化的 2 層小 LLaMA ＋ LoRA，CPU 就跑得動。"""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=4096)
    model = LlamaForCausalLM(config)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    return tokenizer, model
