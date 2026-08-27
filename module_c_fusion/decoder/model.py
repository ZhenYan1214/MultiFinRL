"""Decoder 架構：Z_fused 當 soft-prompt 前綴，接上凍結的 LLaMA-2 backbone + LoRA adapters
（計畫書 3.4/3.5 節）。

- ZFusedProjector：把 Z_fused（z_dim 維）投影成 n_prefix_tokens 個 soft-prompt embedding，
  維度對齊 LLM 的 hidden_size，串接在文字 token embedding 前面，讓生成過程「條件於
  Z_fused」，對應計畫書公式 p(Y|Ht,Hv,Hr) = ∏p(yi|y<i, Zfused)（Z_fused 已經是
  Ht/Hv/Hr 融合後的結果，decoder 只需要吃 Z_fused，不用重新接觸原始三個模態，
  這是既定的 decoupled architecture，見 docs/decisions.md）。
- load_backbone：4-bit 量化載入 LLaMA-2，掛 LoRA adapter（q_proj/v_proj），backbone
  其餘參數凍結，只有 adapter 會被訓練（計畫書 3.5 節 QLoRA 的定義：「updating only the
  projection layers and specific attention adapters while freezing the backbone MLLM」）。
- ZFusedDecoder.forward：算 L_belief = -Σlog p_θ(y_belief,t | y_belief<t, Z_fused)，
  用標準 causal LM cross-entropy 實作，prefix 位置的 label 設 -100，不計入 loss
  （prefix 是條件輸入，不是要被預測的目標）。

只實作 L_belief 一項 loss。L_align 需要 H_v/H_t 聯合訓練（目前 encoder 是凍結的，
架構上還沒開放）；L_ground 需要一份目前也不存在的 oracle relevance scores 標記。
這兩項先不做，範圍見 train.py 檔頭說明與 docs/decisions.md。

本檔案需要 torch/transformers/peft/bitsandbytes（見 requirements.txt Module C 區塊，
已經列在裡面），且 load_backbone() 需要 GPU 與 HuggingFace 登入權限，無法在沒有 GPU 的
環境執行，只能靠語法檢查（py_compile）驗證，實際跑通需要在你自己機器上執行。
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
               attention_mask: torch.Tensor) -> torch.Tensor:
        """z_fused [B,z_dim]，input_ids/attention_mask 是 y_belief 文字 tokenize 後的結果。
        回傳 L_belief（純量 loss）。
        """
        prefix_embeds = self.projector(z_fused)                          # [B, P, H]
        token_embeds = self.llm.get_input_embeddings()(input_ids)        # [B, L, H]
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [B, P+L, H]

        prefix_mask = torch.ones(z_fused.size(0), self.n_prefix_tokens,
                                 device=attention_mask.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # prefix 是條件輸入不是預測目標，label 設 -100 讓 HF 內建的 cross-entropy 跳過這段。
        # padding 位置也要蓋成 -100：tokenizer.pad_token 跟 eos_token 是同一個 token，
        # 如果不蓋掉，padding 區段會變成「看到 eos 預測下一個還是 eos」這種 trivial、
        # loss 趨近於 0 的規律，訓練資料裡敘述文字通常遠短於 max_length，padding 佔比很大，
        # 不蓋掉的話模型會被大量獎勵「盡快輸出 eos」，最後在生成時（沒有 teacher forcing
        # 硬塞正確 token）直接在第一步就選擇輸出 eos，生成空字串——這正是目前遇到的問題。
        content_labels = input_ids.masked_fill(attention_mask == 0, -100)
        prefix_labels = torch.full((z_fused.size(0), self.n_prefix_tokens), -100,
                                   device=input_ids.device, dtype=input_ids.dtype)
        labels = torch.cat([prefix_labels, content_labels], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, z_fused: torch.Tensor, max_new_tokens: int = 200) -> str:
        """給單一一筆 Z_fused（[1, z_dim]），實際生成一段文字（不是算 loss，是真的推論）。
        用 greedy decoding（do_sample=False）而不是隨機抽樣，確保評估時同一份輸入每次
        生成結果都一樣，可重現、可比較（評估用途本來就該用確定性生成，不是創作用途）。

        `@torch.no_grad()` 只關掉梯度計算，不會自動把 module 切成 eval 模式——
        `torch.no_grad()` 跟 `.eval()` 是兩件獨立的事。LoRA 掛了 `lora_dropout=0.05`，
        如果呼叫端忘記在呼叫前切 `.eval()`（例如呼叫完 `train.py` 的 `eval_loss()` 之後，
        它結束時會自動切回 `.train()`），dropout 在生成時還是啟用的，會讓 greedy decoding
        變得不確定、生成品質也會變差。這裡直接在方法內部強制切一次，不依賴呼叫端記得做。
        """
        self.eval()
        prefix_embeds = self.projector(z_fused)  # [1, P, H]
        prefix_mask = torch.ones(1, self.n_prefix_tokens, device=z_fused.device, dtype=torch.long)
        out_ids = self.llm.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
