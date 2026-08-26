"""評估微調完的 decoder：在 test set（訓練時完全沒看過的天）上驗證，方法依 2026 年
評估微調 LLM 的業界共識設計，不是只看一個數字。

為什麼要這樣設計（查證來源見 docs/decisions.md，跟 QLoRA 訓練設定同一次查證）：

    1. 一定要用 test set（train.py 時間序切分出來的最後 15%），不能用 train/val 的資料
       ——不然算出來的數字只是「背得熟不熟」，不是「有沒有學會類化」。
    2. 只看 loss/perplexity 不夠：loss 只反映「逐字猜對的機率」，不反映生成內容對不對。
       BLEU/ROUGE 這類字面比對指標，跟人類對開放式生成文字品質的判斷相關性很弱（換句話
       說意思一樣但字面重疊率低，分數會失真），2026 年的共識做法是搭配結構化準確率
       （這裡量得到的部分：TREND/RISK_LEVEL 標籤對不對）+ LLM-as-judge（量不到規則化
       對錯、但要看「內容有沒有講對」的敘述文字部分）。
    3. 一定要跟「微調前」的同一個 backbone 比較（同一份 test set、同一個 checkpoint
       之外的一切都一樣），只看微調後的絕對數字沒有意義——重點是微調有沒有造成差異
       （delta），不然沒辦法回答「這次微調到底有沒有用」。

本檔案做三件事：
    (1) 用 test set 算 loss，微調後 vs 完全沒微調的 backbone（backbone 相同、
        projector 沒訓練過），呈現微調造成的 loss 差異。
    (2) 對 test set 的一部分（--n_generate 天）實際生成文字（不是算 loss，是真的推論），
        解析生成內容裡的 <TREND>/<RISK_LEVEL> 標籤，跟正確答案比對，算準確率。
    (3)（可選，--llm_judge，會呼叫 LLM API 花錢）用 LLM 當評審，對敘述文字部分的內容
        品質評分 1~5 分，不是只看字面像不像。

用法：
    python -m module_c_fusion.decoder.evaluate --ticker AAPL --n_generate 30
    python -m module_c_fusion.decoder.evaluate --ticker AAPL --n_generate 30 --llm_judge --n_judge 15

前置：train.py 已經跑完，checkpoint 存在 data/outputs/checkpoints/decoder/。
本檔案需要 torch/transformers/peft，無法在沒有 GPU 的環境執行，只驗證過語法（py_compile）。
"""
import argparse
import re

import torch
from torch.utils.data import DataLoader

from shared import paths
from shared.utils import load_config, write_json
from module_c_fusion.decoder.train import load_paired_samples, time_split, YBeliefDataset, eval_loss
from module_c_fusion.decoder.model import ZFusedDecoder, load_base_only, load_finetuned_decoder

TAG_PATTERN = re.compile(r"<TREND>\s*\[(\w+)\]\s*<RISK_LEVEL>\s*\[(\w+)\]\s*(.*)", re.DOTALL)


def parse_tags(text: str):
    """從 y_belief 格式的文字解析出 (trend, risk_level, narrative)，解析不出來回傳 (None, None, "")。

    生成出來的文字不保證格式一定對（這本身就是評估的一部分——格式都學不會，代表微調
    沒有成功學到基本結構），解析失敗不能讓程式崩潰，要能算進「格式錯誤」這個結果裡。
    """
    m = TAG_PATTERN.search(text)
    if not m:
        return None, None, ""
    return m.group(1), m.group(2), m.group(3).strip()


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--n_prefix_tokens", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_generate", type=int, default=30,
                    help="test set 裡實際拿去生成文字、算標籤準確率的天數（生成比算 loss 慢很多，"
                         "不用全部 test set 都做，抽前 N 天即可）")
    ap.add_argument("--llm_judge", action="store_true",
                    help="開啟才會用 LLM API 評分敘述文字品質，會花錢，預設關閉")
    ap.add_argument("--n_judge", type=int, default=15,
                    help="--llm_judge 開啟時，實際送去給 LLM 評分的天數（比 --n_generate 更少，省成本）")
    ap.add_argument("--judge_provider", choices=["claude", "openai", "deepseek"], default="deepseek")
    args = ap.parse_args()

    samples = load_paired_samples(args.ticker)
    _, _, test_rows = time_split(samples)
    if not test_rows:
        raise SystemExit("test set 是空的，資料量不足以切出 15% 的 test，檢查 y_belief/Z_fused 天數")
    print(f"[decoder.evaluate] {args.ticker}: test set 共 {len(test_rows)} 天"
         f"（{test_rows[0][0]} ~ {test_rows[-1][0]}，訓練時完全沒看過）")

    checkpoint_dir = paths.OUTPUTS / "checkpoints" / "decoder"
    z_dim = cfg["fusion"]["z_dim"]

    # --- (1) test loss：微調後 vs 完全沒微調的 backbone ---
    print("[decoder.evaluate] 載入微調後的 decoder...")
    finetuned = load_finetuned_decoder(checkpoint_dir, args.base_model, z_dim, args.n_prefix_tokens)
    test_ds = YBeliefDataset(test_rows, finetuned.tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    finetuned_loss = eval_loss(finetuned, test_loader, finetuned.llm.device)
    print(f"[decoder.evaluate] 微調後 test loss = {finetuned_loss:.4f}")

    print("[decoder.evaluate] 載入完全沒微調的 backbone（對照組，projector 未訓練）...")
    base_tokenizer, base_llm = load_base_only(args.base_model)
    base_decoder = ZFusedDecoder(base_llm, base_tokenizer, z_dim=z_dim, n_prefix_tokens=args.n_prefix_tokens)
    base_ds = YBeliefDataset(test_rows, base_tokenizer, max_length=args.max_length)
    base_loader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False)
    base_loss = eval_loss(base_decoder, base_loader, base_llm.device)
    print(f"[decoder.evaluate] 微調前（對照組）test loss = {base_loss:.4f}")
    print(f"[decoder.evaluate] loss 差異（微調前 - 微調後）= {base_loss - finetuned_loss:.4f}"
         f"（正數代表微調確實有幫助，數字越大代表幫助越明顯）")

    # --- (2) 實際生成 + 結構化標籤準確率 ---
    n_gen = min(args.n_generate, len(test_rows))
    print(f"[decoder.evaluate] 對前 {n_gen} 天實際生成文字，比對 <TREND>/<RISK_LEVEL> 標籤...")
    gen_records = []
    trend_correct, risk_correct, format_ok = 0, 0, 0
    for date, z, gt_text in test_rows[:n_gen]:
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(finetuned.llm.device)
        generated = finetuned.generate(z_t, max_new_tokens=200)
        gt_trend, gt_risk, gt_narrative = parse_tags(gt_text)
        gen_trend, gen_risk, gen_narrative = parse_tags(generated)

        ok_format = gen_trend is not None
        ok_trend = ok_format and gen_trend == gt_trend
        ok_risk = ok_format and gen_risk == gt_risk
        format_ok += int(ok_format)
        trend_correct += int(ok_trend)
        risk_correct += int(ok_risk)

        gen_records.append({
            "date": date, "ground_truth": gt_text, "generated": generated,
            "gt_trend": gt_trend, "gen_trend": gen_trend, "trend_match": ok_trend,
            "gt_risk_level": gt_risk, "gen_risk_level": gen_risk, "risk_match": ok_risk,
            "gt_narrative": gt_narrative, "gen_narrative": gen_narrative,
        })
        print(f"  {date}: trend {gt_trend}->{gen_trend} {'OK' if ok_trend else 'X'} | "
             f"risk {gt_risk}->{gen_risk} {'OK' if ok_risk else 'X'}")

    print(f"[decoder.evaluate] 格式正確率={format_ok}/{n_gen}, "
         f"TREND 準確率={trend_correct}/{n_gen}, RISK_LEVEL 準確率={risk_correct}/{n_gen}")

    # --- (3)（可選）LLM-as-judge 評分敘述文字品質 ---
    judge_results = None
    if args.llm_judge:
        from module_b_encoder.llm_client import CALLERS, DEFAULT_MODEL
        from module_c_fusion.decoder.judge_prompt import SYSTEM_PROMPT, build_user_prompt
        import json

        n_judge = min(args.n_judge, len(gen_records))
        print(f"[decoder.evaluate] 用 {args.judge_provider} 評分前 {n_judge} 筆敘述文字品質...")
        model_name = DEFAULT_MODEL[args.judge_provider]
        scores = []
        for rec in gen_records[:n_judge]:
            user_prompt = build_user_prompt(rec["gt_narrative"], rec["gen_narrative"])
            raw = CALLERS[args.judge_provider](SYSTEM_PROMPT, user_prompt, model_name)
            try:
                start, end = raw.find("{"), raw.rfind("}")
                obj = json.loads(raw[start:end + 1])
                score, reason = obj.get("score"), obj.get("reason", "")
            except (json.JSONDecodeError, ValueError):
                score, reason = None, "解析失敗"
            scores.append({"date": rec["date"], "score": score, "reason": reason})
            print(f"  {rec['date']}: score={score} ({reason})")

        valid = [s["score"] for s in scores if isinstance(s["score"], (int, float))]
        judge_results = {
            "provider": args.judge_provider, "model": model_name,
            "n_judged": len(scores), "n_valid": len(valid),
            "avg_score": sum(valid) / len(valid) if valid else None,
            "details": scores,
        }
        print(f"[decoder.evaluate] LLM-judge 平均分數（1~5）= {judge_results['avg_score']}"
             f"（{len(valid)}/{len(scores)} 筆成功評分）")

    # --- 彙整報告 ---
    report = {
        "ticker": args.ticker,
        "n_test": len(test_rows),
        "test_period": [test_rows[0][0], test_rows[-1][0]],
        "loss": {"finetuned": finetuned_loss, "base_untuned": base_loss,
                "delta": base_loss - finetuned_loss},
        "structured_tag_accuracy": {
            "n_evaluated": n_gen,
            "format_ok_rate": format_ok / n_gen,
            "trend_accuracy": trend_correct / n_gen,
            "risk_level_accuracy": risk_correct / n_gen,
        },
        "llm_judge": judge_results,
        "generation_samples": gen_records,
    }
    out_path = paths.OUTPUTS / "metrics" / "decoder_evaluation_report.json"
    write_json(report, out_path)
    print(f"[decoder.evaluate] 完整評估報告 -> {out_path}")


if __name__ == "__main__":
    main()
