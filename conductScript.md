## A

python -m module_a_data.crawler.fetch_ohlcv --ticker AAPL --start 2021-01-01 --end 2025-12-31
python -m module_a_data.preprocess.chart_generator --ticker AAPL --limit 100
python -m module_a_data.build_dataset --ticker AAPL --limit 100

## B

python -m module_b_encoder.generate_vectors --fake --n 10        # 第一階段：不需要 A 的資料
python -m module_b_encoder.generate_vectors --ticker AAPL --limit 50   # 第二階段：讀 A 的真實 JSON

## C

python -m module_c_fusion.fusion.train --fake --n 32 --epochs 1     # 第一階段：假向量
python -m module_c_fusion.fusion.train --ticker AAPL                # 第二階段：真實向量
python -m module_c_fusion.validation.classifier --ticker AAPL
python -m module_c_fusion.backtest.backtest --ticker AAPL --strategy buy_and_hold

## pipeline

python scripts/run_pipeline.py --fake             # 驗證 B 產出的格式 C 讀不讀得進去
python scripts/run_pipeline.py --ticker AAPL --limit 50    # 真跑一次 A→B→C