# namportfolio

アクティブ運用戦略の**事後的**パフォーマンス評価と可視化。

シグナルの前処理から、分位分析・保有分析・リターン評価・Brinson 帰属・Barra リスク分解
までを、一貫した入力形式で扱う。

```python
import namportfolio as npf

# シグナルを整える
df["value_z"] = npf.signals.standardize(df, factor="value")
df["value_n"] = npf.signals.neutralize(df, factor="value_z", by=["sector", "log_mktcap"])

# 予測力を見る
q = npf.quantile.quantile_returns(df, factor="value_n", forward_return="fwd_ret_1m")
ic = npf.quantile.information_coefficient(df, factor="value_n", forward_return="fwd_ret_1m")
npf.viz.plot_quantile_cumulative(q)

# 勝因を分解する
summary = npf.attribution.brinson_summary(df, segment="sector", asset_return="ret_1m")
npf.viz.plot_waterfall(summary)
```

## 使い方

**[examples/quickstart.ipynb](examples/quickstart.ipynb)** に、サンプルデータの生成から
全機能の呼び出しまでを通した例がある。まずこれを実行するのが早い。

設計方針と「作らないもの」の判断は [docs/DESIGN.md](docs/DESIGN.md) にまとめてある。

## 機能

| モジュール | 内容 |
|---|---|
| `signals` | winsorize / 標準化 / 中立化、カバレッジ診断、シグナル間相関 |
| `quantile` | 分位ポートフォリオ、IC、減衰、ターンオーバー、遷移行列 |
| `holdings` | 集中度、セグメント配分、ポートフォリオ特性、寄与度、売買 |
| `performance` | 絶対・相対リターン指標、ドローダウン、期間集計、ローリング |
| `attribution` | Brinson-Fachler / BHB、多期間リンキング（Carino / GRAP / Frongello） |
| `risk` | Barra リスク分解（MCTR / CCTR）、リターン帰属、バイアス統計量 |
| `stats` | t 検定、Newey-West |
| `viz` | 上記すべての図（matplotlib） |

## 入力形式

すべての関数が **long 形式の DataFrame** を受け取り、使う列を名前で指定する。

```
   date        bid      value   ret_1m   sector   weight
0  2024-01-31  JP1301   1.24    0.012    建設     0.021
1  2024-01-31  JP1332  -0.31   -0.003    水産     0.008
```

カラム名が違う場合は既定を変えるか、関数ごとに上書きする。

```python
npf.set_columns(date="trade_date", id="barra_id")  # セッション開始時に一度
npf.quantile.quantile_returns(df, ..., date_col="dt")  # 個別に上書き
```

戻り値は素の DataFrame / Series。描画関数は Figure を返すだけで `show()` は呼ばない。

## インストール

```bash
uv sync --all-extras          # 開発時
pip install -e '.[viz]'       # 描画込みで利用
```

`pandas` / `numpy` があれば全計算が動く。`matplotlib` は描画のみに必要で、
`npf.viz` に触れるまで import されない（制限環境で本体だけ使える）。

## 開発

```bash
uv run pytest -q       # テスト
uv run ruff format .   # 整形
uv run ruff check .    # lint
```
