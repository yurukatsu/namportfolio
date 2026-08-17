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

## ノートブックでインタラクティブに見る

計算結果は素の DataFrame なので plotly にそのまま渡せる。見た目は
`npf.viz.plotly.apply_theme` で matplotlib 版と揃う。

```python
import plotly.express as px

cum = npf.performance.cumulative_returns(pd.DataFrame({"strategy": r, "benchmark": b}))
fig = px.line(cum)
npf.viz.plotly.apply_theme(fig, title="Cumulative return", percent_axis="y", zero_line=True)
```

分位のように順序があるものは `ordinal=True`、ヒートマップは
`npf.viz.plotly.diverging_scale()` を `px.imshow` に渡す。

図そのものは二重実装していない。**matplotlib 版はレポート・保存用、plotly は探索用**
という使い分け。

## インストール

ビルド用の隔離環境を作れない環境（社内プロキシ配下など）では
`--no-build-isolation` を付ける。

```bash
pip install . --no-build-isolation             # 利用のみ
pip install '.[viz]' --no-build-isolation      # 描画込み
pip install -e '.[dev]' --no-build-isolation   # 開発（テスト・lint 込み）
```

ビルドバックエンドは **setuptools**。`--no-build-isolation` ではインストール先の環境に
**既に入っているビルドツール**が使われるため、広く常駐している setuptools を選んでいる。

### 動作確認済みの構成

| | 開発環境 | 社内環境 |
|---|---|---|
| Python | 3.12 / 3.14 | 3.12 |
| pandas | 3.0.5 | **2.1.4** |
| numpy | 2.5.2 | 1.26.3 |
| matplotlib | 3.11.1 | 3.10.3 |

pandas は 2.1 から 3.0 までの差（頻度文字列の変更など）をパッケージ側で吸収している。
両方でテストが通ることを確認済み。

必須依存は `pandas` と `numpy` だけ。`matplotlib` は描画にのみ必要で、`npf.viz` に
触れるまで import されない。入っていない環境では次のようになる。

```python
>>> npf.performance.sharpe_ratio(returns)   # 動く
0.2099
>>> npf.viz.plot_cumulative_returns(returns)
NamPortfolioError: 描画には matplotlib が必要です。
```

uv が使える環境なら次でも同じ。

```bash
uv sync --all-extras
```

## 開発

```bash
python -m pytest -q       # テスト
python -m ruff format .   # 整形
python -m ruff check .    # lint（ノートブックのセルも検査される）
```
