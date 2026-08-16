# namportfolio

アクティブ運用戦略の**事後的**パフォーマンス評価と可視化。チーム内・自分用の実験ツール。

機能一覧（F1〜F8）、入力規約、「作らないもの」の判断根拠は [docs/DESIGN.md](docs/DESIGN.md) にある。
設計を変える判断をしたら、そちらも更新すること。

## コマンド

```bash
uv run pytest -q         # テスト
uv run ruff format .     # 整形
uv run ruff check .      # lint
```

## 実装状況

| モジュール | 内容 | 状態 |
|---|---|---|
| `core/` | カラム名設定、頻度推定、long ⇄ wide 変換 | ✅ |
| `performance.py` | F4 リターン評価（絶対・相対・DD・期間集計・ローリング） | ✅ |
| `quantile.py` | F2 分位分析（分位リターン・IC・減衰・ターンオーバー） | ✅ |
| `stats.py` | F8 のうち t 値と Newey-West のみ | ⚠️ 部分 |
| `viz/` | `theme` / `performance` / `quantile` / `signals` / `holdings` | ✅ |
| `signals.py` | F1 シグナル前処理・診断（ファクター曝露は F6 待ち） | ⚠️ 部分 |
| `holdings.py` | F3 保有ベース分析（構造・特性・寄与度・売買） | ✅ |
| `attribution.py` | F5 Brinson 帰属 | 未着手 |
| `risk.py` | F6 Barra リスク分解 / F7 妥当性検証 | 未着手 |

全部作ってから使うのではなく、**実際に使うものから作る**。

## 設計の前提

### 実験ツールであること

分析条件（分位数、ホライズン、ユニバース、期間）が頻繁に変わる。変わるものは引数で振れる
ようにし、構造は薄く保つ。**抽象化レイヤを足さない** — ABC・Protocol・結果の基底クラスは
一度書いたうえで意図的に削除した。追加を提案する前に DESIGN.md §2.5「作らないもの」を読むこと。

### 入力データ

主たる形式は **long（tidy）**: `date` / `bid` カラム＋値カラムのフラットな DataFrame。

関数の入口で `core.panel.as_wide()` を呼べば long（カラム持ち／MultiIndex）と wide を吸収
できる。カラム名の既定は `core.config`（`set_columns()` で変更、関数引数で個別上書き）。
他モジュールから `DATE_COL` を直接 import せず、必ず `resolve_columns()` を経由する。

wide 化で空くセルは埋めない（ユニバース変動で穴が空くのは正常な状態）。

### API

- 計算は純粋関数、戻り値は素の DataFrame / Series
- `ReturnsLike = pd.Series | pd.DataFrame` — リターン系列の入力
- `Metric = float | pd.Series` — スカラー指標の戻り値（**入力が Series なら float、
  DataFrame なら列名を index に持つ Series**）
- 描画関数は Figure を返し `show()` しない。`ax` を渡せば既存の Axes に描く
- 分位分析のような long 入力の関数は、使うカラムを名前で指定する
  （`quantile_returns(df, factor="value", forward_return="ret_20d")`）

### 欠損の扱い

暗黙に補完しない。**例外は累積計算（`cumprod`）だけ**で、欠損はリターン 0 として扱う
（欠損 1 つで以降すべてが NaN になり、累積曲線が使い物にならないため）。
年率化の期間数は `len` ではなく `count()` で数える。

### 配色

用途で決まる。手で選ばない。

| 役割 | 使いどころ |
|---|---|
| categorical | 識別（複数戦略）。8 色を固定順、循環させない。9 色目は作らずエラー |
| ordinal | 順序（分位 Q1..Q5）。単一色相の濃淡 |
| sequential | 量（確率・連続値） |
| diverging | 極性（正負）。2 色相＋グレー中点 |

**分位を categorical で塗らない**（順序が色から読めなくなる）。縦軸を 2 本にしない。
色を差し替えるときは色覚特性（P型・D型）下での識別性を再検証すること。

## 環境の注意点

- **pandas 3.0** を使用。頻度文字列は `"ME"` / `"QE"` / `"YE"`（旧 `"M"` は削除済み）。
  `performance.MONTH_END` などの定数がバージョン差を吸収する。
- **`groupby.transform` にカスタム関数を渡さない。** pandas 3.0 では内部で
  `concat(ignore_index=True)` され index が保たれず例外になる。組み込み集約名
  （`transform("count")`）か `groupby.rank` を使う。
- **`DataFrame - Series` は列方向にブロードキャストされる。** 日付方向に引くときは
  `.sub(series, axis=0)` を明示する（一度これでバグを出している）。
- matplotlib / statsmodels / sklearn は optional 依存。`core` と計算モジュールは
  **pandas + numpy だけで動く**必要がある（社内の制限環境で使うため）。
- `npf.viz` は遅延 import。matplotlib が無くても `import namportfolio` は成功する。

## テスト

- 計算の正しさを**手計算値との突合**で確認する。設定・検証そのもののテストは書かない
- 図は「Figure が返るか」「系列数が合うか」を確認する。系列数は `ax.lines` ではなく
  `ax.get_legend_handles_labels()` で数える（ゼロ線が混ざるため）
- **図は PNG に出力して必ず目視する。** テストが通っても目盛やラベルは壊れている
