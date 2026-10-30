#!/usr/bin/env python3
"""杯柄形態 (Cup with Handle) 台股 / ETF 掃描 + 回測

安裝:  pip install yfinance pandas numpy lxml
用法:
  python cup_handle.py scan                       # 掃描全市場(上市/上櫃股票+ETF)
  python cup_handle.py scan 2330.TW 0050.TW       # 只掃指定標的
  python cup_handle.py backtest                   # 全市場回測
  python cup_handle.py backtest 2330.TW 0050.TW   # 指定標的回測
"""
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ───────────── 型態參數(依圖上數字設定,可自行調整) ─────────────
P = dict(
    lookback=160,        # 往回看幾根日K
    cup_min=30, cup_max=120,       # 杯身長度(K棒數)
    depth_min=0.12, depth_max=0.35,  # 杯深 12~35%(圖:約25%)
    rim_tol=0.05,        # 右緣與左峰高度差 ≤5%
    handle_min=4, handle_max=25,   # 柄長度
    handle_dd_min=0.03, handle_dd_max=0.15,  # 柄回落 3~15%(圖:約8%)
    prior_up=0.25, prior_days=60,  # 左峰前 60 日需先漲 ≥25%(上升趨勢中)
    vol_mult=1.5,        # 突破量 ≥ 50日均量 1.5 倍(圖:約3倍,可調高)
    chase_max=0.05,      # 收盤超過買點 5% 不追
    near_pct=0.03,       # 距買點 3% 內算「接近突破」
    min_price=10, min_vol=500_000,  # 流動性:股價≥10、20日均量≥500張
)

# ───────────── 停損停利設定 ─────────────
EXIT = dict(
    stop_buffer=0.99,    # 停損 = 柄低點 × 0.99
    stop_max=0.08,       # 最大虧損 8%(歐尼爾 7~8% 鐵律),取較緊者
    be_trigger=0.10,     # 獲利 ≥10% 後,停損上移到成本價
    tp=0.25,             # 停利 +25%(圖:突破後 +24%)
    trail_ma=20,         # 獲利 ≥10% 後,收盤跌破 20MA 出場
    max_hold=120,        # 最長持有天數
)


def trade_cost(is_etf):
    # 手續費 0.1425%×2(未折扣) + 證交稅(股票0.3% / ETF 0.1%)
    return 0.00285 + (0.001 if is_etf else 0.003)


# ───────────── 型態偵測(以第 t 根K為「今天」) ─────────────
def detect(h, l, c, v, t, p=P):
    s = max(0, t - p["lookback"])
    r0 = max(s, t - p["handle_max"] - 1)
    if t - r0 < p["handle_min"] + 1:
        return None
    R = r0 + int(np.argmax(h[r0:t]))          # 右緣(買點 pivot)
    if t - R < p["handle_min"]:
        return None
    pivot = h[R]

    lo = max(s, R - p["cup_max"])
    hi = R - p["cup_min"]
    if hi <= lo:
        return None
    L = lo + int(np.argmax(h[lo:hi + 1]))      # 左峰
    cup_len = R - L

    if h[L:R + 1].max() > max(h[L], h[R]) * 1.01:
        return None
    if abs(h[R] / h[L] - 1) > p["rim_tol"]:
        return None

    B = L + int(np.argmin(l[L:R + 1]))         # 杯底
    bottom = l[B]
    depth = 1 - bottom / h[L]
    if not p["depth_min"] <= depth <= p["depth_max"]:
        return None
    if not 0.25 <= (B - L) / cup_len <= 0.75:  # 圓弧底,排除 V 形
        return None

    hlow = l[R + 1:t + 1].min()                # 柄低點
    hdd = 1 - hlow / pivot
    if not p["handle_dd_min"] <= hdd <= p["handle_dd_max"]:
        return None
    if hlow < bottom + 0.5 * (h[L] - bottom):  # 柄需在杯身上半部
        return None
    if t - R > 1 and c[R + 1:t].max() > pivot:  # 之前已突破過就不算新訊號
        return None

    i0 = L - p["prior_days"]                   # 先前需有上升趨勢
    if i0 < 0 or c[L] / c[i0] - 1 < p["prior_up"]:
        return None

    va = v[max(0, t - 50):t].mean()
    vr = v[t] / va if va > 0 else 0.0
    if c[t] > pivot:
        if c[t] > pivot * (1 + p["chase_max"]):
            return None
        status = "突破" if vr >= p["vol_mult"] else "突破量不足"
    elif c[t] >= pivot * (1 - p["near_pct"]):
        status = "接近突破"
    else:
        return None

    return dict(status=status, pivot=pivot, handle_low=hlow,
                depth=depth, handle_dd=hdd, cup_len=cup_len, vol_ratio=vr)


# ───────────── 資料 ─────────────
def load_universe():
    """從證交所 ISIN 表抓上市/上櫃的股票與 ETF。回傳 {ticker: (名稱, 類型)}"""
    uni = {}
    for mode, suffix in ((2, ".TW"), (4, ".TWO")):
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        df = pd.read_html(url, encoding="ms950")[0]
        section = ""
        for _, row in df.iterrows():
            a = str(row.iloc[0])
            if a == str(row.iloc[1]):          # 分類標題列
                section = a
                continue
            if section not in ("股票", "ETF") or "\u3000" not in a:
                continue
            code, name = a.split("\u3000", 1)
            uni[code.strip() + suffix] = (name.strip(), section)
    return uni


def download(tickers, period="2y", batch=100):
    out = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        data = yf.download(chunk, period=period, group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
        for tk in chunk:
            try:
                d = data[tk] if len(chunk) > 1 else data
                d = d.dropna(subset=["Close"])
                if len(d) > 150:
                    out[tk] = d
            except KeyError:
                pass
        print(f"  已下載 {min(i + batch, len(tickers))}/{len(tickers)}")
    return out


def arrays(d):
    return (d["Open"].values, d["High"].values, d["Low"].values,
            d["Close"].values, d["Volume"].values.astype(float))


# ───────────── 掃描 ─────────────
def scan(tickers=None):
    uni = load_universe() if not tickers else {t: (t, "") for t in tickers}
    tickers = list(uni)
    print(f"標的數: {len(tickers)}")
    data = download(tickers)
    rows = []
    for tk, d in data.items():
        o, h, l, c, v = arrays(d)
        t = len(c) - 1
        if c[t] < P["min_price"] or v[-20:].mean() < P["min_vol"]:
            continue
        sig = detect(h, l, c, v, t)
        if not sig:
            continue
        stop = max(sig["handle_low"] * EXIT["stop_buffer"], c[t] * (1 - EXIT["stop_max"]))
        rows.append({
            "代號": tk, "名稱": uni[tk][0], "狀態": sig["status"],
            "收盤": round(c[t], 2), "買點": round(sig["pivot"], 2),
            "停損": round(stop, 2), "停利": round(c[t] * (1 + EXIT["tp"]), 2),
            "杯深%": round(sig["depth"] * 100, 1),
            "柄深%": round(sig["handle_dd"] * 100, 1),
            "量比": round(sig["vol_ratio"], 2),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        print("今天沒有符合的標的")
        return res
    res = res.sort_values(["狀態", "量比"], ascending=[True, False])
    res.to_csv("cup_handle_scan.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("\n已輸出 cup_handle_scan.csv")
    return res


# ───────────── 回測 ─────────────
def backtest_one(d, is_etf=False):
    o, h, l, c, v = arrays(d)
    n = len(c)
    ma = pd.Series(c).rolling(EXIT["trail_ma"]).mean().values
    cost = trade_cost(is_etf)
    trades = []
    t = P["prior_days"] + P["cup_min"] + P["handle_min"] + 5
    while t < n - 1:
        sig = detect(h, l, c, v, t)
        if not sig or sig["status"] != "突破":
            t += 1
            continue
        e = t + 1                                   # 隔日開盤進場
        entry = o[e]
        if entry > sig["pivot"] * (1 + P["chase_max"]):
            t += 1
            continue
        stop = max(sig["handle_low"] * EXIT["stop_buffer"], entry * (1 - EXIT["stop_max"]))
        tp_price = entry * (1 + EXIT["tp"])
        peak = entry
        exit_px, exit_j, why = c[-1], n - 1, "期末"
        for j in range(e, n):
            if o[j] <= stop:                         # 跳空跌破停損
                exit_px, exit_j, why = o[j], j, "停損"; break
            if l[j] <= stop:
                exit_px, exit_j, why = stop, j, "停損" if stop < entry else "保本"; break
            if h[j] >= tp_price:
                exit_px, exit_j, why = max(o[j], tp_price), j, "停利"; break
            peak = max(peak, h[j])
            if c[j] / entry - 1 >= EXIT["be_trigger"]:
                stop = max(stop, entry)              # 保本
                if not np.isnan(ma[j]) and c[j] < ma[j]:
                    exit_px, exit_j, why = c[j], j, "跌破20MA"; break
            if j - e >= EXIT["max_hold"]:
                exit_px, exit_j, why = c[j], j, "持有到期"; break
        ret = exit_px / entry - 1 - cost
        trades.append(dict(進場日=d.index[e], 出場日=d.index[exit_j],
                           進場價=round(entry, 2), 出場價=round(exit_px, 2),
                           報酬=ret, 原因=why, 天數=exit_j - e))
        t = exit_j + 1
    return trades


def backtest(tickers=None):
    uni = load_universe() if not tickers else {t: (t, "") for t in tickers}
    data = download(list(uni), period="10y")
    allt = []
    for tk, d in data.items():
        o, h, l, c, v = arrays(d)
        if v[-20:].mean() < P["min_vol"]:
            continue
        for tr in backtest_one(d, uni[tk][1] == "ETF"):
            tr["代號"] = tk
            allt.append(tr)
    if not allt:
        print("沒有交易訊號")
        return
    df = pd.DataFrame(allt)
    df.to_csv("cup_handle_trades.csv", index=False, encoding="utf-8-sig")
    w, lo = df[df["報酬"] > 0], df[df["報酬"] <= 0]
    pf = w["報酬"].sum() / abs(lo["報酬"].sum()) if len(lo) else float("inf")
    print(f"交易數 {len(df)} | 勝率 {len(w)/len(df):.1%} | 平均報酬 {df['報酬'].mean():.2%}")
    print(f"平均獲利 {w['報酬'].mean():.2%} | 平均虧損 {lo['報酬'].mean():.2%} | 獲利因子 {pf:.2f}")
    print(f"最大單筆虧損 {df['報酬'].min():.2%} | 平均持有 {df['天數'].mean():.0f} 天")
    print(df["原因"].value_counts().to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    args = sys.argv[2:] or None
    {"scan": scan, "backtest": backtest}[cmd](args)
