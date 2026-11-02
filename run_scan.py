#!/usr/bin/env python3
"""每日收盤後掃描杯柄形態,結果推送到 Telegram。
環境變數: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, MARKET (TW / HK / US / ALL,預設 TW)
"""
import os
import sys
import json
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone

import yfinance as yf

import cup_handle

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# 判斷該市場今天是否有開盤用的參考標的 與 當地時區(UTC 偏移)
REF = {"TW": ("0050.TW", 8), "HK": ("2800.HK", 8), "US": ("SPY", -5)}


def send(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    chunks.append(cur)
    for ch in chunks:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data=json.dumps({"chat_id": CHAT_ID, "text": ch}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30).read()


def is_trading_day(market):
    """用參考標的最後一根日K日期判斷今天該市場有沒有開盤(假日就不推)。"""
    ref, off = REF[market]
    today = (datetime.now(timezone.utc) + timedelta(hours=off)).date()
    d = yf.download(ref, period="5d", progress=False, auto_adjust=True)
    return (not d.empty) and d.index[-1].date() == today


def fmt(df, today, market="TW"):
    title = f"📊 杯柄掃描 {cup_handle.MARKET_CFG[market]['name']} {today}"
    if df.empty:
        return f"{title}\n今天沒有符合的標的"
    out = [title]
    for status, icon in (("突破", "🚀"), ("突破量不足", "⚠️"), ("接近突破", "👀")):
        sub = df[df["狀態"] == status]
        if sub.empty:
            continue
        out.append(f"\n{icon} {status}({len(sub)})")
        for _, r in sub.iterrows():
            code = r["代號"].split(".")[0]
            name = str(r["名稱"])[:16]
            out.append(
                f"{code} {name} 收{r['收盤']} 買點{r['買點']}\n"
                f"  停損{r['停損']} 停利{r['停利']} 量比{r['量比']}x 杯{r['杯深%']}% 柄{r['柄深%']}%"
            )
    return "\n".join(out)


def main():
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = (arg[0] if arg else os.environ.get("MARKET", "TW")).upper()
    markets = ["TW", "HK", "US"] if sel == "ALL" else [sel]
    failed = False
    for m in markets:
        off = REF[m][1]
        today = (datetime.now(timezone.utc) + timedelta(hours=off)).strftime("%Y-%m-%d")
        try:
            if not is_trading_day(m) and "--force" not in sys.argv:
                print(f"[{m}] 今天非交易日或資料未更新,略過")
                continue
            df = cup_handle.scan(None, m)
            send(fmt(df, today, m))
        except Exception:
            err = traceback.format_exc()[-800:]
            print(err)
            send(f"❌ 杯柄掃描失敗 {m} {today}\n{err}")
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
