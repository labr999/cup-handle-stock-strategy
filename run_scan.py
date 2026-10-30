#!/usr/bin/env python3
"""每日收盤後掃描杯柄形態,結果推送到 Telegram。
環境變數: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
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


def send(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    # Telegram 單則上限 4096 字,超過就分段
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


def is_trading_day():
    """用 0050 最後一根日K日期判斷今天台股有沒有開盤(遇假日就不推)。"""
    today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
    d = yf.download("0050.TW", period="5d", progress=False, auto_adjust=True)
    return (not d.empty) and d.index[-1].date() == today


def fmt(df, today):
    if df.empty:
        return f"📊 杯柄掃描 {today}\n今天沒有符合的標的"
    out = [f"📊 杯柄掃描 {today}"]
    for status, icon in (("突破", "🚀"), ("突破量不足", "⚠️"), ("接近突破", "👀")):
        sub = df[df["狀態"] == status]
        if sub.empty:
            continue
        out.append(f"\n{icon} {status}({len(sub)})")
        for _, r in sub.iterrows():
            code = r["代號"].split(".")[0]
            out.append(
                f"{code} {r['名稱']} 收{r['收盤']} 買點{r['買點']}\n"
                f"  停損{r['停損']} 停利{r['停利']} 量比{r['量比']}x 杯{r['杯深%']}% 柄{r['柄深%']}%"
            )
    return "\n".join(out)


def main():
    today = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d")
    try:
        if not is_trading_day() and "--force" not in sys.argv:
            print("今天非交易日或資料未更新,略過")
            return
        df = cup_handle.scan()
        send(fmt(df, today))
    except Exception:
        err = traceback.format_exc()[-800:]
        print(err)
        send(f"❌ 杯柄掃描失敗 {today}\n{err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
