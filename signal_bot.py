# signal_bot.py
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

TICKER = "GC=F"
OFFSET = -13.83
SWING = 3
RR = 2.0
BUFFER = 0.5
MIN_SL = 1.5
MAX_SL = 15.0
STATE_FILE = "last_signal.txt"

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT = os.environ.get("TG_CHAT")


def get_df(interval, period):
    d = yf.Ticker(TICKER).history(period=period, interval=interval)
    d = d.reset_index()
    d.columns = [str(c).lower() for c in d.columns]
    tcol = "datetime" if "datetime" in d.columns else "date"
    d["time"] = d[tcol].astype(str)
    d = d.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    return d


def flip(df):
    d = df.copy()
    d["high"], d["low"] = -df.low, -df.high
    d["close"] = -df.close
    return d


def swings(df, n):
    h, l = df.high.values, df.low.values
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] == h[i - n:i + n + 1].max():
            highs.append((i, h[i]))
        if l[i] == l[i - n:i + n + 1].min():
            lows.append((i, l[i]))
    return highs, lows


def bias(df):
    sh, sl = swings(df, SWING)
    if len(sh) < 2 or len(sl) < 2:
        return None
    if sh[-1][1] > sh[-2][1] and sl[-1][1] > sl[-2][1]:
        return "buy"
    if sh[-1][1] < sh[-2][1] and sl[-1][1] < sl[-2][1]:
        return "sell"
    return None


def find_buy(df, price):
    h, l, c = df.high.values, df.low.values, df.close.values
    t = df.time.values
    N = len(df)
    _, sl = swings(df, SWING)
    for li, level in reversed(sl[-4:]):
        j = None
        for x in range(li + 1, N):
            if c[x] < level:
                break
            if l[x] < level:
                j = x
                break
        if j is None or N - 1 - j > 40:
            continue
        top = h[li:j + 1].max()
        k = None
        for x in range(j + 1, N):
            if c[x] > top:
                k = x
                break
        if k is None or N - 1 - k > 20:
            continue
        gap = None
        for x in range(j + 2, k + 1):
            if l[x] > h[x - 2]:
                gap = (h[x - 2], l[x])
        if gap is None:
            continue
        stop = l[j:].min() - BUFFER
        if gap[0] <= price <= gap[1] and price > stop:
            return stop, ("buy", str(t[j]))
    return None


def send(msg):
    if not TG_TOKEN or not TG_CHAT:
        print("Missing TG_TOKEN or TG_CHAT")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for chat in TG_CHAT.split(","):
        chat = chat.strip()
        if not chat:
            continue
        try:
            data = urllib.parse.urlencode(
                {"chat_id": chat, "text": msg}).encode()
            urllib.request.urlopen(url, data, timeout=15)
        except Exception as e:
            print("Telegram error for", chat, e)


def main():
    if os.environ.get("TEST_MSG"):
        send("Cloud bot test: Telegram works")
    b = bias(get_df("1h", "1mo"))
    if not b:
        print("No H1 bias")
        return
    full = get_df("15m", "5d")
    price = float(full.close.iloc[-1])
    ltf = full.iloc[:-1].reset_index(drop=True)

    if b == "buy":
        r = find_buy(ltf, price)
    else:
        r = find_buy(flip(ltf), -price)
    if not r:
        print("No setup. Bias:", b)
        return

    stop, key = r
    if b == "sell":
        stop = -stop

    k = f"{b}|{key[1]}"
    if os.path.exists(STATE_FILE) and open(STATE_FILE).read().strip() == k:
        print("Already sent")
        return

    d = abs(price - stop)
    if not (MIN_SL <= d <= MAX_SL):
        print("Stop size out of range:", d)
        return

    tp = price + RR * d if b == "buy" else price - RR * d
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg = (
        f"GOLD SIGNAL ({b.upper()}) at {now}\n"
        f"Entry area: {price + OFFSET:.2f}\n"
        f"Stop loss: {stop + OFFSET:.2f}\n"
        f"Take profit: {tp + OFFSET:.2f}\n"
        f"Risk: ${d:.2f} per oz. Reward: 2x\n\n"
        f"Check your broker price first.\n"
        f"If price already left the entry area, skip it."
    )
    send(msg)
    open(STATE_FILE, "w").write(k)
    print("Signal sent")


if __name__ == "__main__":
    main()
