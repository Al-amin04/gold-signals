# gold_watch.py
# Watches gold and tells you each step by Telegram:
#   4H bias -> last break of structure (BOS) -> zone marked
#   -> price approaching -> MITIGATION -> zone invalidated
# It does NOT place trades. You handle the 1H sweep and 15M confirmation.
# Runs in the cloud (GitHub Actions). pip install yfinance pandas numpy

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- SETTINGS ----------
TICKER = "GC=F"        # gold futures (free data)
OFFSET = 0.0           # broker price minus GC=F price
BIN_OFFSET = "0h"      # shifts the 4H candle times to match your broker
SWING_4H = 4           # candles each side to confirm a 4H swing
APPROACH_PCT = 0.25    # "approaching" = within 0.25% of the zone
STATUS_HOUR_UTC = 6    # daily status message time (UTC)
STATE_FILE = "market_state.json"

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT = os.environ.get("TG_CHAT")

RANK = {"waiting": 0, "approaching": 1, "mitigated": 2, "invalid": 3}
NAMES = {
    "bull": {"label": "BULLISH", "zone": "DEMAND", "act": "buys"},
    "bear": {"label": "BEARISH", "zone": "SUPPLY", "act": "sells"},
}


# ---------- DATA ----------
def get_h1():
    d = yf.Ticker(TICKER).history(period="60d", interval="1h")
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    out = pd.DataFrame({
        "ts": d.index,
        "open": d["Open"].values,
        "high": d["High"].values,
        "low": d["Low"].values,
        "close": d["Close"].values,
    })
    return out.reset_index(drop=True)


def to_4h(h1):
    d = h1.set_index("ts")[["open", "high", "low", "close"]]
    r = d.resample("4h", offset=BIN_OFFSET).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return r.reset_index()


def flip(df):
    d = df.copy()
    d["open"] = -df.open
    d["high"] = -df.low
    d["low"] = -df.high
    d["close"] = -df.close
    return d


# ---------- STRUCTURE ----------
def swings(df, n):
    h, l = df.high.values, df.low.values
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] == h[i - n:i + n + 1].max():
            highs.append((i, h[i]))
        if l[i] == l[i - n:i + n + 1].min():
            lows.append((i, l[i]))
    return highs, lows


def trend(df):
    sh, sl = swings(df, SWING_4H)
    if len(sh) < 2 or len(sl) < 2:
        return None
    if sh[-1][1] > sh[-2][1] and sl[-1][1] > sl[-2][1]:
        return "bull"
    if sh[-1][1] < sh[-2][1] and sl[-1][1] < sl[-2][1]:
        return "bear"
    return None


def bull_zone(df):
    # Last bullish BOS = latest close above an earlier swing high.
    # Zone = last bearish candle at the pullback low before that break.
    o, h, l, c = (df.open.values, df.high.values,
                  df.low.values, df.close.values)
    sh, _ = swings(df, SWING_4H)
    N = len(df)
    best = None
    for si, lvl in sh:
        brk = None
        for x in range(si + 1, N):
            if c[x] > lvl:
                brk = x
                break
        if brk is None:
            continue
        if best is None or (brk, si) > best:
            best = (brk, si)
    if best is None:
        return None
    brk, si = best
    li = si + int(np.argmin(l[si:brk + 1]))
    zi = li
    for x in range(li, max(li - 6, -1), -1):
        if c[x] < o[x]:
            zi = x
            break
    return {"brk": brk, "level": float(h[si]), "zone_idx": zi,
            "low": float(l[zi]), "high": float(h[zi])}


def find_zone(df4, side):
    d = df4 if side == "bull" else flip(df4)
    z = bull_zone(d)
    if z is None:
        return None
    if side == "bear":
        z = {"brk": z["brk"], "zone_idx": z["zone_idx"],
             "level": -z["level"], "low": -z["high"], "high": -z["low"]}
    z["bos_ts"] = df4.ts.iloc[z["brk"]]
    z["zone_ts"] = df4.ts.iloc[z["zone_idx"]]
    z["bos_end"] = z["bos_ts"] + pd.Timedelta("4h")
    z["side"] = side
    z["key"] = f"{side}|{z['bos_ts']}|{z['zone_ts']}"
    return z


def zone_status(z, h1, closed4, price):
    after = h1[h1.ts >= z["bos_end"]]
    later4 = closed4[closed4.ts > z["bos_ts"]]
    if z["side"] == "bull":
        invalid = bool((later4.close < z["low"]).any())
        touched = bool((after.low <= z["high"]).any())
        near = price <= z["high"] * (1 + APPROACH_PCT / 100)
    else:
        invalid = bool((later4.close > z["high"]).any())
        touched = bool((after.high >= z["low"]).any())
        near = price >= z["low"] * (1 - APPROACH_PCT / 100)
    if invalid:
        return "invalid"
    if touched:
        return "mitigated"
    if near:
        return "approaching"
    return "waiting"


# ---------- MESSAGES ----------
def p(v):
    return f"{v + OFFSET:.2f}"


def zone_line(z):
    nm = NAMES[z["side"]]
    return f"{nm['zone']} zone: {p(z['low'])} to {p(z['high'])}"


def gap_to_zone(z, price):
    if z["side"] == "bull":
        return price - z["high"]
    return z["low"] - price


def status_text(bias, z, stage, price):
    lines = ["GOLD STATUS", f"Price now: {p(price)}"]
    if not bias:
        lines.append("4H bias: unclear. No clear trend. Standing by.")
        return "\n".join(lines)
    nm = NAMES[bias]
    lines.append(f"4H bias: {nm['label']} ({nm['act']} only)")
    if z is None:
        lines.append("No break of structure found yet.")
    else:
        lines.append(f"Last BOS level: {p(z['level'])}")
        lines.append(zone_line(z))
        lines.append(f"Stage: {stage}")
        if stage in ("waiting", "approaching"):
            lines.append(f"Distance to zone: {gap_to_zone(z, price):.2f}")
    return "\n".join(lines)


# ---------- CORE ----------
def process(h1, st, now, manual=False):
    msgs = []
    h4 = to_4h(h1)
    closed4 = h4.iloc[:-1].reset_index(drop=True)
    price = float(h1.close.iloc[-1])

    bias = trend(closed4)
    cur = bias or "none"
    if cur != st.get("bias"):
        if bias:
            nm = NAMES[bias]
            msgs.append(
                f"4H BIAS: {nm['label']}\n"
                f"Structure shows a {nm['label'].lower()} trend. "
                f"Look for {nm['act']} only.\n"
                f"Price now: {p(price)}")
        else:
            msgs.append(
                "4H BIAS: UNCLEAR\n"
                "No clear trend on the 4H. Standing by.\n"
                f"Price now: {p(price)}")
        st["bias"] = cur

    zone = find_zone(closed4, bias) if bias else None
    stage = None
    if zone is None:
        st["zone_key"] = None
        st["stage"] = None
    else:
        status = zone_status(zone, h1, closed4, price)
        nm = NAMES[zone["side"]]
        if zone["key"] != st.get("zone_key"):
            st["zone_key"] = zone["key"]
            st["stage"] = status
            head = (f"4H BREAK OF STRUCTURE ({nm['label']})\n"
                    f"Price closed beyond {p(zone['level'])}.\n"
                    f"{zone_line(zone)}\n")
            if status == "invalid":
                tail = ("Price already closed beyond this zone. "
                        "Zone is invalid. Waiting for a new BOS.")
            elif status == "mitigated":
                tail = ("Price has already touched this zone since the "
                        "BOS. No mitigation alert will fire for it.")
            else:
                tail = ("Waiting for price to come back and mitigate it.\n"
                        f"Price now: {p(price)}. "
                        f"Distance: {gap_to_zone(zone, price):.2f}")
            msgs.append(head + tail)
        else:
            old = st.get("stage") or "waiting"
            if RANK[status] > RANK[old]:
                st["stage"] = status
                if status == "approaching":
                    msgs.append(
                        f"PRICE APPROACHING THE {nm['zone']} ZONE\n"
                        f"{zone_line(zone)}\n"
                        f"Price now: {p(price)}. "
                        f"Distance: {gap_to_zone(zone, price):.2f}")
                elif status == "mitigated":
                    msgs.append(
                        f"MITIGATION ALERT ({nm['label']})\n"
                        f"Price has entered the 4H {nm['zone']} zone.\n"
                        f"{zone_line(zone)}\n"
                        f"Price now: {p(price)}\n"
                        "Your turn: check the 1H sweep and the 15M "
                        "confirmation.")
                elif status == "invalid":
                    msgs.append(
                        f"ZONE INVALIDATED\n"
                        f"4H closed beyond the {nm['zone']} zone.\n"
                        f"{zone_line(zone)}\n"
                        "Standing by for the next BOS.")
        stage = st.get("stage")

    today = now.strftime("%Y-%m-%d")
    daily = (now.hour >= STATUS_HOUR_UTC and now.weekday() < 5
             and st.get("status_date") != today)
    if manual or daily:
        msgs.append(status_text(bias, zone, stage, price))
        if daily:
            st["status_date"] = today
    return msgs


# ---------- TELEGRAM + STATE ----------
def send(msg):
    if not TG_TOKEN or not TG_CHAT:
        print("Missing TG_TOKEN or TG_CHAT")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    ok = False
    for chat in TG_CHAT.split(","):
        chat = chat.strip()
        if not chat:
            continue
        try:
            data = urllib.parse.urlencode(
                {"chat_id": chat, "text": msg}).encode()
            urllib.request.urlopen(url, data, timeout=15)
            ok = True
        except Exception as e:
            print("Telegram error for", chat, e)
    return ok


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1)


def main():
    st = load_state()
    h1 = get_h1()
    manual = os.environ.get("MANUAL") == "1"
    msgs = process(h1, st, datetime.now(timezone.utc), manual)
    ok = True
    for m in msgs:
        print(m, "\n---")
        ok = send(m) and ok
    if ok:
        save_state(st)
    else:
        print("Send failed. State not saved, will retry next run.")


if __name__ == "__main__":
    main()
  
