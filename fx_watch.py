# fx_watch.py
# Multi-pair market watcher using OANDA data. It does NOT place trades.
# Steps it tells you about, per pair:
#   4H bias -> last break of structure (BOS) -> zone marked (with quality)
#   -> approaching -> MITIGATION -> liquidity swept at the zone -> invalidated
# Runs in the cloud (GitHub Actions). pip install pandas numpy

import os
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------- SETTINGS ----------
INSTRUMENTS = [            # max 10. OANDA names.
    "XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY",
    "GBP_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
]
OFFSETS = {}               # optional, e.g. {"XAU_USD": -1.5}
ENV = os.environ.get("OANDA_ENV", "practice")   # practice or live
H4_DAILY_ALIGN = 17        # OANDA default: 17:00 New York
H4_ALIGN_TZ = "America/New_York"
SWING_4H = 4               # candles each side to confirm a 4H swing
APPROACH_PCT = 0.2         # "approaching" = within 0.2% of the zone
MIN_QUALITY = 0            # 0 = track all zones, 1 = need gap or sweep, 2 = both
ASIA_START, ASIA_END = 0, 7   # Asian session hours (UTC). My assumption.
STATUS_HOUR_UTC = 6
STATE_FILE = "fx_state.json"

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT = os.environ.get("TG_CHAT")
OANDA_TOKEN = os.environ.get("OANDA_TOKEN")

API = ("https://api-fxtrade.oanda.com" if ENV == "live"
       else "https://api-fxpractice.oanda.com")

RANK = {"waiting": 0, "approaching": 1, "mitigated": 2, "invalid": 3}
NAMES = {
    "bull": {"label": "BULLISH", "zone": "DEMAND", "act": "buys"},
    "bear": {"label": "BEARISH", "zone": "SUPPLY", "act": "sells"},
}


# ---------- OANDA DATA ----------
def candles(inst, gran, count, align=False):
    q = {"granularity": gran, "count": count, "price": "M"}
    if align:
        q["dailyAlignment"] = H4_DAILY_ALIGN
        q["alignmentTimezone"] = H4_ALIGN_TZ
    url = f"{API}/v3/instruments/{inst}/candles?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {OANDA_TOKEN}"})
    try:
        raw = urllib.request.urlopen(req, timeout=25).read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        raise RuntimeError(f"OANDA HTTP {e.code} for {inst} {gran}: {body}")
    rows = []
    for c in json.loads(raw)["candles"]:
        m = c["mid"]
        rows.append({
            "ts": pd.to_datetime(c["time"], utc=True),
            "open": float(m["o"]), "high": float(m["h"]),
            "low": float(m["l"]), "close": float(m["c"]),
            "complete": bool(c["complete"]),
        })
    return pd.DataFrame(rows)


def fetch_pair(inst):
    return {
        "h4": candles(inst, "H4", 250, align=True),
        "h1": candles(inst, "H1", 1000),
        "m15": candles(inst, "M15", 300),
        "d": candles(inst, "D", 5),
    }


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


def bull_zone(df):
    # Last bullish BOS = latest close above an earlier swing high.
    # Zone = last bearish candle at the pullback low before that break.
    # Quality (from the handbook): gap after the zone candle,
    # and the zone candle swept the previous candle's low.
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
    fvg = any(l[x] > h[x - 2] for x in range(zi + 2, brk + 1))
    sweep = zi >= 1 and bool(l[zi] < l[zi - 1])
    return {"brk": brk, "level": float(h[si]), "zone_idx": zi,
            "low": float(l[zi]), "high": float(h[zi]),
            "fvg": bool(fvg), "sweep": sweep}


def trend(df):
    up = bull_zone(df)
    dn = bull_zone(flip(df))
    if up is None and dn is None:
        return None
    ub = up["brk"] if up else -1
    db = dn["brk"] if dn else -1
    return "bull" if ub > db else "bear"


def find_zone(df4, side):
    d = df4 if side == "bull" else flip(df4)
    z = bull_zone(d)
    if z is None:
        return None
    if side == "bear":
        z = {"brk": z["brk"], "zone_idx": z["zone_idx"],
             "level": -z["level"], "low": -z["high"], "high": -z["low"],
             "fvg": z["fvg"], "sweep": z["sweep"]}
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


# ---------- LIQUIDITY (handbook ideas) ----------
def liquidity_events(side, m15, dd, now):
    # Only the side that fits the bias:
    # bull -> sell-side liquidity taken (lows), bear -> buy-side (highs)
    ev = []
    if len(dd) >= 2:
        prev, cur = dd.iloc[-2], dd.iloc[-1]
        day_key = str(cur.ts.date())
        if side == "bull" and cur.low < prev.low:
            ev.append(("Previous day low", float(prev.low), day_key))
        if side == "bear" and cur.high > prev.high:
            ev.append(("Previous day high", float(prev.high), day_key))
    today = now.date()
    day = m15[m15.ts.dt.date == today]
    asia = day[(day.ts.dt.hour >= ASIA_START) & (day.ts.dt.hour < ASIA_END)]
    after = day[day.ts.dt.hour >= ASIA_END]
    if len(asia) >= 4 and len(after) > 0:
        if side == "bull" and after.low.min() < asia.low.min():
            ev.append(("Asian low", float(asia.low.min()), str(today)))
        if side == "bear" and after.high.max() > asia.high.max():
            ev.append(("Asian high", float(asia.high.max()), str(today)))
    return ev


# ---------- FORMAT ----------
def dec(inst):
    if inst.startswith("XAU"):
        return 2
    return 3 if inst.endswith("JPY") else 5


def pr(inst, v):
    return f"{v + OFFSETS.get(inst, 0.0):.{dec(inst)}f}"


def dist_txt(inst, d):
    if inst.startswith("XAU"):
        return f"${d:.2f}"
    pip = 0.01 if inst.endswith("JPY") else 0.0001
    return f"{d / pip:.0f} pips"


def label(inst):
    return inst.replace("_", "/")


def quality_n(z):
    return int(z["fvg"]) + int(z["sweep"])


def quality_txt(z):
    grade = "ABC"[2 - quality_n(z)]
    parts = ["gap after it" if z["fvg"] else "no gap",
             "swept prior candle" if z["sweep"] else "no sweep"]
    return f"{grade} ({', '.join(parts)})"


def zone_line(inst, z):
    nm = NAMES[z["side"]]
    return (f"{nm['zone']} zone: {pr(inst, z['low'])} to "
            f"{pr(inst, z['high'])}\nZone quality: {quality_txt(z)}")


def gap_to_zone(z, price):
    return price - z["high"] if z["side"] == "bull" else z["low"] - price


def row_text(r):
    inst = r["inst"]
    head = f"{label(inst)} {pr(inst, r['price'])}"
    if not r["bias"]:
        return f"{head} | unclear"
    nm = NAMES[r["bias"]]
    if r["zone"] is None:
        return f"{head} | {nm['label']} | no zone"
    z = r["zone"]
    return (f"{head} | {nm['label']} | {nm['zone'].lower()} "
            f"{pr(inst, z['low'])}-{pr(inst, z['high'])} "
            f"[{'ABC'[2 - quality_n(z)]}] | {r['stage']}")


def status_text(rows, errors):
    lines = ["MARKET STATUS"]
    lines += [row_text(r) for r in rows]
    if errors:
        lines.append("Data error: " + ", ".join(label(e) for e in errors))
    return "\n".join(lines)


# ---------- CORE (one pair) ----------
def analyze(inst, data, pst, now):
    msgs = []
    tag = label(inst)
    h4, h1, m15, dd = data["h4"], data["h1"], data["m15"], data["d"]
    closed4 = h4[h4.complete].reset_index(drop=True)
    price = float(m15.close.iloc[-1])

    bias = trend(closed4)
    cur = bias or "none"
    if cur != pst.get("bias"):
        if bias:
            nm = NAMES[bias]
            msgs.append(
                f"[{tag}] 4H BIAS: {nm['label']}\n"
                f"Look for {nm['act']} only.\n"
                f"Price now: {pr(inst, price)}")
        pst["bias"] = cur

    zone = find_zone(closed4, bias) if bias else None
    if zone is not None and quality_n(zone) < MIN_QUALITY:
        zone = None
    if zone is None:
        pst["zone_key"] = None
        pst["stage"] = None
    else:
        status = zone_status(zone, h1, closed4, price)
        nm = NAMES[zone["side"]]
        if zone["key"] != pst.get("zone_key"):
            pst["zone_key"] = zone["key"]
            pst["stage"] = status
            head = (f"[{tag}] 4H BREAK OF STRUCTURE ({nm['label']})\n"
                    f"Price closed beyond {pr(inst, zone['level'])}.\n"
                    f"{zone_line(inst, zone)}\n")
            if status == "invalid":
                tail = ("Price already closed beyond this zone. "
                        "Waiting for a new BOS.")
            elif status == "mitigated":
                tail = ("Price already touched this zone since the BOS. "
                        "No mitigation alert will fire for it.")
            else:
                tail = ("Waiting for price to come back and mitigate it.\n"
                        f"Price now: {pr(inst, price)}. Distance: "
                        f"{dist_txt(inst, gap_to_zone(zone, price))}")
            msgs.append(head + tail)
        else:
            old = pst.get("stage") or "waiting"
            if RANK[status] > RANK[old]:
                pst["stage"] = status
                if status == "approaching":
                    msgs.append(
                        f"[{tag}] PRICE APPROACHING THE {nm['zone']} ZONE\n"
                        f"{zone_line(inst, zone)}\n"
                        f"Price now: {pr(inst, price)}. Distance: "
                        f"{dist_txt(inst, gap_to_zone(zone, price))}")
                elif status == "mitigated":
                    msgs.append(
                        f"[{tag}] MITIGATION ALERT ({nm['label']})\n"
                        f"Price entered the 4H {nm['zone']} zone.\n"
                        f"{zone_line(inst, zone)}\n"
                        f"Price now: {pr(inst, price)}\n"
                        "Your turn: 1H sweep, then 15M confirmation.\n"
                        "Risk idea from the handbook: 1% to 2% max on "
                        "your own account. Move the stop to breakeven "
                        "after the next structure break.")
                elif status == "invalid":
                    msgs.append(
                        f"[{tag}] ZONE INVALIDATED\n"
                        f"4H closed beyond the {nm['zone']} zone.\n"
                        f"{zone_line(inst, zone)}\n"
                        "A failed zone can flip to the other side. "
                        "Standing by for the next BOS.")

        # liquidity sweeps, only after the zone was mitigated
        if pst.get("stage") == "mitigated":
            done = pst.setdefault("sweeps", [])
            for name, level, day_key in liquidity_events(
                    zone["side"], m15, dd, now):
                k = f"{name}|{day_key}"
                if k in done:
                    continue
                done.append(k)
                del done[:-30]
                msgs.append(
                    f"[{tag}] LIQUIDITY SWEPT\n"
                    f"{name} ({pr(inst, level)}) was taken.\n"
                    f"This fits the {nm['label'].lower()} bias.\n"
                    f"Zone stage: {pst['stage']}. "
                    f"Price now: {pr(inst, price)}\n"
                    "Next: look for the 15M change in state of delivery.")

    row = {"inst": inst, "price": price, "bias": bias, "zone": zone,
           "stage": pst.get("stage")}
    return msgs, row


# ---------- RUN ALL PAIRS ----------
def run(st, now, manual=False, fetch=None):
    fetch = fetch or fetch_pair
    first = "pairs" not in st
    st.setdefault("pairs", {})
    out, rows, errors = [], [], []
    for inst in INSTRUMENTS[:10]:
        try:
            data = fetch(inst)
        except Exception as e:
            print("Data error:", inst, e)
            errors.append(inst)
            continue
        pst = st["pairs"].setdefault(inst, {})
        msgs, row = analyze(inst, data, pst, now)
        if not first:
            out.extend(msgs)
        rows.append(row)
    if not rows:
        raise RuntimeError("No data from OANDA for any pair")
    today = now.strftime("%Y-%m-%d")
    daily = (now.hour >= STATUS_HOUR_UTC and now.weekday() < 5
             and st.get("status_date") != today)
    if first or manual or daily:
        out.append(status_text(rows, errors))
        if first or daily:
            st["status_date"] = today
    return out


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
    if not OANDA_TOKEN:
        raise SystemExit("Missing OANDA_TOKEN secret")
    st = load_state()
    manual = os.environ.get("MANUAL") == "1"
    msgs = run(st, datetime.now(timezone.utc), manual)
    ok = True
    for i, m in enumerate(msgs):
        print(m, "\n---")
        if i:
            time.sleep(1)
        ok = send(m) and ok
    if ok:
        save_state(st)
    else:
        print("Send failed. State not saved, will retry next run.")


if __name__ == "__main__":
    main()
