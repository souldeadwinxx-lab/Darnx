import os
import json
import time
import argparse
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 설정
# ============================================================

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

INSTRUMENT = os.getenv("INSTRUMENT", "XAU/USD")

RANGE_INTERVAL = "4h"
ENTRY_INTERVAL = "15min"

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.5"))

RANGE_LOOKBACK = 3
MAX_RANGE_AGE_BARS = 30

TREND_EMA_LEN = 200
SWING_LR = 3

MIN_FVG_POINTS = 0.5
SWEEP_BUFFER_PCT = 0.0002
MAX_SCAN_BARS = 40
SL_BUFFER_PCT = 0.0003

RR_RATIO = 2.0
USE_RANGE_TP = False

USE_ATR_FILTER = True
ATR_LEN = 14
ATR_SPIKE_MULT = 2.0

STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "last_alert_state.json"
)

TWELVEDATA_HOST = "https://api.twelvedata.com"


# ============================================================
# 안전장치
# ============================================================

def safety_precheck():
    if not TWELVEDATA_API_KEY:
        raise RuntimeError(
            "TWELVEDATA_API_KEY가 설정되지 않았습니다."
        )


# ============================================================
# Twelve Data
# ============================================================

def fetch_candles(symbol, interval, outputsize=300):
    url = f"{TWELVEDATA_HOST}/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
        "order": "ASC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data 오류: {data.get('message')}"
        )

    values = data.get("values", [])

    if not values:
        raise RuntimeError(
            f"Twelve Data 데이터 없음: {symbol} / {interval}"
        )

    rows = []

    for v in values:
        rows.append({
            "time": pd.to_datetime(v["datetime"]),
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .reset_index(drop=True)
    )


# ============================================================
# Range + Trend
# ============================================================

def compute_range_and_trend(h4_df):
    df = h4_df.copy()

    df["ema"] = df["close"].ewm(
        span=TREND_EMA_LEN,
        adjust=False
    ).mean()

    window = df.iloc[
        -(RANGE_LOOKBACK + 1):-1
    ]

    range_high = window["high"].max()
    range_low = window["low"].min()

    ema_now = df["ema"].iloc[-1]

    if len(df) > 5:
        ema_prev = df["ema"].iloc[-5]
    else:
        ema_prev = df["ema"].iloc[0]

    close_now = df["close"].iloc[-1]

    trend_up = (
        close_now > ema_now
        and ema_now > ema_prev
    )

    trend_down = (
        close_now < ema_now
        and ema_now < ema_prev
    )

    range_bar_time = df["time"].iloc[-2]

    return (
        range_high,
        range_low,
        trend_up,
        trend_down,
        range_bar_time
    )


# ============================================================
# ATR
# ============================================================

def compute_atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(length).mean()


# ============================================================
# Bullish FVG
# ============================================================

def find_bull_fvg(m15, sweep_idx, mss_idx):
    end = min(
        mss_idx,
        sweep_idx + MAX_SCAN_BARS
    )

    for i in range(
        sweep_idx + 2,
        end
    ):
        high_old = m15["high"].iloc[i - 2]
        low_new = m15["low"].iloc[i]

        if (
            low_new > high_old
            and low_new - high_old >= MIN_FVG_POINTS
        ):
            return low_new, high_old

    return None, None


# ============================================================
# Bearish FVG
# ============================================================

def find_bear_fvg(m15, sweep_idx, mss_idx):
    end = min(
        mss_idx,
        sweep_idx + MAX_SCAN_BARS
    )

    for i in range(
        sweep_idx + 2,
        end
    ):
        low_old = m15["low"].iloc[i - 2]
        high_new = m15["high"].iloc[i]

        if (
            high_new < low_old
            and low_old - high_new >= MIN_FVG_POINTS
        ):
            return low_old, high_new

    return None, None


# ============================================================
# CRT Signal
# ============================================================

def scan_for_signal(
    m15,
    range_high,
    range_low,
    trend_up,
    trend_down,
    range_bar_time
):
    left = SWING_LR
    right = SWING_LR

    n = len(m15)

    last_swing_high = None
    last_swing_low = None

    bull_sweep_on = False
    bull_mss = False
    bull_sweep_level = None
    bull_sweep_idx = None

    bear_sweep_on = False
    bear_mss = False
    bear_sweep_level = None
    bear_sweep_idx = None

    signal = None

    for i in range(n):

        row = m15.iloc[i]

        # ----------------------------------------------------
        # Swing detection
        # ----------------------------------------------------

        confirm_idx = i - right

        if confirm_idx >= left:

            window = m15.iloc[
                confirm_idx - left:
                confirm_idx + right + 1
            ]

            if (
                m15["high"].iloc[confirm_idx]
                == window["high"].max()
            ):
                last_swing_high = (
                    m15["high"].iloc[confirm_idx]
                )

            if (
                m15["low"].iloc[confirm_idx]
                == window["low"].min()
            ):
                last_swing_low = (
                    m15["low"].iloc[confirm_idx]
                )

        # ----------------------------------------------------
        # Range age
        # ----------------------------------------------------

        bars_since_range = (
            row["time"] - range_bar_time
        ).total_seconds() / (15 * 60)

        range_fresh = (
            0 <= bars_since_range <= MAX_RANGE_AGE_BARS
        )

        sweep_buffer = (
            row["close"] * SWEEP_BUFFER_PCT
        )

        # ----------------------------------------------------
        # Bullish setup
        # ----------------------------------------------------

        if range_fresh and trend_up and not bull_mss:

            if not bull_sweep_on:

                if (
                    row["low"]
                    < range_low - sweep_buffer
                    and row["close"] > range_low
                ):
                    bull_sweep_on = True
                    bull_sweep_level = row["low"]
                    bull_sweep_idx = i

            else:

                if (
                    last_swing_high is not None
                    and row["close"] > last_swing_high
                ):

                    bull_mss = True

                    if i == n - 1:

                        entry, _ = find_bull_fvg(
                            m15,
                            bull_sweep_idx,
                            i
                        )

                        if entry is not None:

                            sl = (
                                bull_sweep_level
                                * (1 - SL_BUFFER_PCT)
                            )

                            signal = {
                                "side": "long",
                                "entry": entry,
                                "sl": sl,
                                "range_high": range_high,
                                "range_low": range_low,
                                "bar_time": row["time"],
                            }

                    bull_mss = False
                    bull_sweep_on = False

        # ----------------------------------------------------
        # Bearish setup
        # ----------------------------------------------------

        if range_fresh and trend_down and not bear_mss:

            if not bear_sweep_on:

                if (
                    row["high"]
                    > range_high + sweep_buffer
                    and row["close"] < range_high
                ):
                    bear_sweep_on = True
                    bear_sweep_level = row["high"]
                    bear_sweep_idx = i

            else:

                if (
                    last_swing_low is not None
                    and row["close"] < last_swing_low
                ):

                    bear_mss = True

                    if i == n - 1:

                        _, entry = find_bear_fvg(
                            m15,
                            bear_sweep_idx,
                            i
                        )

                        if entry is not None:

                            sl = (
                                bear_sweep_level
                                * (1 + SL_BUFFER_PCT)
                            )

                            signal = {
                                "side": "short",
                                "entry": entry,
                                "sl": sl,
                                "range_high": range_high,
                                "range_low": range_low,
                                "bar_time": row["time"],
                            }

                    bear_mss = False
                    bear_sweep_on = False

    return signal


# ============================================================
# State
# ============================================================

def load_state():
    if os.path.exists(STATE_FILE):

        try:
            with open(
                STATE_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                return json.load(f)

        except Exception:
            pass

    return {
        "last_alert_bar_time": None
    }


def save_state(state):
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# Discord
# ============================================================

def send_discord(message):
    print(message.replace("**", ""))

    if not DISCORD_WEBHOOK_URL:
        print(
            "[경고] DISCORD_WEBHOOK_URL이 없습니다."
        )
        return

    try:

        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=15
        )

        if response.status_code >= 400:
            print(
                f"[경고] Discord 전송 실패 "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

    except Exception as e:
        print(
            f"[경고] Discord 전송 실패: {e}"
        )


# ============================================================
# Main signal check
# ============================================================

def run_once():

    safety_precheck()

    state = load_state()

    print(
        f"[INFO] Checking {INSTRUMENT}"
    )

    # --------------------------------------------------------
    # H4
    # --------------------------------------------------------

    h4 = fetch_candles(
        INSTRUMENT,
        RANGE_INTERVAL,
        outputsize=max(
            TREND_EMA_LEN + 10,
            220
        )
    )

    # --------------------------------------------------------
    # M15
    # --------------------------------------------------------

    m15 = fetch_candles(
        INSTRUMENT,
        ENTRY_INTERVAL,
        outputsize=300
    )

    if (
        len(h4) < TREND_EMA_LEN + 5
        or len(m15) < 60
    ):
        print(
            "[INFO] 캔들 데이터 부족. 스킵."
        )
        save_state(state)
        return

    # --------------------------------------------------------
    # ATR filter
    # --------------------------------------------------------

    vol_spiking = False

    if USE_ATR_FILTER:

        atr = compute_atr(
            m15,
            ATR_LEN
        )

        atr_now = atr.iloc[-1]

        atr_avg = (
            atr
            .rolling(30)
            .mean()
            .iloc[-1]
        )

        if (
            pd.notna(atr_avg)
            and atr_avg > 0
            and atr_now >= atr_avg * ATR_SPIKE_MULT
        ):
            vol_spiking = True

    # --------------------------------------------------------
    # Range / trend
    # --------------------------------------------------------

    (
        range_high,
        range_low,
        trend_up,
        trend_down,
        range_bar_time
    ) = compute_range_and_trend(h4)

    print(
        f"[INFO] trend_up={trend_up} "
        f"trend_down={trend_down} "
        f"volSpiking={vol_spiking}"
    )

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    if vol_spiking:

        print(
            "[INFO] ATR spike filter로 스킵."
        )

        save_state(state)
        return

    signal = scan_for_signal(
        m15,
        range_high,
        range_low,
        trend_up,
        trend_down,
        range_bar_time
    )

    if signal is None:

        print(
            "[INFO] CRT 신호 없음."
        )

        save_state(state)
        return

    # --------------------------------------------------------
    # Duplicate alert check
    # --------------------------------------------------------

    bar_time_iso = (
        signal["bar_time"].isoformat()
    )

    if (
        bar_time_iso
        == state.get("last_alert_bar_time")
    ):

        print(
            "[INFO] 이미 알림을 보낸 신호입니다."
        )

        save_state(state)
        return

    # --------------------------------------------------------
    # Entry / SL / TP
    # --------------------------------------------------------

    entry = signal["entry"]
    sl = signal["sl"]

    sl_dist = abs(
        entry - sl
    )

    if sl_dist <= 0:

        print(
            "[ERROR] SL distance가 0입니다."
        )

        return

    if USE_RANGE_TP:

        if signal["side"] == "long":
            tp = signal["range_low"]
        else:
            tp = signal["range_high"]

    else:

        if signal["side"] == "long":

            tp = (
                entry
                + sl_dist * RR_RATIO
            )

        else:

            tp = (
                entry
                - sl_dist * RR_RATIO
            )

    # --------------------------------------------------------
    # Position size
    # --------------------------------------------------------

    risk_amount = (
        ACCOUNT_EQUITY
        * RISK_PERCENT
        / 100
    )

    units = (
        risk_amount
        / sl_dist
    )

    # --------------------------------------------------------
    # Discord message
    # --------------------------------------------------------

    if signal["side"] == "long":
        side_kr = "🟢 롱"
    else:
        side_kr = "🔴 숏"

    message = (
        f"**{side_kr} 신호 발생 — {INSTRUMENT}**\n"
        f"진입가: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP: {tp:.2f}\n"
        f"참고 수량(가상): {units:.2f}\n"
        f"리스크: {RISK_PERCENT}% "
        f"(약 ${risk_amount:.2f})\n"
        f"가상 자본: ${ACCOUNT_EQUITY:,.0f}\n"
        f"⚠️ 자동 주문 없음 — 직접 체결하세요."
    )

    send_discord(message)

    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    state["last_alert_bar_time"] = bar_time_iso

    save_state(state)

    print(
        "[INFO] Alert state 저장 완료."
    )


# ============================================================
# Loop mode
# ============================================================

def run_loop(interval_sec=60):

    print(
        "루프 모드 실행 중. Ctrl+C로 종료."
    )

    while True:

        try:
            run_once()

        except Exception as e:

            print(
                f"[에러] {e}"
            )

        time.sleep(interval_sec)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true"
    )

    parser.add_argument(
        "--loop",
        action="store_true"
    )

    args = parser.parse_args()

    if args.loop:
        run_loop()
    else:
        run_once()
