"""
ICT CRT 알림 전용 봇 (단일 파일) — Twelve Data 시세 + Discord 알림
====================================================================
실제 주문은 넣지 않습니다. CRT(레인지+스윕+MSS+FVG) 신호가 뜨면 Discord로
진입가/SL/TP 및 참고용(가상) 포지션 사이즈를 알려주기만 하는 알림 봇입니다.
주문 실행은 사용자가 알림을 보고 직접 넣어야 합니다.

필요한 것:
  1. Twelve Data API 키 — https://twelvedata.com (구글 계정으로 가입 가능, 전화번호 인증 불필요)
     가입 후 대시보드(API Keys)에서 키 확인
  2. Discord 채널 웹훅 URL
  3. Python 3.9+, pip install -r requirements.txt (requests, pandas, python-dotenv)

실행:
  python crt_discord_alert.py --once   # 1회 체크
  python crt_discord_alert.py --loop   # 60초 주기로 반복 (로컬 상시 실행용)

GitHub Actions로 15분마다 자동 실행하려면 .github/workflows/crt_alert.yml 참고.
"""

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
# 설정값
# ============================================================
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

INSTRUMENT = os.getenv("INSTRUMENT", "XAU/USD")  # Twelve Data 표기법 (슬래시 포함)
RANGE_INTERVAL = "4h"
ENTRY_INTERVAL = "15min"

# 실제 주문은 안 넣으므로, 알림 메시지에 표시할 참고용 가상 자본/리스크%
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.5"))

# --- CRT 레인지 / 구조 (기존 Pine/자동매매 봇과 동일 파라미터) ---
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

STATE_FILE = os.path.join(os.path.dirname(__file__), "last_alert_state.json")

TWELVEDATA_HOST = "https://api.twelvedata.com"


# ============================================================
# 안전장치
# ============================================================
def safety_precheck():
    if not TWELVEDATA_API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY가 .env에 설정되어 있지 않습니다.")


# ============================================================
# Twelve Data 헬퍼
# ============================================================
def fetch_candles(symbol, interval, outputsize=300):
    url = f"{TWELVEDATA_HOST}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
        "order": "ASC",  # 과거 -> 최신 순으로 받기
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data 오류: {data.get('message')}")

    rows = []
    for v in data.get("values", []):
        rows.append({
            "time": pd.to_datetime(v["datetime"]),
            "open": float(v["open"]), "high": float(v["high"]),
            "low": float(v["low"]), "close": float(v["close"]),
        })
    # order=ASC로 요청했지만 혹시 몰라 한 번 더 정렬
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


# ============================================================
# 신호 로직 (Pine/자동매매 봇과 동일 로직)
# ============================================================
def compute_range_and_trend(h4_df):
    h4_df = h4_df.copy()
    h4_df["ema"] = h4_df["close"].ewm(span=TREND_EMA_LEN, adjust=False).mean()
    window = h4_df.iloc[-(RANGE_LOOKBACK + 1):-1]
    range_high, range_low = window["high"].max(), window["low"].min()
    ema_now = h4_df["ema"].iloc[-1]
    ema_prev = h4_df["ema"].iloc[-5] if len(h4_df) > 5 else h4_df["ema"].iloc[0]
    close_now = h4_df["close"].iloc[-1]
    trend_up = close_now > ema_now > ema_prev
    trend_down = close_now < ema_now < ema_prev
    range_bar_time = h4_df["time"].iloc[-2]
    return range_high, range_low, trend_up, trend_down, range_bar_time


def compute_atr(df, length=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def find_bull_fvg(m15, sweep_idx, mss_idx):
    end = min(mss_idx, sweep_idx + MAX_SCAN_BARS)
    for i in range(sweep_idx + 2, end):
        high_old, low_new = m15["high"].iloc[i - 2], m15["low"].iloc[i]
        if low_new > high_old and (low_new - high_old) >= MIN_FVG_POINTS:
            return low_new, high_old
    return None, None


def find_bear_fvg(m15, sweep_idx, mss_idx):
    end = min(mss_idx, sweep_idx + MAX_SCAN_BARS)
    for i in range(sweep_idx + 2, end):
        low_old, high_new = m15["low"].iloc[i - 2], m15["high"].iloc[i]
        if high_new < low_old and (low_old - high_new) >= MIN_FVG_POINTS:
            return low_old, high_new
    return None, None


def scan_for_signal(m15, range_high, range_low, trend_up, trend_down, range_bar_time):
    left = right = SWING_LR
    n = len(m15)
    last_swing_high = last_swing_low = None
    bull_sweep_on = bull_mss = False
    bull_sweep_level = bull_sweep_idx = None
    bear_sweep_on = bear_mss = False
    bear_sweep_level = bear_sweep_idx = None
    signal = None

    for i in range(n):
        row = m15.iloc[i]
        confirm_idx = i - right
        if confirm_idx >= left:
            w = m15.iloc[confirm_idx - left: confirm_idx + right + 1]
            if m15["high"].iloc[confirm_idx] == w["high"].max():
                last_swing_high = m15["high"].iloc[confirm_idx]
            if m15["low"].iloc[confirm_idx] == w["low"].min():
                last_swing_low = m15["low"].iloc[confirm_idx]

        bars_since_range = (row["time"] - range_bar_time).total_seconds() / (15 * 60)
        range_fresh = 0 <= bars_since_range <= MAX_RANGE_AGE_BARS
        sweep_buf = row["close"] * SWEEP_BUFFER_PCT

        if range_fresh and trend_up and not bull_mss:
            if not bull_sweep_on:
                if row["low"] < range_low - sweep_buf and row["close"] > range_low:
                    bull_sweep_on, bull_sweep_level, bull_sweep_idx = True, row["low"], i
            else:
                if last_swing_high is not None and row["close"] > last_swing_high:
                    bull_mss = True
                    if i == n - 1:
                        top, _ = find_bull_fvg(m15, bull_sweep_idx, i)
                        if top is not None:
                            signal = {"side": "long", "entry": top,
                                      "sl": bull_sweep_level * (1 - SL_BUFFER_PCT),
                                      "range_high": range_high, "range_low": range_low,
                                      "bar_time": row["time"]}
                    bull_mss = bull_sweep_on = False

        if range_fresh and trend_down and not bear_mss:
            if not bear_sweep_on:
                if row["high"] > range_high + sweep_buf and row["close"] < range_high:
                    bear_sweep_on, bear_sweep_level, bear_sweep_idx = True, row["high"], i
            else:
                if last_swing_low is not None and row["close"] < last_swing_low:
                    bear_mss = True
                    if i == n - 1:
                        _, bottom = find_bear_fvg(m15, bear_sweep_idx, i)
                        if bottom is not None:
                            signal = {"side": "short", "entry": bottom,
                                      "sl": bear_sweep_level * (1 + SL_BUFFER_PCT),
                                      "range_high": range_high, "range_low": range_low,
                                      "bar_time": row["time"]}
                    bear_mss = bear_sweep_on = False

    return signal


# ============================================================
# 로컬 상태 (중복 알림 방지용)
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_alert_bar_time": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ============================================================
# Discord 로그
# ============================================================
def send_discord(message: str):
    print(message.replace("**", ""))
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception as e:
        print(f"[경고] Discord 전송 실패: {e}")


# ============================================================
# 메인 루틴
# ============================================================
def run_once():
    safety_precheck()
    state = load_state()

    h4 = fetch_candles(INSTRUMENT, RANGE_INTERVAL, outputsize=max(TREND_EMA_LEN + 10, 220))
    m15 = fetch_candles(INSTRUMENT, ENTRY_INTERVAL, outputsize=300)

    if len(h4) < TREND_EMA_LEN + 5 or len(m15) < 60:
        print("캔들 데이터 부족. 스킵.")
        save_state(state)
        return

    vol_spiking = False
    if USE_ATR_FILTER:
        atr = compute_atr(m15, ATR_LEN)
        atr_now, atr_avg = atr.iloc[-1], atr.rolling(30).mean().iloc[-1]
        if pd.notna(atr_avg) and atr_avg > 0 and atr_now >= atr_avg * ATR_SPIKE_MULT:
            vol_spiking = True

    range_high, range_low, trend_up, trend_down, range_bar_time = compute_range_and_trend(h4)

    if not vol_spiking:
        signal = scan_for_signal(m15, range_high, range_low, trend_up, trend_down, range_bar_time)

        bar_time_iso = signal["bar_time"].isoformat() if signal else None
        if signal and bar_time_iso != state.get("last_alert_bar_time"):
            entry, sl = signal["entry"], signal["sl"]
            sl_dist = abs(entry - sl)
            if sl_dist > 0:
                tp = (signal["range_high"] if signal["side"] == "short" else signal["range_low"]) if USE_RANGE_TP \
                    else (entry + sl_dist * RR_RATIO if signal["side"] == "long" else entry - sl_dist * RR_RATIO)

                risk_amount = ACCOUNT_EQUITY * RISK_PERCENT / 100
                units = risk_amount / sl_dist

                side_kr = "🟢 롱" if signal["side"] == "long" else "🔴 숏"
                send_discord(
                    f"**{side_kr} 신호 발생 — {INSTRUMENT}**\n"
                    f"진입가: {entry:.2f} / SL: {sl:.2f} / TP: {tp:.2f}\n"
                    f"참고 수량(가상): {units:.2f} / 리스크: {RISK_PERCENT}% "
                    f"(약 ${risk_amount:.2f}, 가상 자본 ${ACCOUNT_EQUITY:,.0f})\n"
                    f"⚠️ 자동 주문 없음 — 직접 체결하세요."
                )
                state["last_alert_bar_time"] = bar_time_iso

    print(f"[{datetime.now(timezone.utc)}] trend_up={trend_up} trend_down={trend_down} "
          f"volSpiking={vol_spiking}")
    save_state(state)


def run_loop(interval_sec=60):
    print("루프 모드 실행 중. Ctrl+C로 종료.")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[에러] {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if args.loop:
        run_loop()
    else:
        run_once()
