"""
MarketRadar Backend — FastAPI + yfinance + Telegram
Çalıştırmak için:
    pip install fastapi uvicorn yfinance python-telegram-bot apscheduler requests
    python main.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
import uvicorn
import yfinance as yf
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("marketradar")

# ── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")   # .env veya ortam değişkeni
TELEGRAM_CHATID = os.getenv("TELEGRAM_CHATID", "")
REPORT_HOUR     = int(os.getenv("REPORT_HOUR", "8")) # Sabah kaçta rapor gitsin
REPORT_MINUTE   = int(os.getenv("REPORT_MINUTE", "0"))

# ── Enstrüman Tanımları ───────────────────────────────────────────────────────
INSTRUMENTS = [
    {"sym": "XAUUSD", "name": "Altın",    "category": "Emtia",  "ticker": "GC=F"},
    {"sym": "NASDAQ", "name": "NASDAQ",   "category": "Endeks", "ticker": "NQ=F"},
    {"sym": "DAX",    "name": "DAX",      "category": "Endeks", "ticker": "^GDAXI"},
    {"sym": "BTCUSD", "name": "Bitcoin",  "category": "Kripto", "ticker": "BTC-USD"},
    {"sym": "EURUSD", "name": "EUR/USD",  "category": "Forex",  "ticker": "EURUSD=X"},
    {"sym": "GBPUSD", "name": "GBP/USD",  "category": "Forex",  "ticker": "GBPUSD=X"},
    {"sym": "USDJPY", "name": "USD/JPY",  "category": "Forex",  "ticker": "USDJPY=X"},
    {"sym": "EURJPY", "name": "EUR/JPY",  "category": "Forex",  "ticker": "EURJPY=X"},
    {"sym": "EURAUD", "name": "EUR/AUD",  "category": "Forex",  "ticker": "EURAUD=X"},
    {"sym": "GBPJPY", "name": "GBP/JPY",  "category": "Forex",  "ticker": "GBPJPY=X"},
    {"sym": "USDCAD", "name": "USD/CAD",  "category": "Forex",  "ticker": "USDCAD=X"},
    {"sym": "EURCAD", "name": "EUR/CAD",  "category": "Forex",  "ticker": "EURCAD=X"},
    {"sym": "GBPCAD", "name": "GBP/CAD",  "category": "Forex",  "ticker": "GBPCAD=X"},
]

DECIMALS = {
    "USDJPY": 3, "EURJPY": 3, "GBPJPY": 3,
    "EURUSD": 4, "GBPUSD": 4, "EURAUD": 4,
    "USDCAD": 4, "EURCAD": 4, "GBPCAD": 4,
    "XAUUSD": 2, "NASDAQ": 2, "DAX": 2, "BTCUSD": 1,
}

# ── In-memory cache ───────────────────────────────────────────────────────────
market_cache: dict = {}
alarms: list = []
signal_history: list = []

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="MarketRadar API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve HTML frontend from ../frontend/
if os.path.isdir("../frontend"):
    app.mount("/app", StaticFiles(directory="../frontend", html=True), name="frontend")


# ── Pydantic Modeller ─────────────────────────────────────────────────────────
class AlarmCreate(BaseModel):
    sym: str
    direction: str   # "above" | "below"
    price: float

class TelegramConfig(BaseModel):
    token: str
    chat_id: str

class TelegramSend(BaseModel):
    token: str
    chat_id: str
    report_type: str  # "full" | "bias" | "test" | "alarms" | "single"
    sym: Optional[str] = None


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def fmt(sym: str, price: float) -> str:
    d = DECIMALS.get(sym, 4)
    return f"{price:.{d}f}"


def calc_levels(sym: str, price: float, high: float, low: float) -> dict:
    """Pivot noktaları + Fibonacci seviyeleri hesapla."""
    # ── Pivot (klasik) ──
    pivot = (high + low + price) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)

    # ── Fibonacci (son yüksek-alçak arası) ──
    diff = high - low
    fib = {
        "0%":    fmt(sym, low),
        "23.6%": fmt(sym, low + diff * 0.236),
        "38.2%": fmt(sym, low + diff * 0.382),
        "50%":   fmt(sym, low + diff * 0.500),
        "61.8%": fmt(sym, low + diff * 0.618),
        "78.6%": fmt(sym, low + diff * 0.786),
        "100%":  fmt(sym, high),
    }

    return {
        "R3": fmt(sym, r3), "R2": fmt(sym, r2), "R1": fmt(sym, r1),
        "S1": fmt(sym, s1), "S2": fmt(sym, s2), "S3": fmt(sym, s3),
        "pivot": fmt(sym, pivot),
        "fib": fib,
    }


def calc_bias(price: float, sma20: float, sma50: float, change_pct: float) -> str:
    """
    Bias: fiyat SMA20 ve SMA50'nin üzerindeyse + günlük değişim pozitifse YÜKSELİŞ.
    """
    above_20 = price > sma20 if sma20 else None
    above_50 = price > sma50 if sma50 else None

    score = 0
    if above_20: score += 1
    if above_50: score += 1
    if change_pct > 0.1: score += 1
    if change_pct < -0.1: score -= 1

    if score >= 2:   return "bull"
    if score <= -1:  return "bear"
    return "neutral"


# ── Fiyat Çekme ───────────────────────────────────────────────────────────────
async def fetch_instrument(inst: dict) -> Optional[dict]:
    """yfinance ile günlük + haftalık veri çek, seviyeleri hesapla."""
    try:
        ticker = yf.Ticker(inst["ticker"])

        # Son 60 günlük veri (SMA50 için yeterli)
        hist = ticker.history(period="60d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 2:
            log.warning(f"{inst['sym']} — veri boş")
            return None

        closes  = hist["Close"].values
        highs   = hist["High"].values
        lows    = hist["Low"].values

        price   = float(closes[-1])
        prev    = float(closes[-2])
        change  = ((price - prev) / prev) * 100

        # Swing high/low (son 20 bar)
        period_high = float(highs[-20:].max())
        period_low  = float(lows[-20:].min())

        # Hareketli ortalamalar
        sma20 = float(closes[-20:].mean()) if len(closes) >= 20 else price
        sma50 = float(closes[-50:].mean()) if len(closes) >= 50 else price

        levels = calc_levels(inst["sym"], price, period_high, period_low)
        bias   = calc_bias(price, sma20, sma50, change)

        return {
            "sym":      inst["sym"],
            "name":     inst["name"],
            "category": inst["category"],
            "price":    price,
            "price_fmt":fmt(inst["sym"], price),
            "change":   round(change, 3),
            "sma20":    round(sma20, 4),
            "sma50":    round(sma50, 4),
            "high_20":  round(period_high, 4),
            "low_20":   round(period_low, 4),
            "levels":   levels,
            "bias":     bias,
            "updated":  datetime.now().isoformat(),
        }

    except Exception as e:
        log.error(f"{inst['sym']} fetch hatası: {e}")
        return None


async def refresh_all():
    """Tüm enstrümanları asenkron olarak yenile."""
    log.info("🔄 Fiyatlar yenileniyor...")
    tasks = [fetch_instrument(inst) for inst in INSTRUMENTS]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r:
            sym = r["sym"]
            prev_price = market_cache.get(sym, {}).get("price")
            market_cache[sym] = r
            # Alarmları kontrol et
            if prev_price:
                await check_alarms(sym, r["price"], prev_price)

    log.info(f"✅ {sum(1 for r in results if r)}/{len(INSTRUMENTS)} enstrüman güncellendi")


# ── Alarm Kontrolü ────────────────────────────────────────────────────────────
async def check_alarms(sym: str, price: float, prev_price: float):
    triggered = []
    for alarm in alarms:
        if alarm["sym"] != sym or alarm["triggered"]:
            continue
        hit = (alarm["direction"] == "above" and price >= alarm["price"]) or \
              (alarm["direction"] == "below" and price <= alarm["price"])
        if hit:
            alarm["triggered"] = True
            alarm["triggered_at"] = datetime.now().isoformat()
            alarm["triggered_price"] = price
            triggered.append(alarm)

            direction_str = "↑ üzerine çıktı" if alarm["direction"] == "above" else "↓ altına indi"
            signal = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "date": datetime.now().strftime("%d.%m.%Y"),
                "sym": sym,
                "type": "bull" if alarm["direction"] == "above" else "bear",
                "desc": f"Alarm tetiklendi: {sym} {direction_str} {alarm['price']}",
            }
            signal_history.insert(0, signal)
            if len(signal_history) > 100:
                signal_history.pop()

            # Telegram bildirimi
            if TELEGRAM_TOKEN and TELEGRAM_CHATID:
                icon = "🟢" if alarm["direction"] == "above" else "🔴"
                msg = (f"{icon} *ALARM TETİKLENDİ*\n\n"
                       f"📌 *{sym}* {direction_str}\n"
                       f"🎯 Hedef: `{alarm['price']}`\n"
                       f"💰 Mevcut: `{fmt(sym, price)}`\n"
                       f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
                await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHATID, msg)


# ── Telegram ──────────────────────────────────────────────────────────────────
async def send_telegram_message(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.json()
    except Exception as e:
        log.error(f"Telegram hatası: {e}")
        return {"ok": False, "description": str(e)}


def build_report(report_type: str, sym: Optional[str] = None) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    if report_type == "test":
        return f"✅ *MarketRadar Test Mesajı*\n\nBağlantı başarılı! ⚡\n_{now}_"

    if report_type == "bias":
        lines = [f"📊 *GÜNLÜK BIAS ÖZETİ*\n_{now}_\n"]
        for inst in INSTRUMENTS:
            d = market_cache.get(inst["sym"])
            if not d:
                continue
            icon = "🟢" if d["bias"] == "bull" else "🔴" if d["bias"] == "bear" else "🟡"
            label = "YÜKSELİŞ" if d["bias"] == "bull" else "DÜŞÜŞ" if d["bias"] == "bear" else "NÖTR"
            lines.append(f"{icon} *{inst['sym']}*: {label} — `{d['price_fmt']}`")
        return "\n".join(lines)

    if report_type == "full":
        lines = [f"📈 *TAM PİYASA RAPORU*\n_{now}_\n"]
        for inst in INSTRUMENTS:
            d = market_cache.get(inst["sym"])
            if not d:
                continue
            icon = "🟢" if d["bias"] == "bull" else "🔴" if d["bias"] == "bear" else "🟡"
            change_str = f"+{d['change']:.2f}%" if d["change"] >= 0 else f"{d['change']:.2f}%"
            lines.append(
                f"{icon} *{inst['sym']}* ({inst['name']})\n"
                f"💰 `{d['price_fmt']}` ({change_str})\n"
                f"📍 R1: {d['levels']['R1']} | S1: {d['levels']['S1']}\n"
                f"📐 Fib 61.8%: {d['levels']['fib']['61.8%']} | 38.2%: {d['levels']['fib']['38.2%']}\n"
            )
        return "\n".join(lines)

    if report_type == "alarms":
        active = [a for a in alarms if not a["triggered"]]
        triggered = [a for a in alarms if a["triggered"]]
        lines = [f"🔔 *ALARM DURUMU*\n_{now}_\n",
                 f"⏳ Aktif alarm: {len(active)}"]
        for a in active:
            lines.append(f"• {a['sym']} {'↑' if a['direction']=='above' else '↓'} `{a['price']}`")
        lines.append(f"\n✅ Tetiklenen: {len(triggered)}")
        for a in triggered:
            lines.append(f"• {a['sym']} → `{a['price']}` ({a.get('triggered_at','')[:16]})")
        return "\n".join(lines)

    if report_type == "single" and sym:
        d = market_cache.get(sym)
        if not d:
            return f"❌ {sym} verisi bulunamadı"
        icon = "🟢" if d["bias"] == "bull" else "🔴" if d["bias"] == "bear" else "🟡"
        change_str = f"+{d['change']:.2f}%" if d["change"] >= 0 else f"{d['change']:.2f}%"
        lv = d["levels"]
        fib = lv["fib"]
        return (
            f"{icon} *{sym}* Detaylı Analiz\n_{now}_\n\n"
            f"💰 Fiyat: `{d['price_fmt']}` ({change_str})\n"
            f"📊 SMA20: `{d['sma20']}` | SMA50: `{d['sma50']}`\n\n"
            f"📍 *Direnç Seviyeleri*\n"
            f"R3: `{lv['R3']}` | R2: `{lv['R2']}` | R1: `{lv['R1']}`\n\n"
            f"📍 *Destek Seviyeleri*\n"
            f"S1: `{lv['S1']}` | S2: `{lv['S2']}` | S3: `{lv['S3']}`\n\n"
            f"📐 *Fibonacci Seviyeleri*\n"
            f"78.6%: `{fib['78.6%']}` | 61.8%: `{fib['61.8%']}`\n"
            f"50%: `{fib['50%']}` | 38.2%: `{fib['38.2%']}`\n"
            f"23.6%: `{fib['23.6%']}`"
        )

    return "❓ Bilinmeyen rapor türü"


# ── Zamanlanmış Görev ─────────────────────────────────────────────────────────
async def scheduled_morning_report():
    """Her sabah otomatik rapor gönder."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHATID:
        return
    log.info("📨 Sabah raporu gönderiliyor...")
    await refresh_all()
    msg = build_report("bias")
    await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHATID, msg)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/prices")
async def get_prices():
    """Tüm enstrümanların güncel fiyatlarını döndür."""
    return {"data": list(market_cache.values()), "updated": datetime.now().isoformat()}


@app.get("/api/prices/{sym}")
async def get_price(sym: str):
    sym = sym.upper()
    d = market_cache.get(sym)
    if not d:
        raise HTTPException(404, f"{sym} bulunamadı")
    return d


@app.post("/api/refresh")
async def trigger_refresh():
    """Manuel fiyat yenileme."""
    asyncio.create_task(refresh_all())
    return {"status": "refreshing"}


@app.get("/api/alarms")
async def get_alarms():
    return {"alarms": alarms}


@app.post("/api/alarms")
async def create_alarm(alarm: AlarmCreate):
    new_alarm = {
        "id": int(datetime.now().timestamp() * 1000),
        "sym": alarm.sym.upper(),
        "direction": alarm.direction,
        "price": alarm.price,
        "triggered": False,
        "created_at": datetime.now().isoformat(),
    }
    alarms.append(new_alarm)
    log.info(f"⚡ Alarm eklendi: {new_alarm['sym']} {new_alarm['direction']} {new_alarm['price']}")
    return new_alarm


@app.delete("/api/alarms/{alarm_id}")
async def delete_alarm(alarm_id: int):
    global alarms
    alarms = [a for a in alarms if a["id"] != alarm_id]
    return {"status": "deleted"}


@app.get("/api/signals")
async def get_signals():
    return {"signals": signal_history}


@app.post("/api/telegram/send")
async def telegram_send(body: TelegramSend):
    msg = build_report(body.report_type, body.sym)
    result = await send_telegram_message(body.token, body.chat_id, msg)
    if result.get("ok"):
        signal_history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%d.%m.%Y"),
            "sym": body.sym or "SYSTEM",
            "type": "bull",
            "desc": f"Telegram raporu gönderildi: {body.report_type}",
        })
        return {"status": "ok"}
    raise HTTPException(400, result.get("description", "Bilinmeyen hata"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "instruments": len(market_cache),
        "alarms": len(alarms),
        "signals": len(signal_history),
        "time": datetime.now().isoformat(),
    }


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    log.info("🚀 MarketRadar başlatılıyor...")

    # Başlangıçta fiyatları çek
    asyncio.create_task(refresh_all())

    # Zamanlayıcı: Her 5 dakikada fiyat yenile
    scheduler = AsyncIOScheduler()
    scheduler.add_job(refresh_all, "interval", minutes=5, id="price_refresh")
    scheduler.add_job(
        scheduled_morning_report,
        "cron", hour=REPORT_HOUR, minute=REPORT_MINUTE,
        id="morning_report"
    )
    scheduler.start()
    log.info(f"⏰ Sabah raporu saati: {REPORT_HOUR:02d}:{REPORT_MINUTE:02d}")
    log.info("✅ MarketRadar hazır → http://localhost:8000")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
