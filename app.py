# app.py (fixed)
import streamlit as st
import websocket
import threading
import queue
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
import collections, time, os, csv, io
import altair as alt

# ---------------- CONFIG ----------------
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
ROLL_WINDOW_DEFAULT = 60
HISTORY_LEN = 5000
SPREAD_Z_DEFAULT = 3.0
PRICE_MOVE_TICKS_DEFAULT = 3
PAPER_SL_PIPS_DEFAULT = 20
PAPER_TP_PIPS_DEFAULT = 40
EVENTS_FILE = "events_log.csv"

# ---------------- HELPERS ----------------
def now_iso(): return datetime.now(timezone.utc).isoformat()
def pip_value(sym):
    if "JPY" in sym: return 0.01
    if "XAU" in sym: return 0.01
    return 0.0001

def write_event(evt):
    header = not os.path.exists(EVENTS_FILE)
    with open(EVENTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["time","symbol","event","side","entry","sl","tp","close","pnl","meta"])
        w.writerow([
            evt.get("time",""),
            evt.get("symbol",""),
            evt.get("event",""),
            evt.get("side",""),
            evt.get("entry",""),
            evt.get("sl",""),
            evt.get("tp",""),
            evt.get("close",""),
            evt.get("pnl",""),
            json.dumps(evt.get("meta",""))
        ])

# ---------------- PAGE SETUP ----------------
st.set_page_config(layout="wide", page_title="Liquidity Sweeper (personal)")
st.title("Liquidity Sweeper — Personal (paper-trade)")
st.caption("Research tool. Paper-trade only. Not financial advice.")

# ---------------- SESSION STATE ----------------
if "mq" not in st.session_state: st.session_state.mq = queue.Queue(maxsize=5000)
if "ws_thread" not in st.session_state: st.session_state.ws_thread = None
if "ws_obj" not in st.session_state: st.session_state.ws_obj = None
if "connected" not in st.session_state: st.session_state.connected = False

if "ROLL_WINDOW" not in st.session_state: st.session_state.ROLL_WINDOW = ROLL_WINDOW_DEFAULT
if "SPREAD_Z" not in st.session_state: st.session_state.SPREAD_Z = SPREAD_Z_DEFAULT
if "PRICE_MOVE_TICKS" not in st.session_state: st.session_state.PRICE_MOVE_TICKS = PRICE_MOVE_TICKS_DEFAULT
if "PAPER_SL" not in st.session_state: st.session_state.PAPER_SL = PAPER_SL_PIPS_DEFAULT
if "PAPER_TP" not in st.session_state: st.session_state.PAPER_TP = PAPER_TP_PIPS_DEFAULT

if "df" not in st.session_state:
    utc = datetime.now(timezone.utc)
    rows = []
    for s in SYMBOLS:
        rows.append({"symbol": s, "time": utc, "bid": 0.0, "ask": 0.0, "mid": 0.0, "spread": 0.0,
                     "h_spread": 0.0, "l_spread": 999999.0, "signal": "-", "ATR": 0.0, "VWAP": 0.0, "MA20": 0.0, "spread_atr": 0.0})
    st.session_state.df = pd.DataFrame(rows)

if "hist" not in st.session_state:
    st.session_state.hist = {s: collections.deque(maxlen=HISTORY_LEN) for s in SYMBOLS}
if "roll" not in st.session_state:
    st.session_state.roll = {s: collections.deque(maxlen=ROLL_WINDOW_DEFAULT) for s in SYMBOLS}
if "events" not in st.session_state: st.session_state.events = []
if "positions" not in st.session_state: st.session_state.positions = []

# ---------------- INDICATORS ----------------
def compute_atr_from_mids(mids, window=14):
    if len(mids) < 2: return 0.0
    diffs = [abs(mids[i]-mids[i-1]) for i in range(1,len(mids))]
    recent = diffs[-window:]
    return float(np.mean(recent)) if recent else 0.0

def compute_vwap(history):
    if not history: return 0.0
    arr = np.array([h["mid"] for h in history], dtype=float)
    return float(arr.mean())

def compute_ma(mids, n=20):
    if len(mids) < 1: return 0.0
    arr = np.array(mids, dtype=float)
    if len(arr) < n: return float(arr.mean())
    return float(np.mean(arr[-n:]))

# ---------------- SWEEP DETECTION ----------------
def detect_sweep(symbol, mid, spread):
    r = st.session_state.roll[symbol]
    if len(r) < max(10, int(st.session_state.ROLL_WINDOW/2)): return None
    spreads = np.array([x["spread"] for x in r], dtype=float)
    mean = spreads.mean(); std = spreads.std(ddof=0)
    if std == 0: return None
    z = (spread - mean) / std
    mids = [x["mid"] for x in r]
    recent_max = max(mids); recent_min = min(mids)
    pip = pip_value(symbol)
    move_thresh = st.session_state.PRICE_MOVE_TICKS * pip
    if z >= st.session_state.SPREAD_Z and mid > recent_max + move_thresh:
        return {"type":"BUY_SWEEP","z":float(z),"recent_max":recent_max,"mid":mid}
    if z >= st.session_state.SPREAD_Z and mid < recent_min - move_thresh:
        return {"type":"SELL_SWEEP","z":float(z),"recent_min":recent_min,"mid":mid}
    return None

# ---------------- PAPER TRADING ----------------
def open_paper(symbol, side, entry):
    pip = pip_value(symbol)
    sl = entry - st.session_state.PAPER_SL * pip if side=="BUY" else entry + st.session_state.PAPER_SL * pip
    tp = entry + st.session_state.PAPER_TP * pip if side=="BUY" else entry - st.session_state.PAPER_TP * pip
    pos = {"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,"open":now_iso()}
    st.session_state.positions.append(pos)
    ev = {"time":now_iso(),"symbol":symbol,"event":"OPEN","side":side,"entry":entry,"sl":sl,"tp":tp}
    st.session_state.events.append(ev); write_event(ev)
    return pos

def close_paper(pos, reason, close_price):
    pip = pip_value(pos["symbol"])
    pnl = (close_price - pos["entry"]) / pip * (1 if pos["side"]=="BUY" else -1)
    ev = {"time":now_iso(),"symbol":pos["symbol"],"event":"CLOSE","side":pos["side"],"close":close_price,"pnl":pnl,"reason":reason}
    st.session_state.events.append(ev); write_event(ev)
    try: st.session_state.positions.remove(pos)
    except: pass
    return ev

# ---------------- PROCESS LOOP ----------------
def process_loop(stop_event):
    while not stop_event.is_set():
        try:
            raw = st.session_state.mq.get(timeout=0.5)
        except queue.Empty:
            time.sleep(0.01); continue
        try:
            data = json.loads(raw)
        except:
            continue
        sym = data.get("symbol") or data.get("instrument")
        if not sym: continue
        try:
            bid = float(data.get("bid",0)); ask = float(data.get("ask",0))
        except:
            continue
        mid = (bid+ask)/2.0; spread = ask-bid
        # update df
        if sym in st.session_state.df['symbol'].values:
            idx = st.session_state.df[st.session_state.df['symbol']==sym].index[0]
            ts = datetime.fromtimestamp(int(data.get("timestamp", time.time()*1000))/1000, tz=timezone.utc) if data.get("timestamp") else now_iso()
            st.session_state.df.at[idx,'time'] = ts
            st.session_state.df.at[idx,'bid'] = bid
            st.session_state.df.at[idx,'ask'] = ask
            st.session_state.df.at[idx,'mid'] = mid
            st.session_state.df.at[idx,'spread'] = spread
            st.session_state.df.at[idx,'h_spread'] = max(st.session_state.df.at[idx,'h_spread'], spread)
            st.session_state.df.at[idx,'l_spread'] = min(st.session_state.df.at[idx,'l_spread'], spread)
        # append history and rolling
        st.session_state.hist.setdefault(sym, collections.deque(maxlen=HISTORY_LEN)).append({"time":now_iso(),"mid":mid,"spread":spread})
        st.session_state.roll.setdefault(sym, collections.deque(maxlen=st.session_state.ROLL_WINDOW)).append({"time":now_iso(),"mid":mid,"spread":spread})
        # compute indicators
        mids_roll = [x["mid"] for x in st.session_state.roll[sym]]
        atr = compute_atr_from_mids(mids_roll, window=14)
        vwap = compute_vwap(list(st.session_state.hist[sym])[-st.session_state.ROLL_WINDOW:])
        ma20 = compute_ma([x["mid"] for x in list(st.session_state.hist[sym])], n=20)
        spread_atr = (spread/atr) if atr>0 else np.nan
        if sym in st.session_state.df['symbol'].values:
            idx = st.session_state.df[st.session_state.df['symbol']==sym].index[0]
            st.session_state.df.at[idx,'ATR'] = atr
            st.session_state.df.at[idx,'VWAP'] = vwap
            st.session_state.df.at[idx,'MA20'] = ma20
            st.session_state.df.at[idx,'spread_atr'] = spread_atr
        # detect event
        event = None
        # optional session filter
        allow_session = True
        if st.session_state.get("session_filter_on", False):
            nowh = datetime.utcnow().hour
            starth = st.session_state.get("session_start", 12)
            endh = st.session_state.get("session_end", 16)
            allow_session = (starth <= nowh <= endh)
        if allow_session:
            ev = detect_sweep(sym, mid, spread)
            if ev:
                require_trend = st.session_state.get("require_trend", True)
                ok_trend = True
                if require_trend:
                    if ev["type"]=="BUY_SWEEP":
                        ok_trend = (mid > vwap) and (mid > ma20)
                    else:
                        ok_trend = (mid < vwap) and (mid < ma20)
                if ok_trend:
                    event = ev
        # act on event
        if event:
            side = "BUY" if event["type"]=="BUY_SWEEP" else "SELL"
            open_paper(sym, side, float(event["mid"]))
            if sym in st.session_state.df['symbol'].values:
                idx = st.session_state.df[st.session_state.df['symbol']==sym].index[0]
                st.session_state.df.at[idx,'signal'] = f"{side}(SWEEP)"
            st.session_state.events.append({"time":now_iso(),"symbol":sym,"event":"DETECTED","meta":event})
        else:
            if sym in st.session_state.df['symbol'].values:
                idx = st.session_state.df[st.session_state.df['symbol']==sym].index[0]
                st.session_state.df.at[idx,'signal'] = "-"
        # manage positions
        for p in list(st.session_state.positions):
            cur = float(st.session_state.df.loc[st.session_state.df['symbol']==p['symbol'],'mid'].values[0])
            if p['side']=="BUY":
                if cur <= p['sl']:
                    close_paper(p,"SL",cur)
                elif cur >= p['tp']:
                    close_paper(p,"TP",cur)
            else:
                if cur >= p['sl']:
                    close_paper(p,"SL",cur)
                elif cur <= p['tp']:
                    close_paper(p,"TP",cur)

# ---------------- WEBSOCKET CALLBACKS ----------------
def on_message(ws, raw):
    try:
        st.session_state.mq.put_nowait(raw)
    except queue.Full:
        pass

def on_error(ws, e):
    # store error
    st.session_state.last_error = str(e)

def on_close(ws, code, msg):
    st.session_state.connected = False
    st.session_state.last_error = f"Closed {code} {msg}"

def on_open(ws):
    try:
        syms = ",".join(SYMBOLS)
        payload = {"userKey": st.session_state.get("api_key",""), "symbol": syms}
        ws.send(json.dumps(payload))
        st.session_state.connected = True
    except Exception as e:
        st.session_state.last_error = str(e)

def start_ws(api_key, ws_url="wss://marketdata.tradermade.com/feedadv"):
    st.session_state.api_key = api_key
    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    st.session_state.ws_obj = ws
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    st.session_state.ws_thread = t
    stop_event = threading.Event()
    st.session_state.stop_event = stop_event
    pt = threading.Thread(target=process_loop, args=(stop_event,), daemon=True)
    pt.start()
    st.session_state.processor_thread = pt
    st.session_state.connected = True

def stop_all():
    try:
        if st.session_state.ws_obj: st.session_state.ws_obj.close()
    except: pass
    try:
        if st.session_state.stop_event: st.session_state.stop_event.set()
    except: pass
    st.session_state.connected = False

# ---------------- CSV SIMULATOR ----------------
def simulate_from_csv(file_bytes, speed=1.0):
    try:
        s = file_bytes.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(s))
    except Exception:
        try:
            df = pd.read_csv(file_bytes)
        except Exception:
            st.error("CSV read failed"); return
    times = []
    for t in df.iloc[:,0]:
        try:
            ts = int(t)
        except:
            try:
                ts = int(pd.to_datetime(t).timestamp()*1000)
            except:
                ts = int(time.time()*1000)
        times.append(ts)
    df['__ts_ms'] = times
    st.session_state.connected = True
    stop_event = threading.Event()
    st.session_state.stop_event = stop_event
    pt = threading.Thread(target=process_loop, args=(stop_event,), daemon=True)
    pt.start()
    st.session_state.processor_thread = pt
    def sender():
        start = df['__ts_ms'].iloc[0]
        for i,row in df.iterrows():
            if st.session_state.stop_event.is_set(): break
            rel = (row['__ts_ms'] - start)/1000.0
            time.sleep(max(0.0, rel / max(0.01, speed)))
            msg = json.dumps({"symbol": str(row['symbol']), "bid": float(row['bid']), "ask": float(row['ask']), "timestamp": int(row['__ts_ms'])})
            try:
                st.session_state.mq.put_nowait(msg)
            except queue.Full:
                pass
        st.session_state.connected = False
    th = threading.Thread(target=sender, daemon=True)
    th.start()
    st.session_state.simulate_thread = th

# ---------------- UI ----------------
left, right = st.columns([3,1])
with right:
    st.subheader("Controls")
    default_key = st.secrets["TRADERMADE_KEY"] if "TRADERMADE_KEY" in st.secrets else ""
    api_in = st.text_input("TraderMade API Key (or 'mock')", type="password", value=default_key)
    uploaded = st.file_uploader("Upload CSV to simulate (timestamp,symbol,bid,ask)", type=["csv"])
    if st.button("Start (WS or Mock)"):
        if api_in.strip().lower() == "mock" or uploaded is not None:
            if uploaded is None:
                st.warning("Using mock but no CSV uploaded.")
            else:
                simulate_from_csv(uploaded, speed=1.0)
                st.success("Started CSV simulation.")
        else:
            if api_in.strip()=="":
                st.warning("Enter API key or use mock.")
            else:
                start_ws(api_in.strip()); st.success("Connecting...")

    if st.button("Stop"):
        stop_all(); st.success("Stopped.")

    st.markdown("---")
    st.write("Tuning")
    st.session_state.SPREAD_Z = st.number_input("Spread z-threshold", value=float(SPREAD_Z_DEFAULT), step=0.5)
    st.session_state.ROLL_WINDOW = st.number_input("Rolling window (ticks)", value=int(ROLL_WINDOW_DEFAULT), min_value=10, max_value=1000)
    st.session_state.PRICE_MOVE_TICKS = st.number_input("Price move ticks", value=int(PRICE_MOVE_TICKS_DEFAULT), min_value=1)
    st.session_state.PAPER_SL = st.number_input("Paper SL (pips)", value=int(PAPER_SL_PIPS_DEFAULT), min_value=1)
    st.session_state.PAPER_TP = st.number_input("Paper TP (pips)", value=int(PAPER_TP_PIPS_DEFAULT), min_value=1)
    st.checkbox("Require trend confirmation (VWAP+MA)", value=True, key="require_trend")
    st.checkbox("Enable session filter (UTC hours below)", value=False, key="session_filter_on")
    st.session_state.session_start = st.number_input("Session start hour (UTC)", value=12, min_value=0, max_value=23)
    st.session_state.session_end = st.number_input("Session end hour (UTC)", value=16, min_value=0, max_value=23)
    st.markdown("---")
    st.write("Papertrade")
    st.write(f"Open positions: {len(st.session_state.positions)}")
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE,"rb") as f:
            st.download_button("Download events log", f, file_name=EVENTS_FILE)

with left:
    st.subheader("Live table")
    df_show = st.session_state.df.copy()
    df_show['time'] = df_show['time'].astype(str)
    st.dataframe(df_show.style.format({"bid":"{:.5f}","ask":"{:.5f}","mid":"{:.5f}","spread":"{:.5f}","ATR":"{:.5f}","VWAP":"{:.5f}","MA20":"{:.5f}","spread_atr":"{:.2f}"}), height=300)
    st.markdown("### Pair ranking (live)")
    ranks = []
    for s in SYMBOLS:
        histlist = list(st.session_state.hist.get(s, []))[-st.session_state.ROLL_WINDOW:]
        avg_spread = np.mean([h['spread'] for h in histlist]) if histlist else np.nan
        spread_std = np.std([h['spread'] for h in histlist]) if histlist else np.nan
        row = st.session_state.df[st.session_state.df['symbol']==s].iloc[0].to_dict()
        score = (1.0/(1+ (avg_spread if not np.isnan(avg_spread) else 1))) + (0 if np.isnan(spread_std) else 1.0/(1+spread_std)) + (row['ATR']*100 if row['ATR']>0 else 0)
        ranks.append({"symbol":s,"avg_spread":avg_spread,"spread_std":spread_std,"ATR":row['ATR'],"score":score})
    rank_df = pd.DataFrame(sorted(ranks, key=lambda x:-x['score']))[['symbol','avg_spread','spread_std','ATR','score']]
    st.table(rank_df)

    st.markdown("### Recent events & positions")
    if st.session_state.events:
        ev_df = pd.DataFrame(st.session_state.events[-200:])
        st.dataframe(ev_df, height=220)
    else:
        st.write("No events yet.")
    st.markdown("Open positions")
    if st.session_state.positions:
        st.table(pd.DataFrame(st.session_state.positions))
    else:
        st.write("No open positions.")

    st.markdown("### Chart (select symbol)")
    sym_choice = st.selectbox("Symbol", SYMBOLS)
    hist = list(st.session_state.hist.get(sym_choice, []))[-1000:]
    if len(hist) >= 5:
        hdf = pd.DataFrame(hist); hdf['time'] = pd.to_datetime(hdf['time'])
        chart = alt.Chart(hdf).mark_line().encode(x='time:T', y='mid:Q').properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Waiting for data...")

st.markdown("---")
st.caption("Paper-trade only. Test thoroughly. Not financial advice.")
