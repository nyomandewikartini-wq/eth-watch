import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model

# =========================================================
# ETH_Watch — UI redesign inspired by the approved mockup
# Keeps the existing LSTM artifacts and rule-based CEP logic.
# =========================================================
st.set_page_config(
    page_title="ETH Watch | Ethereum Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette
BG = "#0B0912"          # Midnight
PANEL = "#181322"       # Deep Plum
PANEL_2 = "#211A2E"
BORDER = "#352843"
TEXT = "#F7F2FF"
MUTED = "#B9AEC9"
LILAC = "#C084FC"
TEAL = "#35D6A5"         # Positive / normal
AMBER = "#F6C453"        # Caution / volume
RED = "#FF5D8F"          # Anomaly / negative
BLUE = "#A78BFA"
PURPLE = "#C084FC"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    h1, h2, h3, .hero-title, .brand-name {{ font-family: 'Playfair Display', Georgia, serif !important; font-weight:600 !important; letter-spacing:-.45px !important; }}
    .stApp {{ background: radial-gradient(ellipse at 80% 0%, #1A0D2B 0%, {BG} 52%); color: {TEXT}; }}
    [data-testid="stHeader"] {{ background: rgba(8,5,16,.92); }}
    [data-testid="stSidebar"] {{ background: #090611; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.1rem; }}
    /* Keep navigation visible and reduce the chance of accidental collapse. */
    [data-testid="stSidebar"] {{ min-width: 285px !important; }}
    [data-testid="stSidebarCollapseButton"] {{ visibility: hidden !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ line-height: 1.55; }}
    [data-testid="stRadio"] > label {{ font-size: .78rem; font-weight: 800; letter-spacing: 1px; color: {LILAC} !important; }}
    [data-testid="stRadio"] div[role="radiogroup"] label {{ padding: 7px 9px; border-radius: 9px; }}
    [data-testid="stRadio"] div[role="radiogroup"] label:hover {{ background: {PANEL_2}; }}
    .block-container {{ max-width: 1660px; padding-top: 1.05rem; padding-bottom: 2.2rem; }}
    h1, h2, h3, h4, p, label {{ color: {TEXT}; }}
    [data-testid="stCaptionContainer"] p, .muted {{ color: {MUTED} !important; }}
    div[data-testid="stMetric"] {{
      background: linear-gradient(145deg, #1C112C, {PANEL});
      border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 17px;
      min-height: 112px; box-shadow: 0 10px 28px rgba(0,0,0,.16);
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .78rem; letter-spacing: .2px; }}
    [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 750; }}
    [data-testid="stMetricDelta"] {{ font-size: .77rem; }}
    div[data-testid="stPlotlyChart"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 13px; padding: 5px 5px 0 5px; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 11px; overflow: hidden; }}
    div[data-testid="stAlert"] {{ border-radius: 11px; }}
    hr {{ border-color: {BORDER}; }}
    .brand-row {{ display:flex; align-items:center; gap:11px; padding: 5px 3px 20px 3px; }}
    .brand-icon {{ display:flex; align-items:center; justify-content:center; width:43px; height:43px; border-radius:13px; background:linear-gradient(135deg,#C084FC,#8B5CF6); color:#10071C; font-weight:900; font-size:17px; box-shadow:0 0 22px rgba(192,132,252,.18); }}
    .brand-name {{ font-size:22px; font-weight:800; letter-spacing:-.8px; line-height:1.1; color:{TEXT}; }}
    .brand-sub {{ color:{MUTED}; font-size:11px; margin-top:4px; }}
    .hero {{ border:1px solid {BORDER}; border-radius:19px; padding:25px 28px; margin: 4px 0 17px 0;
      background: radial-gradient(circle at 78% 45%, rgba(182,76,255,.23), transparent 25%),
                  radial-gradient(circle at 92% 22%, rgba(141,91,255,.16), transparent 23%),
                  linear-gradient(110deg,#1B1029 0%,#10091B 58%,#0A0611 100%);
      min-height:124px; box-shadow: inset 0 1px 0 rgba(255,255,255,.025); }}
    .hero-kicker {{ color:{LILAC}; font-size:10px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }}
    .hero-title {{ color:{TEXT}; font-family:Playfair Display, Georgia, serif; font-size:clamp(30px,3vw,44px); font-weight:600; letter-spacing:-.5px; margin-top:4px; line-height:1.08; }}
    .hero-title span {{ color:{LILAC}; }}
    .hero-copy {{ color:#C8BBD9; font-size:14px; margin-top:8px; }}
    .section-title {{ font-family:Playfair Display, Georgia, serif; font-size:19px; font-weight:600; color:{TEXT}; margin: 4px 0 3px 0; }}
    .section-sub {{ font-size:11px; color:{MUTED}; margin-bottom:10px; }}
    .pill {{ display:inline-block; padding:5px 10px; border:1px solid {BORDER}; border-radius:999px; font-size:11px; color:{MUTED}; background:{PANEL}; }}
    .status-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{TEAL}; margin-right:7px; }}
    .stButton>button {{ border-radius:9px; border:1px solid {BORDER}; background:#1A1027; color:{TEXT}; }}
    .stButton>button:hover {{ border-color:#C084FC; color:#C084FC; }}
    [data-testid="stRadio"] label {{ color:{MUTED}; }}
    [data-testid="stRadio"] div[role="radiogroup"] {{ gap:.35rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "eth_lstm_best.keras"
SCALER_PATH = MODEL_DIR / "eth_close_scaler.joblib"
CONFIG_PATH = MODEL_DIR / "eth_lstm_config.json"
RESULT_DIR = BASE_DIR / "results"
TEST_METRICS_PATH = RESULT_DIR / "lstm_final_test_metrics.csv"
TEST_PREDICTIONS_PATH = RESULT_DIR / "lstm_final_test_predictions.csv"


def load_test_evaluation():
    """Load persisted final test evaluation; never recompute it on live data."""
    metrics = pd.read_csv(TEST_METRICS_PATH) if TEST_METRICS_PATH.exists() else None
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH) if TEST_PREDICTIONS_PATH.exists() else None
    if predictions is not None:
        required = {"Date", "Actual_Close", "LSTM_Prediction"}
        if not required.issubset(predictions.columns):
            predictions = None
        else:
            predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
            for col in ("Actual_Close", "LSTM_Prediction", "Naive_Prediction", "LSTM_Error"):
                if col in predictions.columns:
                    predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
            predictions = predictions.dropna(subset=["Date", "Actual_Close", "LSTM_Prediction"]).sort_values("Date")
    return metrics, predictions


@st.cache_resource
def load_lstm_artifacts():
    for path in (MODEL_PATH, SCALER_PATH, CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(f"File yang diperlukan tidak ditemukan: {path}")
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    lookback = int(config["lookback"])
    if model.input_shape[1] != lookback:
        raise ValueError(
            f"Lookback config ({lookback}) tidak cocok dengan input model ({model.input_shape[1]})."
        )
    if model.input_shape[-1] != 1 or model.output_shape[-1] != 1:
        raise ValueError("Model yang digunakan harus menerima dan menghasilkan satu fitur.")
    return model, scaler, config


@st.cache_data(ttl=60)
def get_eth_intraday_data():
    """Fetch recent 1-minute ETH candles for near-real-time CEP monitoring.

    This is a third-party aggregated market feed, not exchange tick-by-tick data.
    Yahoo Finance may delay, omit, or limit intraday observations.
    """
    data = yf.download("ETH-USD", period="1d", interval="1m", auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("Intraday data is currently unavailable from Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    if "Datetime" not in data.columns and "Date" in data.columns:
        data = data.rename(columns={"Date": "Datetime"})
    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Intraday fields unavailable: {missing}")
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Datetime", "Close"]).sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid intraday observations were returned.")
    return data


@st.cache_data(ttl=300)
def get_eth_data():
    data = yf.download("ETH-USD", period="1y", interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("Data ETH tidak ditemukan. Periksa koneksi atau ketersediaan Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    if "Date" not in data.columns and "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Kolom data tidak tersedia: {missing}")
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
    return data


def dark_chart(fig, height=360, show_legend=True):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=11),
        margin=dict(l=18, r=18, t=48, b=18),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=MUTED, size=10), bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#203149", gridwidth=.6, tickfont=dict(color=MUTED), zeroline=False)
    return fig


def heading(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def money(value):
    return f"${float(value):,.2f}"


# Sidebar navigation
with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-row">
          <div class="brand-icon">Ξ</div>
          <div><div class="brand-name">ETH_Watch</div>
          <div class="brand-sub">Market Intelligence Platform</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "NAVIGATION",
        ["Market Overview · Ringkasan", "LSTM Forecasting · Peramalan", "CEP Monitoring · Kejadian", "Market Analytics · Analitik", "Model & Methodology · Metode", "Data Explorer · Data"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown(
        f"""
        <div style="padding:12px 11px;border:1px solid {BORDER};border-radius:13px;background:{PANEL};">
          <div style="font-size:11px;color:{TEAL};font-weight:800;letter-spacing:1px;">RINGKASAN PROYEK / PROJECT OVERVIEW</div>
          <div style="font-size:13px;font-weight:650;margin-top:8px;color:{TEXT};">Analisis Pasar Ethereum</div>
          <div style="font-size:11px;line-height:1.6;color:{MUTED};margin-top:7px;">Peramalan harga dengan LSTM dan deteksi kejadian berbasis aturan CEP.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Sumber data / Data source: Yahoo Finance · data historis harian, diperbarui saat data diminta / daily historical data, refreshed on request")
    if st.button("Muat Ulang Data / Refresh Data", use_container_width=True):
        get_eth_data.clear()
        st.rerun()

# Load data and model
try:
    with st.spinner("Memuat data Ethereum..."):
        eth_data = get_eth_data()
    if len(eth_data) < 2:
        raise ValueError("Data belum cukup untuk menghitung perubahan harga.")

    latest = eth_data.iloc[-1]
    previous = eth_data.iloc[-2]
    latest_price = float(latest["Close"])
    previous_price = float(previous["Close"])
    change_pct = (latest_price - previous_price) / previous_price * 100 if previous_price else 0.0
    latest_date = pd.Timestamp(latest["Date"])
    latest_volume = float(latest["Volume"])

    # Shared forecast calculation
    model, scaler, config = load_lstm_artifacts()
    lookback = int(config["lookback"])
    horizon = int(config.get("forecast_horizon_days", 1))
    if len(eth_data) < lookback:
        raise ValueError(f"Data tidak cukup; model memerlukan {lookback} observasi.")
    recent_close = eth_data[["Close"]].tail(lookback).values
    X_latest = scaler.transform(recent_close).reshape(1, lookback, 1)
    prediction_scaled = model.predict(X_latest, verbose=0)
    prediction_usd = float(scaler.inverse_transform(prediction_scaled)[0, 0])
    forecast_date = latest_date + pd.Timedelta(days=horizon)

    # CEP calculations shared by pages
    cep_window = 14
    cep_threshold = 2.0
    cep_data = eth_data[["Date", "Close", "Volume"]].copy()
    cep_data["Rolling_Mean"] = cep_data["Close"].rolling(cep_window, min_periods=cep_window).mean().shift(1)
    cep_data["Rolling_SD"] = cep_data["Close"].rolling(cep_window, min_periods=cep_window).std().shift(1)
    cep_data["Upper_Bound"] = cep_data["Rolling_Mean"] + cep_threshold * cep_data["Rolling_SD"]
    cep_data["Lower_Bound"] = cep_data["Rolling_Mean"] - cep_threshold * cep_data["Rolling_SD"]
    cep_data["Price_Anomaly"] = (
        (cep_data["Close"] > cep_data["Upper_Bound"]) |
        (cep_data["Close"] < cep_data["Lower_Bound"])
    ) & cep_data["Rolling_Mean"].notna()
    price_anomalies = cep_data.loc[cep_data["Price_Anomaly"]].copy()

    trend_days = 3
    trend_data = eth_data[["Date", "Close"]].copy()
    trend_data["Daily_Change"] = trend_data["Close"].diff()
    trend_data["Up_Day"] = trend_data["Daily_Change"] > 0
    trend_data["Down_Day"] = trend_data["Daily_Change"] < 0
    trend_data["Up_Sequence"] = trend_data["Up_Day"].rolling(trend_days, min_periods=trend_days).sum().eq(trend_days)
    trend_data["Down_Sequence"] = trend_data["Down_Day"].rolling(trend_days, min_periods=trend_days).sum().eq(trend_days)
    trend_events = trend_data.loc[trend_data["Up_Sequence"] | trend_data["Down_Sequence"]].copy()
    trend_events["Trend_Event"] = "Harga naik 3 hari berturut-turut"
    trend_events.loc[trend_events["Down_Sequence"], "Trend_Event"] = "Harga turun 3 hari berturut-turut"

    volume_window = 14
    volume_threshold = 2.0
    sequence_days = 3
    sequence_data = eth_data[["Date", "Close", "Volume"]].copy()
    sequence_data["Volume_Mean_Past"] = sequence_data["Volume"].rolling(volume_window, min_periods=volume_window).mean().shift(1)
    sequence_data["Volume_SD_Past"] = sequence_data["Volume"].rolling(volume_window, min_periods=volume_window).std().shift(1)
    sequence_data["Volume_Upper_Bound"] = sequence_data["Volume_Mean_Past"] + volume_threshold * sequence_data["Volume_SD_Past"]
    sequence_data["Volume_Anomaly"] = (
        (sequence_data["Volume"] > sequence_data["Volume_Upper_Bound"]) &
        sequence_data["Volume_Upper_Bound"].notna()
    )
    sequence_data["Price_Anomaly"] = cep_data["Price_Anomaly"].values
    sequence_records = []
    for price_idx in sequence_data.index[sequence_data["Price_Anomaly"]]:
        end_idx = min(price_idx + sequence_days, len(sequence_data) - 1)
        following = sequence_data.loc[price_idx + 1:end_idx]
        volume_matches = following[following["Volume_Anomaly"]]
        if not volume_matches.empty:
            volume_idx = int(volume_matches.index[0])
            price_event = sequence_data.loc[price_idx]
            volume_event = sequence_data.loc[volume_idx]
            sequence_records.append({
                "Tanggal Anomali Harga": price_event["Date"],
                "Harga (USD)": float(price_event["Close"]),
                "Tanggal Anomali Volume": volume_event["Date"],
                "Volume": float(volume_event["Volume"]),
                "Jarak (hari observasi)": volume_idx - int(price_idx),
            })
    sequence_events = pd.DataFrame(sequence_records)
    volume_anomalies = sequence_data.loc[sequence_data["Volume_Anomaly"]].copy()

    # Header
    status_text = f"Historical daily data: {latest_date.strftime('%d %b %Y')} · Yahoo Finance"
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;gap:15px;margin-bottom:8px;">
          <div style="font-size:12px;color:{MUTED};">MARKET INTELLIGENCE / <span style="color:{TEXT};">{page.upper()}</span></div>
          <div style="font-size:11px;color:{MUTED};"><span class="status-dot"></span>{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Market Overview · Ringkasan":
        st.markdown(
            f"""
            <div class="hero">
              <div class="hero-kicker">ETHEREUM MARKET INTELLIGENCE</div>
              <div class="hero-title">Ethereum Market <span>Watch</span></div>
              <div class="hero-copy"><strong>Market Monitoring, Short-Term Forecasting &amp; Anomaly Detection Using LSTM and CEP.</strong><br/>Memantau harga dan volume harian Ethereum, memperkirakan harga penutupan berikutnya dengan LSTM, serta menandai pergerakan yang tidak biasa menggunakan aturan statistik dan Complex Event Processing (CEP).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            st.metric("Harga Terakhir / Latest ETH Price", money(latest_price), delta=f"{change_pct:+.2f}% vs hari sebelumnya / previous day")
        with k2:
            forecast_change = (prediction_usd - latest_price) / latest_price * 100 if latest_price else 0
            st.metric(f"Ramalan {horizon} Hari / Next Forecast", money(prediction_usd), delta=f"{forecast_change:+.2f}% vs harga terakhir / latest close")
        with k3:
            st.metric("Volume Harian / Daily Volume", f"{latest_volume:,.0f}", delta=f"{(latest_volume / eth_data['Volume'].iloc[-2] - 1) * 100:+.2f}% vs hari sebelumnya / previous day" if eth_data["Volume"].iloc[-2] else None)
        with k4:
            st.metric("Anomali Harga / Price Anomalies (1Y)", f"{len(price_anomalies)}", delta="Historical rule-based CEP", delta_color="off")

        with st.expander("Cara membaca ringkasan pasar"):
            st.markdown(
                "- **Current price:** harga penutupan harian terbaru, bukan kuotasi tick-by-tick.\n"
                "- **Forecast:** keluaran LSTM untuk penutupan hari target; bukan saran investasi.\n"
                "- **Volume:** volume pada bar harian terakhir yang disediakan sumber data.\n"
                "- **Price anomalies:** jumlah observasi historis yang berada di luar batas rolling mean ± 2 simpangan baku. Ini menandai nilai yang tidak biasa menurut aturan, bukan bukti kesalahan data."
            )

        left, right = st.columns([2.35, 1], gap="medium")
        with left:
            heading("Harga Ethereum & Ramalan / Price & Forecast", "Harga penutupan aktual dan satu titik ramalan LSTM berikutnya / Actual close plus the next LSTM estimate")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=eth_data["Date"], open=eth_data["Open"], high=eth_data["High"],
                low=eth_data["Low"], close=eth_data["Close"], name="Daily OHLC",
                increasing_line_color=TEAL, decreasing_line_color=RED,
                increasing_fillcolor="rgba(182,76,255,0.72)",
                decreasing_fillcolor="rgba(255,82,119,0.72)",
            ))
            fig.add_trace(go.Scatter(
                x=[latest_date, forecast_date], y=[latest_price, prediction_usd],
                mode="lines+markers", name="LSTM next-close estimate",
                line=dict(color=AMBER, width=2.5, dash="dash"),
                marker=dict(color=AMBER, size=8),
            ))
            if not price_anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=price_anomalies["Date"], y=price_anomalies["Close"],
                    mode="markers", name="CEP price anomaly",
                    marker=dict(color=RED, size=8, symbol="circle-open", line=dict(width=2)),
                ))
            fig.update_layout(
                title="ETH-USD · Daily Candles & Next-Close Forecast",
                xaxis_title=None, yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
            )
            dark_chart(fig, 390)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            heading("Informasi ETH Terbaru / Latest ETH Information", "Observasi harian terbaru / Most recent daily observation")
            latest_info = pd.DataFrame({
                "Metric": ["Date", "Open", "High", "Low", "Close", "Volume", "Daily Change"],
                "Value": [
                    latest_date.strftime("%d %b %Y"), money(latest["Open"]), money(latest["High"]),
                    money(latest["Low"]), money(latest["Close"]), f"{latest_volume:,.0f}", f"{change_pct:+.2f}%"
                ]
            })
            st.dataframe(latest_info, use_container_width=True, hide_index=True, height=290)

        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            heading("Deteksi Anomali Harga / Price Anomaly Detection (CEP)", "Close di luar rata-rata bergulir ± 2 simpangan baku / Close outside rolling mean ± 2 SD")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cep_data["Date"], y=cep_data["Upper_Bound"], name="Upper Bound", line=dict(color=BLUE, dash="dash", width=1.2)))
            fig.add_trace(go.Scatter(x=cep_data["Date"], y=cep_data["Lower_Bound"], name="Lower Bound", line=dict(color=BLUE, dash="dash", width=1.2)))
            fig.add_trace(go.Scatter(x=cep_data["Date"], y=cep_data["Close"], name="Price", line=dict(color=TEAL, width=1.8)))
            if not price_anomalies.empty:
                fig.add_trace(go.Scatter(x=price_anomalies["Date"], y=price_anomalies["Close"], mode="markers", name="Anomaly", marker=dict(color=RED, size=8, symbol="x")))
            dark_chart(fig, 270)
            fig.update_layout(title="Price Anomalies")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            heading("Analisis Volume / Volume Analysis", "Menandai volume yang melampaui batas historis / Flags volume above its historical threshold")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sequence_data["Date"], y=sequence_data["Volume"], name="Volume", marker_color=AMBER))
            if not volume_anomalies.empty:
                fig.add_trace(go.Scatter(x=volume_anomalies["Date"], y=volume_anomalies["Volume"], mode="markers", name="Volume Anomaly", marker=dict(color=RED, size=8, symbol="circle-open", line=dict(width=2))))
            dark_chart(fig, 270)
            fig.update_layout(title="Daily Volume", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            heading("Deteksi Tren / Trend Detection (CEP)", "Tiga perubahan close berturut-turut dengan arah sama / Three consecutive changes in the same direction")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_data["Date"], y=trend_data["Close"], mode="lines", name="Price", line=dict(color=BLUE, width=1.7)))
            ups = trend_events[trend_events["Up_Sequence"]]
            downs = trend_events[trend_events["Down_Sequence"]]
            if not ups.empty:
                fig.add_trace(go.Scatter(x=ups["Date"], y=ups["Close"], mode="markers", name="Uptrend", marker=dict(color=TEAL, size=8, symbol="triangle-up")))
            if not downs.empty:
                fig.add_trace(go.Scatter(x=downs["Date"], y=downs["Close"], mode="markers", name="Downtrend", marker=dict(color=RED, size=8, symbol="triangle-down")))
            dark_chart(fig, 270)
            fig.update_layout(title="Consecutive Trend Events")
            st.plotly_chart(fig, use_container_width=True)

        b1, b2 = st.columns([1.25, 1], gap="medium")
        with b1:
            heading("Kejadian Pasar Terbaru / Recent Market Events", "Anomali dan tren terbaru yang memenuhi aturan / Recent events matching the rules")
            event_rows = []
            for _, r in price_anomalies.tail(6).iterrows():
                event_rows.append({"Date": r["Date"], "Event Type": "Price Anomaly", "Description": "Close outside rolling bounds", "Price (USD)": r["Close"]})
            for _, r in trend_events.tail(6).iterrows():
                event_rows.append({"Date": r["Date"], "Event Type": r["Trend_Event"], "Description": "Consecutive close-price sequence", "Price (USD)": r["Close"]})
            events_df = pd.DataFrame(event_rows)
            if not events_df.empty:
                events_df = events_df.sort_values("Date", ascending=False).head(8).copy()
                events_df["Date"] = pd.to_datetime(events_df["Date"]).dt.strftime("%Y-%m-%d")
                events_df["Price (USD)"] = events_df["Price (USD)"].map(lambda x: f"${x:,.2f}")
                st.dataframe(events_df, use_container_width=True, hide_index=True, height=270)
            else:
                st.info("Belum ada event yang memenuhi aturan CEP pada periode ini.")
        with b2:
            heading("Konfigurasi Model / Model Configuration", "Ringkasan konfigurasi LSTM yang dimuat / Loaded LSTM settings")
            m1, m2 = st.columns(2)
            m1.metric("Lookback Period", f"{lookback} days")
            m2.metric("Forecast Horizon", f"{horizon} day(s)")
            st.markdown(f"""
            <div style="border:1px solid {BORDER};border-radius:12px;background:{PANEL};padding:14px;margin-top:8px;">
              <div style="font-size:11px;color:{MUTED};">MODEL STATUS</div>
              <div style="font-size:16px;color:{TEAL};font-weight:750;margin-top:5px;">● Model loaded successfully</div>
              <div style="font-size:12px;color:{MUTED};margin-top:8px;">LSTM · One-feature closing-price input</div>
              <div style="font-size:11px;color:{MUTED};margin-top:6px;">Model memperkirakan harga penutupan ETH untuk periode berikutnya.</div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "LSTM Forecasting · Peramalan":
        st.markdown('<div class="hero"><div class="hero-kicker">Forecasting workspace</div><div class="hero-title">Price <span>Forecasting</span></div><div class="hero-copy">Bandingkan prediksi historis pada data uji dengan ramalan periode berikutnya dari model LSTM terlatih.</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Close", money(latest_price), delta=f"{change_pct:+.2f}%")
        c2.metric(f"Forecast in {horizon} day(s)", money(prediction_usd), delta=f"{(prediction_usd/latest_price-1)*100:+.2f}% vs harga terakhir / latest close" if latest_price else None)
        c3.metric("Forecast Date", forecast_date.strftime("%d %b %Y"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eth_data["Date"], y=eth_data["Close"], mode="lines", name="Actual Price", line=dict(color=TEAL, width=2.4)))
        fig.add_trace(go.Scatter(x=[latest_date, forecast_date], y=[latest_price, prediction_usd], mode="lines+markers", name="LSTM Forecast", line=dict(color=AMBER, width=3, dash="dash"), marker=dict(color=AMBER, size=9)))
        fig.update_layout(title="Actual Price vs Next LSTM Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
        dark_chart(fig, 480)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Prediksi menggunakan {lookback} observasi harga penutupan harian terakhir. Ini adalah estimasi model, bukan jaminan harga aktual.")
        with st.expander("Apa arti angka pada halaman ini?"):
            st.markdown(
                f"- **Latest close:** harga penutupan terbaru yang tersedia dari sumber data.\n"
                f"- **Forecast:** estimasi harga penutupan untuk {forecast_date.strftime('%d %b %Y')} berdasarkan pola pada {lookback} observasi terakhir.\n"
                "- **Perubahan terhadap harga terakhir:** selisih prediksi dibandingkan harga penutupan terbaru, dalam persen; bukan probabilitas bahwa harga akan naik atau turun."
            )
        st.caption("Fokus halaman ini adalah ramalan harga penutupan ETH untuk periode berikutnya. Evaluasi teknis model dan perbandingan model tidak ditampilkan pada tampilan pengguna.")

    elif page == "CEP Monitoring · Kejadian":
        st.markdown('<div class="hero"><div class="hero-kicker">Complex Event Processing</div><div class="hero-title">CEP <span>Monitoring</span></div><div class="hero-copy">Pemantauan intraday yang diperbarui berkala, disertai analisis anomali pada riwayat harga harian.</div></div>', unsafe_allow_html=True)
        st.markdown("### Pemantauan intraday / Intraday monitoring")
        st.caption("Grafik berikut menggunakan candle 1 menit yang tersedia dari Yahoo Finance dan diperbarui otomatis setiap 60 detik saat halaman ini aktif. Ketersediaan dan keterlambatan data bergantung pada penyedia; ini bukan feed transaksi langsung dari bursa.")

        @st.fragment(run_every="60s")
        def render_intraday_monitor():
            try:
                intraday = get_eth_intraday_data()
                intraday = intraday.tail(300).copy()
                last_bar = intraday.iloc[-1]
                received_at = pd.Timestamp.now()
                age_seconds = max(0, int((received_at - pd.Timestamp(last_bar["Datetime"])).total_seconds()))
                a, b, c = st.columns(3)
                a.metric("Harga candle terakhir / Latest price", money(last_bar["Close"]))
                b.metric("Observasi intraday / 1-min bars", f"{len(intraday):,}")
                c.metric("Usia observasi terakhir / Data age", f"{age_seconds // 60} min")
                st.caption(f"Waktu candle terakhir / Last candle: {pd.Timestamp(last_bar['Datetime']).strftime('%Y-%m-%d %H:%M:%S')} · waktu aplikasi lokal")

                window = 14
                intraday["Price_Mean"] = intraday["Close"].rolling(window, min_periods=window).mean().shift(1)
                intraday["Price_SD"] = intraday["Close"].rolling(window, min_periods=window).std().shift(1)
                intraday["Upper_Bound"] = intraday["Price_Mean"] + 2 * intraday["Price_SD"]
                intraday["Lower_Bound"] = intraday["Price_Mean"] - 2 * intraday["Price_SD"]
                intraday["Price_Anomaly"] = ((intraday["Close"] > intraday["Upper_Bound"]) | (intraday["Close"] < intraday["Lower_Bound"])) & intraday["Price_Mean"].notna()
                intraday["Volume_Mean"] = intraday["Volume"].rolling(window, min_periods=window).mean().shift(1)
                intraday["Volume_SD"] = intraday["Volume"].rolling(window, min_periods=window).std().shift(1)
                intraday["Volume_Upper"] = intraday["Volume_Mean"] + 2 * intraday["Volume_SD"]
                intraday["Volume_Anomaly"] = (intraday["Volume"] > intraday["Volume_Upper"]) & intraday["Volume_Upper"].notna()

                p1, p2 = st.columns(2)
                p1.metric("Anomali harga / Price flags", int(intraday["Price_Anomaly"].sum()))
                p2.metric("Lonjakan volume / Volume spikes", int(intraday["Volume_Anomaly"].sum()))
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(x=intraday["Datetime"], y=intraday["Close"], name="ETH price", line=dict(color=LILAC, width=1.8)))
                price_fig.add_trace(go.Scatter(x=intraday["Datetime"], y=intraday["Upper_Bound"], name="Upper bound", line=dict(color=AMBER, dash="dash", width=1)))
                price_fig.add_trace(go.Scatter(x=intraday["Datetime"], y=intraday["Lower_Bound"], name="Lower bound", line=dict(color=AMBER, dash="dash", width=1)))
                flags = intraday[intraday["Price_Anomaly"]]
                if not flags.empty:
                    price_fig.add_trace(go.Scatter(x=flags["Datetime"], y=flags["Close"], name="Price anomaly", mode="markers", marker=dict(color=RED, size=9, symbol="x")))
                price_fig.update_layout(title="Harga ETH per menit / ETH 1-minute price", yaxis_title="Harga (USD)")
                dark_chart(price_fig, 330)
                st.plotly_chart(price_fig, use_container_width=True)

                vol_fig = go.Figure()
                vol_fig.add_trace(go.Bar(x=intraday["Datetime"], y=intraday["Volume"], name="Volume", marker_color=AMBER))
                vol_flags = intraday[intraday["Volume_Anomaly"]]
                if not vol_flags.empty:
                    vol_fig.add_trace(go.Scatter(x=vol_flags["Datetime"], y=vol_flags["Volume"], name="Volume spike", mode="markers", marker=dict(color=RED, size=8, symbol="x")))
                vol_fig.update_layout(title="Volume per menit / 1-minute volume", yaxis_title="Volume")
                dark_chart(vol_fig, 280)
                st.plotly_chart(vol_fig, use_container_width=True)

                if age_seconds > 300:
                    st.warning("Data intraday terakhir berusia lebih dari 5 menit. Feed mungkin tertunda atau tidak sedang diperbarui.")
                else:
                    st.success("Data intraday tersedia. Pemeriksaan otomatis berjalan saat halaman ini aktif; keterlambatan penyedia tetap mungkin terjadi.")
                st.caption("Aturan pada panel ini menggunakan 14 candle sebelumnya dan batas ±2 simpangan baku. Panel ini terpisah dari CEP historis harian di bawah, serta tidak mengubah input model LSTM harian.")
            except Exception:
                st.warning("Data intraday sementara tidak tersedia dari Yahoo Finance. Riwayat harian di bawah tetap dapat digunakan. Coba lagi beberapa saat lagi.")

        render_intraday_monitor()
        st.divider()
        st.markdown("### Analisis historis / Historical CEP")
        with st.expander("Aturan deteksi / Detection rules"):
            st.markdown(
                "- **Anomali harga / Price anomaly:** harga penutupan berada di luar rentang acuan dari 14 hari sebelumnya.\n"
                "- **Anomali volume / Volume anomaly:** volume lebih tinggi dari batas atas berdasarkan 14 hari sebelumnya.\n"
                "- **Urutan harga–volume / Price–volume sequence:** anomali volume muncul dalam tiga observasi setelah anomali harga.\n\n"
                "Data yang ditampilkan adalah data historis harian Yahoo Finance, bukan streaming transaksi langsung. Data diperbarui ketika aplikasi mengambil data baru."
            )
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomali Harga / Price Anomalies", len(price_anomalies))
        c2.metric("Anomali Volume / Volume Anomalies", int(sequence_data["Volume_Anomaly"].sum()))
        c3.metric("Urutan Harga–Volume / Sequences", len(sequence_events))
        left, right = st.columns(2, gap="medium")
        with left:
            heading("Anomali Harga / Price Anomaly", "Titik merah muda menandai harga di luar rentang acuan / Pink markers show prices outside the reference range.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cep_data["Date"], y=cep_data["Upper_Bound"], name="Upper Bound", line=dict(color=AMBER, dash="dash")))
            fig.add_trace(go.Scatter(x=cep_data["Date"], y=cep_data["Lower_Bound"], name="Lower Bound", line=dict(color=AMBER, dash="dash")))
            fig.add_trace(go.Scatter(x=cep_data["Date"], y=cep_data["Close"], name="ETH Close", line=dict(color=LILAC, width=2)))
            if not price_anomalies.empty:
                fig.add_trace(go.Scatter(x=price_anomalies["Date"], y=price_anomalies["Close"], mode="markers", name="Anomaly", marker=dict(color=RED, size=9, symbol="x")))
            fig.update_layout(title="Harga dan batas acuan / Price and reference bounds", yaxis_title="Harga / Price (USD)")
            dark_chart(fig, 370)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            heading("Anomali Volume / Volume Anomaly", "Batang kuning menunjukkan volume; tanda merah muda menyorot lonjakan / Yellow bars show volume; pink markers highlight spikes.")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sequence_data["Date"], y=sequence_data["Volume"], name="Volume", marker_color=AMBER))
            if not volume_anomalies.empty:
                fig.add_trace(go.Scatter(x=volume_anomalies["Date"], y=volume_anomalies["Volume"], mode="markers", name="Anomaly", marker=dict(color=RED, size=9, symbol="x")))
            fig.update_layout(title="Volume harian dan lonjakan / Daily volume and spikes", yaxis_title="Volume")
            dark_chart(fig, 370)
            st.plotly_chart(fig, use_container_width=True)
        heading("Riwayat Anomali Harga / Recent Price Anomalies", "Observasi terbaru yang memenuhi aturan deteksi / Latest observations that triggered the rule.")
        if price_anomalies.empty:
            st.success("Tidak ada anomali harga berdasarkan aturan dan parameter saat ini.")
        else:
            table = price_anomalies[["Date", "Close", "Rolling_Mean", "Upper_Bound", "Lower_Bound"]].sort_values("Date", ascending=False).head(20).copy()
            table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(table, use_container_width=True, hide_index=True)
        heading("Urutan Harga–Volume / Price–Volume Sequences", "Anomali volume yang muncul hingga tiga observasi setelah anomali harga / Volume anomaly within three observations after a price anomaly.")
        if sequence_events.empty:
            st.info("Tidak ditemukan urutan anomali harga yang diikuti anomali volume dalam tiga hari observasi berikutnya.")
        else:
            table = sequence_events.sort_values("Tanggal Anomali Volume", ascending=False).copy()
            for col in ["Tanggal Anomali Harga", "Tanggal Anomali Volume"]:
                table[col] = pd.to_datetime(table[col]).dt.strftime("%Y-%m-%d")
            st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("Data: harga dan volume harian dari Yahoo Finance. Anomali menandai observasi yang memenuhi aturan deteksi; hasil ini bukan bukti manipulasi pasar atau rekomendasi investasi. / Daily Yahoo Finance data. Flags indicate rule-triggered observations, not market manipulation or investment advice.")
        with st.expander("Panduan warna / Color guide"):
            st.markdown(
                "- **Merah muda / Rose:** observasi yang ditandai sebagai anomali.\n"
                "- **Kuning / Amber:** batas acuan pada grafik harga dan batang volume.\n"
                "- **Ungu muda / Lilac:** garis harga penutupan ETH.\n\n"
                "Jumlah pada kartu adalah jumlah observasi yang memenuhi aturan dalam periode data yang sedang ditampilkan. Pilih tabel di bawah untuk melihat tanggal dan nilainya."
            )

    elif page == "Market Analytics · Analitik":
        st.markdown('<div class="hero"><div class="hero-kicker">Market analytics</div><div class="hero-title">Market <span>Analytics</span></div><div class="hero-copy">Jelajahi rentang harga harian, volume perdagangan, return, dan volatilitas bergulir pada data historis.</div></div>', unsafe_allow_html=True)
        analytics = eth_data.copy()
        analytics["Daily Return (%)"] = analytics["Close"].pct_change() * 100
        analytics["Rolling Volatility (7d)"] = analytics["Daily Return (%)"].rolling(7).std()
        c1, c2, c3 = st.columns(3)
        c1.metric("Highest High (1y)", money(analytics["High"].max()))
        c2.metric("Lowest Low (1y)", money(analytics["Low"].min()))
        c3.metric("Average Daily Volume", f"{analytics['Volume'].mean():,.0f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=analytics["Date"], y=analytics["Close"], mode="lines", name="Close", line=dict(color=LILAC, width=2)))
        fig.update_layout(title="Harga Penutupan Harian / Daily Closing Price", yaxis_title="Harga / Price (USD)")
        dark_chart(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2, gap="medium")
        with a:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=analytics["Date"], y=analytics["Volume"], name="Volume", marker_color=AMBER))
            fig.update_layout(title="Volume Perdagangan Harian / Daily Trading Volume", yaxis_title="Volume")
            dark_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analytics["Date"], y=analytics["Rolling Volatility (7d)"], mode="lines", name="7-day volatility", line=dict(color=TEAL, width=2)))
            fig.update_layout(title="Volatilitas Bergulir 7 Hari / 7-Day Rolling Volatility", yaxis_title="Simpangan baku return harian / Daily return SD (%)")
            dark_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Model & Methodology · Metode":
        st.markdown('<div class="hero"><div class="hero-kicker">Model details</div><div class="hero-title">Model <span>Information</span></div><div class="hero-copy">Transparansi sumber data, konfigurasi LSTM, proses evaluasi, dan batasan hasil peramalan.</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Architecture", "LSTM")
        c2.metric("Lookback", f"{lookback} days")
        c3.metric("Forecast Horizon", f"{horizon} day(s)")
        st.markdown("#### Artifact status")
        artifact_table = pd.DataFrame({
            "Component": ["LSTM forecasting model", "Price scaler", "Model configuration"],
            "Status": ["Available" if MODEL_PATH.exists() else "Missing",
                       "Available" if SCALER_PATH.exists() else "Missing",
                       "Available" if CONFIG_PATH.exists() else "Missing"],
            "Purpose": ["Generates the next closing-price estimate",
                        "Converts prices to and from the model's scale",
                        "Stores lookback and forecast settings"],
        })
        st.dataframe(artifact_table, use_container_width=True, hide_index=True)
        with st.expander("What do these model settings mean?"):
            st.markdown(
                f"- **Lookback ({lookback} days):** number of recent daily closing prices supplied to the model.\n"
                f"- **Forecast horizon ({horizon} day(s)):** how far ahead the displayed estimate targets.\n"
                "- **LSTM:** a recurrent neural network designed to learn patterns across sequences.\n"
                "- **Scaler:** puts price values on the scale used during model training, then converts the prediction back to USD."
            )
        with st.expander("Apa arti metrik evaluasi? / What do the metrics mean?"):
            st.markdown(
                "- **RMSE:** akar rata-rata kuadrat kesalahan; lebih sensitif terhadap kesalahan besar.\n"
                "- **MAE:** rata-rata nilai absolut selisih prediksi dan aktual.\n"
                "- **MAPE:** rata-rata kesalahan persentase absolut; perlu kehati-hatian jika nilai aktual mendekati nol.\n\n"
                "RMSE penalizes larger errors more heavily; MAE is the average absolute error; MAPE expresses errors as percentages."
            )
        st.markdown("#### Konfigurasi Model / Model Configuration")
        st.json(config)
        st.warning("Dashboard ini tidak menghitung RMSE/MAE/R² secara otomatis. Jangan menampilkan metrik evaluasi kecuali dihitung dari prediksi pada data uji yang terpisah.")

    elif page == "Data Explorer · Data":
        st.markdown('<div class="hero"><div class="hero-kicker">Historical data</div><div class="hero-title">Data <span>Explorer</span></div><div class="hero-copy">Periksa dan saring observasi harian ETH-USD yang diperoleh dari Yahoo Finance.</div></div>', unsafe_allow_html=True)
        min_date = eth_data["Date"].min().date()
        max_date = eth_data["Date"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        filtered = eth_data.copy()
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]
        st.caption(f"{len(filtered):,} observasi / observations · periode / period: {min_date} – {max_date}")
        display = filtered[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date", ascending=False).copy()
        display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True, height=520)
        st.download_button(
            "Download filtered data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="eth_usd_historical_data.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption("ETH_Watch · Harga harian untuk LSTM dan analisis historis; candle 1 menit untuk panel intraday jika tersedia. Pembaruan bergantung pada Yahoo Finance. Ramalan adalah estimasi, bukan saran investasi.")

except Exception as error:
    st.error("Data atau model belum dapat dimuat / Data or model could not be loaded.")
    # Keep machine-specific file paths and stack traces out of the public interface.
    if isinstance(error, FileNotFoundError):
        st.info("File model belum lengkap. Pastikan model LSTM, scaler, dan konfigurasi tersedia di folder models pada proyek. / Required model artifacts are missing from the project models folder.")
    elif isinstance(error, (ConnectionError, TimeoutError)):
        st.info("Koneksi ke sumber data mungkin bermasalah. Coba muat ulang beberapa saat lagi.")
    else:
        st.info("Periksa koneksi internet, dependensi Python, serta kelengkapan file model. Rincian teknis tidak ditampilkan pada dashboard.")
