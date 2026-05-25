"""Streamlit dashboard — point it at a running FastAPI service.

Run:
    streamlit run src/diploma_nids/ui/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from collections import deque

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("DIPLOMA_API_URL", "http://127.0.0.1:8000")
REFRESH_SEC = float(os.environ.get("DIPLOMA_UI_REFRESH_SEC", "2.0"))


def fetch_alerts(n: int = 100) -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/alerts/recent", params={"n": n}, timeout=2.0)
        r.raise_for_status()
        return r.json().get("alerts", [])
    except requests.RequestException:
        return []


def fetch_info() -> dict:
    try:
        return requests.get(f"{API_URL}/info", timeout=2.0).json()
    except requests.RequestException:
        return {}


def main() -> None:
    st.set_page_config(page_title="diploma_nids", layout="wide")
    st.title("NIDS dashboard")
    st.caption(f"API: {API_URL}")

    info_panel = st.empty()
    metrics_panel = st.empty()
    table_panel = st.empty()
    chart_panel = st.empty()

    scores_history: deque[tuple[float, float]] = deque(maxlen=200)

    while True:
        info = fetch_info()
        with info_panel.container():
            cols = st.columns(3)
            cols[0].metric("Threshold", f"{info.get('threshold', 0.0):.4f}")
            cols[1].metric("Window", info.get("window", "—"))
            cols[2].metric("Model", info.get("model_version", "—"))

        alerts = fetch_alerts(200)
        n_total = len(alerts)
        n_high = sum(1 for a in alerts if a.get("severity") in ("high", "critical"))
        mean_score = sum(a.get("score", 0.0) for a in alerts) / max(1, n_total)

        with metrics_panel.container():
            mc = st.columns(3)
            mc[0].metric("Alerts (200 last)", n_total)
            mc[1].metric("High / Critical", n_high)
            mc[2].metric("Mean score", f"{mean_score:.3f}")

        if alerts:
            df = pd.DataFrame(alerts)
            table_panel.dataframe(df.tail(50), height=400, use_container_width=True)
            for a in alerts[-20:]:
                scores_history.append((float(a["ts"]), float(a["score"])))
            if scores_history:
                hdf = pd.DataFrame(scores_history, columns=["ts", "score"])
                chart_panel.line_chart(hdf.set_index("ts"))
        else:
            table_panel.info("No alerts yet — waiting for the service to score.")

        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
