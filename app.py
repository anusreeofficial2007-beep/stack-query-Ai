import streamlit as st
from PIL import Image
import base64
from openai import OpenAI
import cv2
import numpy as np
import io
import datetime
import json
import os
from streamlit_image_comparison import image_comparison
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
}

.hero-banner {
    background: linear-gradient(135deg, #0B1120 0%, #132038 55%, #0F2A2C 100%);
    border: 1px solid #22304A;
    border-radius: 10px;
    padding: 28px 32px;
    margin-bottom: 26px;
}

.hero-banner h1 {
    margin: 0;
    font-size: 2.1rem;
    color: #E8EDF5;
}

.hero-banner p {
    margin: 6px 0 0 0;
    color: #8B9AB5;
    font-size: 0.98rem;
}

div[data-testid="stMetric"] {
    background: #131B2E;
    border: 1px solid #22304A;
    border-left: 3px solid #3ED6C4;
    border-radius: 8px;
    padding: 14px 18px;
}

button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #8B9AB5;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #3ED6C4;
}

.stButton > button, .stDownloadButton > button {
    background-color: #3ED6C4;
    color: #0B1120;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #34BFAE;
    color: #0B1120;
}

.streamlit-expanderHeader {
    font-family: 'Space Grotesk', sans-serif;
    color: #E8EDF5;
}
</style>

<div class="hero-banner">
<h1>🛰️ SatQuery AI</h1>
<p>Interactive AI assistant for satellite image analysis and change detection</p>
</div>
""", unsafe_allow_html=True)


try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

client = OpenAI()


HISTORY_FILE = "history.json"


def load_history():

    if os.path.exists(HISTORY_FILE):

        try:

            with open(HISTORY_FILE, "r") as f:
                return json.load(f)

        except (json.JSONDecodeError, ValueError):

            return []

    return []


def save_history(history):

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


if "history" not in st.session_state:
    st.session_state.history = load_history()


def image_to_base64(image):

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def detect_changes(before, after, threshold=60):

    before_np = np.array(before.convert("RGB"))
    after_np = np.array(after.convert("RGB"))

    before_np = cv2.resize(
        before_np,
        (after_np.shape[1], after_np.shape[0])
    )

    diff = cv2.absdiff(
        before_np,
        after_np
    )

    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    changed_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    highlighted = after_np.copy()
    highlighted[mask > 0] = heatmap_rgb[mask > 0]

    return highlighted, change_percentage


def generate_pdf_report(report_text, image_np):

    buffer = io.BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    y = page_height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "SatQuery AI - Change Detection Report")
    y -= 30

    c.setFont("Helvetica", 10)

    for line in report_text.split("\n"):

        if y < 120:
            c.showPage()
            y = page_height - 50
            c.setFont("Helvetica", 10)

        c.drawString(50, y, line[:100])
        y -= 14

    img_pil = Image.fromarray(image_np)
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    if y < 280:
        c.showPage()
        y = page_height - 50

    c.drawImage(
        img_reader,
        50,
        y - 260,
        width=320,
        height=240,
        preserveAspectRatio=True
    )

    c.save()
    buffer.seek(0)

    return buffer.getvalue()


tab1, tab2, tab3 = st.tabs([
    "🔍 Single Image Analysis",
    "🛰️ Change Detection",
    "📜 History"
])


with tab1:

    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Satellite Image",
            width="stretch"
        )

        query = st.text_input(
            "💬 Ask a question about this satellite image",
            value="What do you see in this satellite image?"
        )

        if st.button("🔍 Analyze Image"):

            if query:

                base64_image = image_to_base64(image)

                response = client.responses.create(

                    model="gpt-5.6-luna",

                    input=[
                        {
                            "role": "user",
                            "content": [

                                {
                                    "type": "input_text",
                                    "text": f"""
You are SatQuery AI, a satellite image analysis assistant.

Analyze the uploaded satellite image carefully.

User question:
{query}

Give a clear and concise answer.

Do not invent objects or locations that are not clearly visible.
"""
                                },

                                {
                                    "type": "input_image",
                                    "image_url":
                                        f"data:image/png;base64,{base64_image}"
                                }

                            ]
                        }
                    ]
                )

                st.success("🤖 AI Analysis")

                st.write(
                    response.output_text
                )

            else:

                st.warning(
                    "Please enter a question."
                )


with tab2:

    with st.expander("📍 Location Details (optional)"):

        loc_col1, loc_col2 = st.columns(2)

        with loc_col1:

            latitude = st.text_input(
                "Latitude",
                value=""
            )

        with loc_col2:

            longitude = st.text_input(
                "Longitude",
                value=""
            )

    with st.expander("⚙️ Detection Settings", expanded=True):

        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:

            threshold = st.slider(
                "🎚️ Change Sensitivity (lower = more sensitive)",
                min_value=10,
                max_value=150,
                value=60
            )

        with settings_col2:

            alert_limit = st.slider(
                "🔔 Alert me if change exceeds (%)",
                min_value=1.0,
                max_value=100.0,
                value=20.0
            )

    uploaded_images = st.file_uploader(
        "📷 Upload 2 or more images in chronological order "
        "(BEFORE → AFTER, or a full time-series)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi_images"
    )

    if uploaded_images and len(uploaded_images) >= 2:

        images = [Image.open(f) for f in uploaded_images]

        st.subheader("↔️ Before / After Comparison")

        image_comparison(
            img1=images[0],
            img2=images[-1],
            label1="Before",
            label2="After",
            width=700
        )

        change_query = st.text_input(
            "💬 Ask about the changes",
            value="What changes happened between these images?"
        )

        if st.button("🔍 Compare Images"):

            pair_results = []

            for i in range(len(images) - 1):

                pair_highlighted, pair_change = detect_changes(
                    images[i],
                    images[i + 1],
                    threshold=threshold
                )

                pair_results.append({
                    "pair": f"{i + 1} → {i + 2}",
                    "change_percentage": pair_change,
                    "highlighted": pair_highlighted
                })

            overall_change = sum(
                r["change_percentage"] for r in pair_results
            ) / len(pair_results)

            st.success("🤖 AI Change Analysis")

            if len(pair_results) > 1:

                st.subheader("📈 Change Trend Over Time")

                st.line_chart(
                    {"Change %": [r["change_percentage"] for r in pair_results]}
                )

            st.subheader("🗺️ Visual Change Map(s) — Heatmap Intensity")

            for r in pair_results:

                st.image(
                    r["highlighted"],
                    caption=f"Pair {r['pair']} — {r['change_percentage']:.2f}% change",
                    width="stretch"
                )

            location_text = ""

            if latitude and longitude:

                location_text = (
                    f"Location coordinates: "
                    f"Latitude {latitude}, Longitude {longitude}.\n"
                )

            before_base64 = image_to_base64(images[0])
            after_base64 = image_to_base64(images[-1])

            response = client.responses.create(

                model="gpt-5.6-luna",

                input=[
                    {
                        "role": "user",

                        "content": [

                            {
                                "type": "input_text",

                                "text": f"""
You are SatQuery AI.

Compare the FIRST and LAST satellite images in this time-series.

{location_text}
User question:
{change_query}

Image-processing algorithm detected approximately
{overall_change:.2f}% average visual difference across
{len(pair_results)} consecutive image pair(s).

Explain the major visible changes clearly.

Do not invent exact locations or objects that are not clearly visible.
"""
                            },

                            {
                                "type": "input_image",

                                "image_url":
                                    f"data:image/png;base64,{before_base64}"
                            },

                            {
                                "type": "input_image",

                                "image_url":
                                    f"data:image/png;base64,{after_base64}"
                            }

                        ]
                    }
                ]
            )

            st.write(
                response.output_text
            )

            st.metric(
                "📊 Overall Estimated Change",
                f"{overall_change:.2f}%"
            )

            if overall_change >= alert_limit:

                st.error(
                    f"🔔 ALERT: Change ({overall_change:.2f}%) has exceeded "
                    f"your alert threshold ({alert_limit:.2f}%)!"
                )

            else:

                st.success(
                    f"✅ Change ({overall_change:.2f}%) is within your "
                    f"alert threshold ({alert_limit:.2f}%)."
                )

            st.subheader("📥 Download Report")

            location_report_line = ""

            if latitude and longitude:

                location_report_line = (
                    f"Location: Latitude {latitude}, Longitude {longitude}\n"
                )

            report_text = f"""SatQuery AI - Change Detection Report
=========================================

{location_report_line}Question: {change_query}
Sensitivity Threshold: {threshold}
Alert Threshold: {alert_limit:.2f}%

Overall Estimated Change: {overall_change:.2f}%
Number of Image Pairs Compared: {len(pair_results)}

AI Analysis:
{response.output_text}
"""

            pdf_bytes = generate_pdf_report(
                report_text,
                pair_results[-1]["highlighted"]
            )

            download_col1, download_col2, download_col3 = st.columns(3)

            with download_col1:

                st.download_button(
                    label="📕 Download Report (PDF)",
                    data=pdf_bytes,
                    file_name="satquery_change_report.pdf",
                    mime="application/pdf"
                )

            with download_col2:

                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report_text,
                    file_name="satquery_change_report.txt",
                    mime="text/plain"
                )

            with download_col3:

                last_highlighted_pil = Image.fromarray(
                    pair_results[-1]["highlighted"]
                )
                image_buffer = io.BytesIO()
                last_highlighted_pil.save(image_buffer, format="PNG")

                st.download_button(
                    label="🖼️ Download Change Map (PNG)",
                    data=image_buffer.getvalue(),
                    file_name="change_map.png",
                    mime="image/png"
                )

            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": change_query,
                "change_percentage": float(overall_change),
                "location": f"{latitude}, {longitude}" if latitude and longitude else "-",
                "alert": bool(overall_change >= alert_limit)
            })

            save_history(st.session_state.history)


with tab3:

    st.header("📜 Change History")

    if st.session_state.history:

        for idx, record in enumerate(
            reversed(st.session_state.history), start=1
        ):
            st.write(
                f"*{idx}.* 🕒 {record['time']} — "
                f"📊 Change: {record['change_percentage']:.2f}% — "
                f"📍 {record.get('location', '-')} — "
                f"💬 {record['query']}"
                + (" — 🔔 ALERT" if record.get('alert') else "")
            )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            save_history(st.session_state.history)
            st.rerun()

    else:

        st.info(
            "No comparisons yet. Upload BEFORE/AFTER images "
            "and click Compare to start."
        )