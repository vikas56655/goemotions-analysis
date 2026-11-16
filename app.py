"""Interactive local demo. Train the baseline before launching."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(page_title="Emotion Atlas", page_icon="🎭", layout="wide")
st.title("🎭 Emotion Atlas")
st.caption("Interpretable multi-label emotion detection · Google GoEmotions")

MODEL = Path("models/baseline.joblib")
META = Path("models/metadata.json")
REPORTS = Path("reports")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL)  # Load only locally trained, trusted artifacts.
    metadata = json.loads(META.read_text(encoding="utf-8"))
    return model, metadata


tab_predict, tab_analytics, tab_method = st.tabs(
    ["🔮 Emotion explorer", "📊 Research visualizations", "🧪 Methodology"]
)
with tab_predict:
    if not MODEL.exists() or not META.exists():
        st.info("No trained model found. Run: python -m src.train")
        st.code("python -m src.train\nstreamlit run app.py", language="bash")
    else:
        model, metadata = load_model()
        text = st.text_area("Write something to analyze",
                            "I'm so excited about this opportunity, but a little nervous.",
                            height=125)
        threshold = st.slider("Prediction threshold", 0.05, 0.95,
                              float(metadata["threshold"]), 0.05)
        if text.strip():
            probs = np.asarray(model.predict_proba([text]))[0]
            frame = pd.DataFrame({"Emotion": metadata["label_names"], "Probability": probs})
            frame = frame.sort_values("Probability", ascending=False)
            positive = frame.loc[frame.Probability >= threshold, "Emotion"].tolist()
            st.markdown("**Detected emotions:** " + (", ".join(positive) if positive else
                                                     "None above threshold"))
            fig = px.bar(frame.head(12).sort_values("Probability"),
                         x="Probability", y="Emotion", orientation="h",
                         color="Probability", color_continuous_scale="Teal",
                         range_x=[0, 1], title="Top predicted emotions")
            fig.add_vline(x=threshold, line_dash="dash", line_color="orange")
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=510)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Probabilities are independent per label, not a softmax distribution. "
                       "Outputs may include multiple emotions or none. "
                       "Scores should not be used to infer someone's psychological state.")
with tab_analytics:
    metrics_file = REPORTS / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        cols = st.columns(3)
        cols[0].metric("Test micro-F1", f'{metrics["test"]["micro_f1"]:.3f}')
        cols[1].metric("Test macro-F1", f'{metrics["test"]["macro_f1"]:.3f}')
        cols[2].metric("Threshold", f'{metrics["threshold"]:.2f}')
        st.caption(f'Test split: {metrics["sizes"]["test"]:,} examples. '
                   'Scores come from the local trained baseline.')
    else:
        st.info("Run training to generate real evaluation figures.")
    for name, title in [
        ("label_frequency.png", "Emotion frequency in training data"),
        ("cooccurrence.png", "Emotion co-occurrence"),
        ("per_label_f1.png", "Test F1 per label"),
        ("threshold_curve.png", "Validation threshold tuning"),
    ]:
        path = REPORTS / name
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
with tab_method:
    st.markdown("""
**Dataset:** GoEmotions simplified configuration, with its original train,
validation and test splits.

**Task:** Multi-label prediction across 27 emotions plus neutral.

**Baseline:** TF-IDF word unigrams/bigrams → 28 independent regularized logistic
regressions → sigmoid probabilities → threshold.

**Leakage control:** Fit vocabulary and classifier on training only. Choose the
decision threshold on validation only. Report hold-out test performance afterward.

**Caveats:** Reddit labels are subjective, label frequencies are uneven, and
linguistic performance may vary by demographic, dialect, and domain. Emotion
labels are not mental-health diagnoses.
""")
