import os
import io
import json
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model from '{model_path}': {exc}") from exc


def infer_input_spec(model) -> Tuple[Tuple[int, int], int]:
    """Infer (height, width) and channels from a tf.keras model."""
    input_shape = getattr(model, "input_shape", None)
    if not input_shape or len(input_shape) != 4:
        # Fallback to common default
        return (224, 224), 3

    # Expecting (None, H, W, C)
    _, height, width, channels = input_shape
    if not isinstance(height, int) or not isinstance(width, int):
        height = 224 if not isinstance(height, int) else height
        width = 224 if not isinstance(width, int) else width
    if channels not in (1, 3):
        channels = 3
    return (height, width), channels


def load_class_names_from_json(json_path: str) -> Optional[List[str]]:
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept either list or dict mapping class->index
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # sort by index
                try:
                    sorted_items = sorted(data.items(), key=lambda kv: kv[1])
                    return [k for k, _ in sorted_items]
                except Exception:  # noqa: BLE001
                    return list(data.keys())
    except Exception:
        return None
    return None


def parse_manual_labels(raw_text: str) -> Optional[List[str]]:
    labels = [x.strip() for x in raw_text.split(",") if x.strip()]
    return labels if labels else None


def preprocess_image(img: Image.Image, target_size: Tuple[int, int], channels: int) -> np.ndarray:
    mode = "L" if channels == 1 else "RGB"
    img_converted = img.convert(mode)
    img_resized = img_converted.resize(target_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    if channels == 1:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def predict_batch(model, batch: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    preds = model.predict(batch, verbose=0)
    preds = np.asarray(preds)

    # Ensure 2D shape (N, K)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=-1)
    if preds.shape[1] == 1:
        # Binary sigmoid -> create 2-class probabilities
        p1 = preds[:, 0]
        preds = np.stack([1.0 - p1, p1], axis=1)
    else:
        # Best-effort: clip to [0,1] and normalize per row if sums not close to 1
        preds = np.clip(preds, 0.0, 1.0)
        row_sums = preds.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        preds = preds / row_sums
    return preds


def main() -> None:
    st.set_page_config(page_title="VisionCraft - Image Classifier", layout="wide")

    # Minimal modern styling
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(180deg, #f6f9ff 0%, #eef2ff 100%);}        
        .vc-hero {padding: 18px 22px; border-radius: 14px; background: linear-gradient(135deg,#0ea5e9, #6366f1); color: white;}
        .vc-hero h1 {margin: 0 0 6px 0; font-size: 1.7rem;}
        .vc-hero p {margin: 0; opacity: .95;}
        .vc-card {border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 16px; background: rgba(255,255,255,.9); box-shadow: 0 2px 14px rgba(0,0,0,.06); transition: transform .15s ease, box-shadow .15s ease;}        
        .vc-card:hover {transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,.10);}        
        .vc-muted {color: rgba(0,0,0,.6);}        
        @media (prefers-color-scheme: dark) {
          .stApp {background: linear-gradient(180deg, #0b1020 0%, #0a0f1f 100%);}        
          .vc-card { border-color: rgba(255,255,255,.08); background: rgba(20,20,30,.6); }
          .vc-muted { color: rgba(255,255,255,.7); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="vc-hero">
          <h1>VisionCraft — Image Classification</h1>
          <p>Upload image(s) and get instant predictions from your trained TensorFlow/Keras model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Supports JPG, JPEG, PNG. Model file: `model.h5` in app folder.")

    model_path = "model.h5"
    if not os.path.exists(model_path):
        st.error(
            "model.h5 not found in the project directory. Please place your trained model file "
            "named 'model.h5' in the same folder as this app and reload."
        )
        st.stop()

    with st.spinner("Loading model..."):
        try:
            model = load_model(model_path)
        except RuntimeError as exc:
            st.exception(exc)
            st.stop()

    # Force required preprocessing size (224x224). Channels inferred for correct mode.
    inferred_hw, channels = infer_input_spec(model)
    target_hw = (224, 224)

    with st.sidebar:
        st.subheader("Settings")
        st.markdown(
            f"Input size inferred from model: {target_hw[0]}x{target_hw[1]} with {channels} channel(s)."
        )

        default_labels = load_class_names_from_json("class_indices.json") or []
        manual_label_text = st.text_area(
            "Class labels (comma-separated). If empty, labels derived automatically.",
            value=", ".join(default_labels),
            height=80,
        )
        class_names = parse_manual_labels(manual_label_text) if manual_label_text else default_labels

        st.divider()
        st.markdown("""
        Tips:
        - Provide `class_indices.json` (list of labels or {label: index}) for nicer labels.
        - Leave labels blank to show generic class_0, class_1, ...
        - For binary models, positive class is the second bar.
        """)

        st.divider()
        st.markdown("### ℹ️ About App")
        st.markdown(
            "This app loads `model.h5`, preprocesses images to 224×224 (0–1), and predicts classes."
        )
        st.markdown("**👩‍💻 Made by Falguni Shinde**")

    tabs = st.tabs(["Single Image", "Batch Upload"])

    # Single Image tab
    with tabs[0]:
        st.markdown("Drag & drop or browse to upload a single image.")
        single = st.file_uploader(
            "Upload an image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="single_uploader",
        )
        if single is not None:
            try:
                img = Image.open(io.BytesIO(single.read()))
                images = [img]
                filenames = [single.name]
            except Exception:  # noqa: BLE001
                st.warning("Selected file is not a valid image.")
                images = []
                filenames = []

            if images:
                preprocessed = [preprocess_image(img, target_hw, channels) for img in images]
                batch = np.stack(preprocessed, axis=0)
                preds = predict_batch(model, batch)
                num_classes = preds.shape[1]
                labels = class_names if class_names and len(class_names) == num_classes else [f"class_{i}" for i in range(num_classes)]

                st.subheader("Result")
                for i, (img, fname) in enumerate(zip(images, filenames)):
                    top_idx = int(np.argmax(preds[i]))
                    top_label = labels[top_idx]
                    top_prob = float(preds[i, top_idx])
                    c_left, c_center, c_right = st.columns([1, 2, 1])
                    with c_center:
                        st.markdown("<div class='vc-card'>", unsafe_allow_html=True)
                        st.image(img, caption=fname, use_column_width=True)
                        st.markdown(f"**Prediction:** {top_label}")
                        st.success(f"✅ Predicted: {top_label}  •  Confidence: {top_prob:.2%}")
                        st.progress(min(max(top_prob, 0.0), 1.0), text=f"Confidence: {top_prob:.2%}")
                        try:
                            import pandas as pd
                            import altair as alt
                            df = pd.DataFrame({"class": labels, "probability": preds[i]})
                            chart = (
                                alt.Chart(df)
                                .mark_bar(cornerRadius=4)
                                .encode(
                                    x=alt.X("probability:Q", title="Confidence", scale=alt.Scale(domain=[0,1])),
                                    y=alt.Y("class:N", sort='-x', title="Class"),
                                    color=alt.Color("probability:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                    tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".2%")],
                                )
                                .properties(height=max(140, 24 * len(labels)))
                            )
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:  # noqa: BLE001
                            st.write({label: float(p) for label, p in zip(labels, preds[i])})
                        st.markdown("</div>", unsafe_allow_html=True)

    # Batch Upload tab
    with tabs[1]:
        st.markdown("Upload multiple images and compare predictions.")
        uploaded_files = st.file_uploader(
            "Upload image(s) (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader",
        )
        if uploaded_files:
            images: List[Image.Image] = []
            filenames: List[str] = []
            for uf in uploaded_files:
                try:
                    img = Image.open(io.BytesIO(uf.read()))
                    images.append(img)
                    filenames.append(uf.name)
                except Exception:  # noqa: BLE001
                    st.warning(f"Skipping file '{uf.name}' (not a valid image).")

            if images:
                preprocessed = [preprocess_image(img, target_hw, channels) for img in images]
                batch = np.stack(preprocessed, axis=0)
                preds = predict_batch(model, batch)
                num_classes = preds.shape[1]
                labels = class_names if class_names and len(class_names) == num_classes else [f"class_{i}" for i in range(num_classes)]

                st.subheader("Results")
                for i, (img, fname) in enumerate(zip(images, filenames)):
                    top_idx = int(np.argmax(preds[i]))
                    top_label = labels[top_idx]
                    top_prob = float(preds[i, top_idx])
                    c_left, c_center, c_right = st.columns([1, 2, 1])
                    with c_center:
                        st.markdown("<div class='vc-card'>", unsafe_allow_html=True)
                        st.image(img, caption=fname, use_column_width=True)
                        st.markdown(f"**Prediction:** {top_label}")
                        st.success(f"🎯 Predicted: {top_label}  •  Confidence: {top_prob:.2%}")
                        st.progress(min(max(top_prob, 0.0), 1.0), text=f"Confidence: {top_prob:.2%}")
                        try:
                            import pandas as pd
                            import altair as alt
                            df = pd.DataFrame({"class": labels, "probability": preds[i]})
                            chart = (
                                alt.Chart(df)
                                .mark_bar(cornerRadius=4)
                                .encode(
                                    x=alt.X("probability:Q", title="Confidence", scale=alt.Scale(domain=[0,1])),
                                    y=alt.Y("class:N", sort='-x', title="Class"),
                                    color=alt.Color("probability:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                    tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".2%")],
                                )
                                .properties(height=max(140, 24 * len(labels)))
                            )
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:  # noqa: BLE001
                            st.write({label: float(p) for label, p in zip(labels, preds[i])})
                        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <hr/>
    <div style="text-align:center;opacity:.8;">👩‍💻 Made by Falguni Shinde</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


