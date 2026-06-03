import json
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
from PIL import Image

IMG_SIZE = 64
MODEL_PATH = "models/model.keras"
LABELS_PATH = "labels.json"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def segment_characters(gray):
    """
    Segment handwritten word into individual characters.
    """

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w > 8 and h > 8:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    segments = []

    for x, y, w, h in boxes:

        char_img = gray[y:y+h, x:x+w]

        pad = 12

        char_img = cv2.copyMakeBorder(
            char_img,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=255
        )

        char_img = cv2.resize(
            char_img,
            (IMG_SIZE, IMG_SIZE)
        )

        char_img = cv2.bitwise_not(char_img)

        segments.append(char_img)

    return segments


def predict_char(model, labels, img_arr):

    x = img_arr.astype("float32") / 255.0
    x = x[np.newaxis, ..., np.newaxis]

    probs = model.predict(
        x,
        verbose=0
    )[0]

    idx = int(np.argmax(probs))

    top3_idx = np.argsort(probs)[-3:][::-1]

    top3 = [
        (
            labels[str(i)],
            float(probs[i])
        )
        for i in top3_idx
    ]

    return labels[str(idx)], float(probs[idx]), top3


def main():

    st.set_page_config(
        page_title="Bangla OCR",
        page_icon="বা",
        layout="wide"
    )

    st.title("বাংলা হস্তলিখিত শব্দ চেনা")
    st.caption(
        "Draw a Bangla handwritten word and click Recognize."
    )

    try:
        model = load_model()
        labels = load_labels()

    except Exception as e:
        st.error(
            f"Model not found.\n\nTrain first using:\npython train.py\n\n{e}"
        )
        return

    canvas = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=220,
        width=800,
        drawing_mode="freedraw",
        key="canvas"
    )

    col1, col2 = st.columns(2)

    with col1:
        recognize = st.button(
            "🔍 Recognize Word",
            use_container_width=True
        )

    with col2:
        clear = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

        if clear:
            st.rerun()

    if recognize:

        if canvas.image_data is None:
            st.warning("Draw something first.")
            return

        img_array = canvas.image_data.astype(np.uint8)

        gray = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGBA2GRAY
        )

        if np.mean(gray) > 250:
            st.warning("Canvas is empty.")
            return

        segments = segment_characters(gray)

        if len(segments) == 0:
            st.warning(
                "No characters detected. Try writing larger."
            )
            return

        results = []

        for seg in segments:
            pred, conf, top3 = predict_char(
                model,
                labels,
                seg
            )

            results.append(
                {
                    "char": pred,
                    "conf": conf,
                    "top3": top3,
                    "img": seg
                }
            )

        word = "".join(
            r["char"]
            for r in results
        )

        st.divider()

        st.success(
            f"Recognized Word: {word}"
        )

        st.write(
            f"Characters Detected: {len(results)}"
        )

        cols = st.columns(len(results))

        for col, result in zip(cols, results):

            with col:

                display_img = Image.fromarray(
                    cv2.bitwise_not(
                        result["img"]
                    )
                )

                st.image(
                    display_img,
                    width=90
                )

                st.markdown(
                    f"### {result['char']}"
                )

                st.write(
                    f"Confidence: {result['conf']*100:.1f}%"
                )

                with st.expander("Top 3"):

                    for c, p in result["top3"]:

                        st.write(
                            f"{c} ({p*100:.1f}%)"
                        )


if __name__ == "__main__":
    main()