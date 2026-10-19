from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.inference import (
    DEFAULT_RESULT_DIR,
    FreshCheckClassifier,
)


st.set_page_config(
    page_title="FreshCheck",
    page_icon="🥬",
    layout="wide",
)


@st.cache_resource
def load_classifier():
    return FreshCheckClassifier()


def confidence_label(
    value: float,
) -> str:

    if value >= 0.90:
        return "High"

    if value >= 0.70:
        return "Moderate"

    return "Low"


def main():

    st.title(
        "🥬 FreshCheck"
    )

    st.caption(
        "Visual produce-quality classification "
        "using EfficientNet-B0."
    )

    st.info(
        "FreshCheck classifies visual appearance into "
        "Damaged, Dried, Old, Ripe, or Unripe. "
        "The output is a visual model prediction — "
        "not a food-safety or shelf-life assessment."
    )

    try:
        classifier = (
            load_classifier()
        )

    except Exception as error:

        st.error(
            "Model gagal dimuat."
        )

        st.exception(
            error
        )

        st.stop()

    with st.sidebar:

        st.header(
            "Model"
        )

        st.write(
            "**EfficientNet-B0**"
        )

        st.write(
            "Classes:"
        )

        for class_name in (
            classifier.class_names
        ):
            st.write(
                f"- {class_name}"
            )

        st.divider()

        st.write(
            f"Input size: "
            f"`{classifier.image_size} × "
            f"{classifier.image_size}`"
        )

        st.write(
            f"Device: "
            f"`{classifier.device}`"
        )

        st.divider()

        st.caption(
            "Every inference result is automatically "
            "saved to tests/results/."
        )

    uploaded_file = (
        st.file_uploader(
            "Upload produce image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
        )
    )

    if uploaded_file is None:

        st.markdown(
            """
### Supported visual-quality labels

- **Damaged**
- **Dried**
- **Old**
- **Ripe**
- **Unripe**

The model was evaluated on the mounted VegNet
quality-classification dataset used by this project.

Upload an image to start inference.
"""
        )

        return

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            "File gambar tidak dapat dibaca."
        )

        st.exception(
            error
        )

        return

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.subheader(
            "Input"
        )

        st.image(
            image,
            width="stretch",
        )

    with st.spinner(
        "Analyzing visual quality..."
    ):

        try:

            result = classifier.predict(
                image,
                source_name=uploaded_file.name,
            )

            result = classifier.save_result(
                original_image=image,
                result=result,
            )

        except Exception as error:

            st.error(
                "Inference gagal."
            )

            st.exception(
                error
            )

            return

    with right:

        st.subheader(
            "Prediction"
        )

        st.metric(
            "Visual Quality",
            result.predicted_class,
        )

        st.metric(
            "Confidence",
            f"{result.confidence:.2%}",
        )

        st.caption(
            "Confidence level: "
            f"{confidence_label(result.confidence)}"
        )

        if result.confidence < 0.70:

            st.warning(
                "Model confidence is relatively low. "
                "Interpret this prediction cautiously."
            )

    st.divider()

    st.subheader(
        "Class Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Class": list(
                result.probabilities.keys()
            ),
            "Probability": [
                value * 100
                for value
                in result.probabilities.values()
            ],
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    st.bar_chart(
        probability_df,
        x="Class",
        y="Probability",
    )

    st.dataframe(
        probability_df.style.format(
            {
                "Probability": "{:.2f}%"
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Top-3 Predictions"
    )

    top3_df = pd.DataFrame(
        [
            {
                "Rank": rank,
                "Class": item["class"],
                "Confidence": (
                    item["confidence"]
                    * 100
                ),
            }
            for rank, item
            in enumerate(
                result.top3,
                start=1,
            )
        ]
    )

    st.dataframe(
        top3_df.style.format(
            {
                "Confidence": "{:.2f}%"
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Saved Test Result"
    )

    st.success(
        "Inference result successfully saved."
    )

    st.code(
        str(
            Path(
                result.saved_image
            ).relative_to(
                ROOT
            )
        )
    )

    st.code(
        str(
            Path(
                result.saved_json
            ).relative_to(
                ROOT
            )
        )
    )

    history_path = (
        DEFAULT_RESULT_DIR
        / "prediction_history.csv"
    )

    if history_path.exists():

        with st.expander(
            "Prediction History"
        ):

            history = pd.read_csv(
                history_path
            )

            st.dataframe(
                history.tail(20),
                width="stretch",
                hide_index=True,
            )

    st.divider()

    st.markdown(
        """
### Interpretation Boundary

FreshCheck predicts **visual quality appearance**.

The model output should not be interpreted as:

- a food-safety guarantee;
- chemical freshness;
- remaining shelf life;
- microbial contamination;
- an edible / not-edible decision;
- universal quality for all fruits and vegetables.

The current model is an MVP evaluated on the
project's VegNet data configuration.
"""
    )


if __name__ == "__main__":
    main()