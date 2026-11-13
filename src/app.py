
import streamlit as st
import torch
from torch import nn
from torchvision import models
from PIL import Image

from common import (
    MODEL_PATH,
    CLASSES_PATH,
    IMAGE_SIZE,
    DEVICE,
    get_transforms,
    load_class_names,
)

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "'python src/train_model.py' first."
        )

    class_names = load_class_names()

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel, len(class_names)
    )

    state_dict = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, class_names


st.title("♻️ EcoSort AI")
st.write(
    "Upload a waste image to classify it into one of the trained waste categories."
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            model, class_names = load_model()
            image_tensor = get_transforms()(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            confidence, predicted_index = torch.max(probabilities, dim=0)
            predicted_class = class_names[predicted_index.item()]

        st.subheader(f"Prediction: {predicted_class.title()}")
        st.metric("Model confidence", f"{confidence.item():.1%}")

        st.write("Class probabilities")
        for index, name in enumerate(class_names):
            st.write(f"**{name.title()}** — {probabilities[index].item():.1%}")
            st.progress(float(probabilities[index].item()))

        st.caption(
            "Confidence is the model's estimated probability, not a guarantee "
            "that the classification is correct."
        )

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not process this image: {error}")
