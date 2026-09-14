import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 224
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def distance_weighted_loss(y_true, y_pred, num_classes=5, distance_weight=0.5):
    y_true = tf.cast(y_true, tf.int32)
    cce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    class_indices = tf.range(num_classes, dtype=tf.float32)
    true_class_f = tf.cast(y_true, tf.float32)
    distances = tf.abs(tf.expand_dims(true_class_f, axis=1) - class_indices)
    expected_distance = tf.reduce_sum(y_pred * distances, axis=1)
    return cce + distance_weight * expected_distance

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
    "dr_model_ordinal_fixed.keras",
    custom_objects={"distance_weighted_loss": distance_weighted_loss}
)

model = load_model()
base_model = model.layers[0]

def preprocess_fundus(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_rgb[y:y+h, x:x+w]

    h, w = cropped.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)

def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name="top_activation"):
    conv_layer_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(last_conv_layer_name).output
    )
    with tf.GradientTape() as tape:
        conv_output = conv_layer_model(img_array)
        tape.watch(conv_output)
        x = conv_output
        for layer in model.layers[1:]:
            x = layer(x)
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy(), predictions.numpy()[0]

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original_uint8 = img_array.astype(np.uint8)
    return cv2.addWeighted(original_uint8, 1 - alpha, heatmap_colored, alpha, 0)

st.title("Explainable DR Screening")
st.write("Upload a retinal fundus image to get a diabetic retinopathy severity prediction with an explainability heatmap.")

uploaded_file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    processed = preprocess_fundus(img)
    img_array = np.expand_dims(processed, axis=0)

    heatmap, pred_class, probs = make_gradcam_heatmap(img_array, model, base_model)
    overlay = overlay_heatmap(processed, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(processed.astype(np.uint8), caption="Preprocessed Image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM Explainability", use_container_width=True)

    st.subheader(f"Prediction: {CLASS_NAMES[pred_class]}")
    st.write(f"Confidence: {probs[pred_class]*100:.1f}%")

    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})