# app.py
"""
# IDentrix: Person Re-Identification Application

This is the main entry point for the IDentrix Streamlit application.
It serves as the user interface for performing person re-identification
queries using a trained model and a pre-computed gallery of embeddings.
Main Streamlit web application for the IDentrix Person Re-Identification Project.

This script serves as the primary user interface for running image-based queries.
It has been upgraded to include:
- TIER 1: Adjustable search parameters and detailed result information.
- TIER 2: Interactive search chaining and multiple input methods (file, URL).
- TIER 3: Explainable AI (XAI) using Grad-CAM to show model focus.

This app is part of a multi-page structure. Other pages provide additional
functionality like gallery exploration and video processing.

================================================================================
                            --- EXECUTION PATH ---
================================================================================
This project requires a specific sequence of steps to run correctly.
Follow this path from a clean state:

1.  SETUP THE ENVIRONMENT:
    - Ensure Python, Kaggle API, and W&B are configured.
    - Create a virtual environment and activate it.
    - Run: pip install -r requirements.txt
    - Run: wandb login

2.  PREPARE THE DATASET:
    - This script downloads the Market-1501 dataset from Kaggle.
    - Run: python prepare_dataset.py

3.  TRAIN THE MODEL:
    - This trains the DualBackboneNet and saves checkpoints.
    - Run: python train.py

4.  PRE-COMPUTE GALLERY EMBEDDINGS (CRITICAL FOR PERFORMANCE):
    - This step is essential to avoid a >1 hour startup time for the app.
    - It generates embeddings for all gallery images just once.
    - Ensure 'checkpoints/best_model.pth' exists from the previous step.
    - Run: python precompute_gallery.py

5.  RUN THE APPLICATION:
    - This command starts the interactive web server.
    - Run: streamlit run app.py
================================================================================
"""

# --- Standard Library Imports ---
import os
import logging
import requests
from io import BytesIO

# --- Third-Party Imports ---
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM

# --- Local Application Imports ---
from model import DualBackboneNet
from utils import preprocess_image, match_topk

# --- Logging Configuration ---
# Set up a basic logger to output informational messages to the console,
# helping to trace the application's execution flow and debug issues.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# --- Resource Loading Functions (Cached for Performance) ---

@st.cache_resource
def load_model():
    """
    Loads and caches the pre-trained DualBackboneNet model.

    Decorated with `@st.cache_resource` to load the model from disk only once,
    significantly speeding up subsequent app interactions. It handles model
    initialization, weight loading, and setting the model to evaluation mode.

    Returns:
        Optional[DualBackboneNet]: The loaded model, or None if the file is not found.
    """
    logging.info("Attempting to load the model...")
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        return None  # Return None for the calling function to handle gracefully
    
    model = DualBackboneNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()  # Set to evaluation mode
    
    logging.info("Model loaded successfully and cached.")
    return model


@st.cache_data
def load_gallery_assets():
    """
    Loads and caches gallery paths and their pre-computed embeddings.

    Decorated with `@st.cache_data` for efficient caching of data objects.
    This function is critical for the app's fast startup, as it avoids
    re-computing thousands of embeddings on every run.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple of sorted gallery paths and their
                                      corresponding embeddings. Returns empty
                                      objects on failure.
    """
    gallery_dir, embeddings_path = './data/gallery', './data/gallery_embeddings.npy'
    logging.info(f"Loading gallery assets...")
    
    if not os.path.exists(embeddings_path):
        return [], np.array([])
    
    # Sorting is crucial for consistency between paths and embeddings.
    gallery_paths = sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')])
    
    # Load the embeddings from the binary file. This is extremely fast.
    gallery_embs = np.load(embeddings_path)
    
    # Sanity check to prevent data mismatch errors.
    if len(gallery_paths) != len(gallery_embs):
        st.error("Mismatch between image count and embedding count. The gallery may be out of sync.")
        st.warning("Please re-run `python precompute_gallery.py`.", icon="🔄")
        return [], np.array([])
        
    logging.info(f"Gallery assets loaded and cached: {len(gallery_paths)} items.")
    return gallery_paths, gallery_embs


# --- Helper Functions ---

def get_person_id_from_path(path: str) -> str:
    """
    Extracts the person ID from the image filename based on Market-1501 convention.
    
    Args:
        path (str): The file path of the image.

    Returns:
        str: The extracted person ID as a string, or "N/A" on failure.
    """
    try:
        # Example: `0002_c1s1_000451_01.jpg` -> "0002"
        return os.path.basename(path).split('_')[0]
    except (ValueError, IndexError):
        return "N/A"


def generate_gradcam_overlay(model: torch.nn.Module, pil_image: Image.Image) -> Image.Image:
    """
    Generates a Grad-CAM heatmap to visualize model attention and overlays it on the image.

    This function provides explainability by showing which parts of the image the
    CNN backbone focused on to generate its feature embedding. It adapts the
    standard Grad-CAM method, which is typically used for classification, to
    work with an embedding model.

    Args:
        model (torch.nn.Module): The trained DualBackboneNet model.
        pil_image (Image.Image): The input image.

    Returns:
        Image.Image: The input image with the Grad-CAM heatmap overlay.
    """
    # 1. DEFINE THE TARGET LAYER
    # We target the last convolutional block of the ResNet backbone. This layer
    # contains high-level spatial features that are most relevant to the final
    # decision, making it an ideal target for Grad-CAM.
    target_layer = model.cnn.layer4
    
    # 2. PREPARE THE INPUT
    # The input PIL image must be preprocessed (resized, converted to a tensor,
    # and normalized) to match the format the model was trained on.
    input_tensor = preprocess_image(pil_image)

    # 3. INITIALIZE THE CAM EXTRACTOR
    # We use torchcam's GradCAM implementation. The `with` statement ensures that
    # the hooks registered on the model are properly removed after execution.
    with GradCAM(model, target_layer) as extractor:
        # 4. FORWARD PASS AND TARGET INDEX SELECTION (THE CRITICAL FIX)
        # First, get the model's output, which is the [1, 128] feature embedding.
        out = model(input_tensor)
        
        # The `torchcam` API requires a `class_idx` to know which output neuron
        # to backpropagate from. Since our Re-ID model doesn't have classes, we
        # must define a proxy. The standard technique is to use the index of the
        # neuron with the highest activation in the output embedding.
        # This asks the question: "Which parts of the image were most responsible
        # for the most prominent feature in the final embedding?"
        # `torch.argmax(out)` finds this index, and `.item()` extracts it as a
        # standard Python integer (e.g., 42).
        class_idx = torch.argmax(out).item()

        # 5. COMPUTE THE CLASS ACTIVATION MAP
        # We pass the required positional `class_idx` and the `scores` (which is
        # the model's output tensor) to the extractor.
        cams = extractor(class_idx, out)
        
        # The result `cams` is a list containing one CAM tensor. We select it,
        # remove the batch dimension (squeeze), and move it to the CPU for
        # processing with NumPy/Matplotlib.
        cam = cams[0].squeeze(0).cpu().numpy()

    # 6. VISUALIZE AND OVERLAY THE HEATMAP
    # To overlay the heatmap, the base image must be the same size as the
    # model's input (224x224).
    resized_pil_image = pil_image.resize((224, 224))
    
    # Use matplotlib to create a visually appealing heatmap from the raw CAM data.
    # The 'jet' colormap is a common choice for heatmaps.
    # We create a figure and axes but immediately turn the axes off to avoid
    # printing ticks, labels, or a border.
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.imshow(cam, cmap='jet')
    ax.axis('off')
    
    # Instead of saving to disk, we save the plot to an in-memory binary buffer.
    # This is a highly efficient way to convert a matplotlib plot to a PIL Image.
    # `bbox_inches='tight'` and `pad_inches=0` ensure there is no extra whitespace.
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    heatmap_pil = Image.open(buf)
    plt.close(fig)  # It's crucial to close the figure to prevent memory leaks.

    # The `Image.blend` function requires both images to have the same mode and size.
    # The `resized_pil_image` is in 'RGB' mode. The `heatmap_pil` image, saved
    # from a Matplotlib PNG, is likely in 'RGBA' (with a transparency channel).
    # We must explicitly convert the heatmap to 'RGB' before blending.
    heatmap_pil = heatmap_pil.convert('RGB')
    
    # Blend the original (resized) image with the heatmap. The alpha value
    # controls the transparency of the heatmap overlay.
    overlayed_image = Image.blend(resized_pil_image, heatmap_pil.resize(resized_pil_image.size), alpha=0.5)
    
    return overlayed_image


# --- Main Application UI and Logic ---

def main():
    """
    The main function that defines the Streamlit application's UI and logic.
    """
    
    # --- Page Configuration ---
    # This should be the first Streamlit command in the app.
    st.set_page_config(page_title="IDentrix Search", layout="wide", initial_sidebar_state="expanded")
    
    # --- State Initialization ---
    # `st.session_state` is used to persist variables across user interactions and pages.
    # Here, we initialize `query_image` if it doesn't already exist.
    if 'query_image' not in st.session_state:
        st.session_state.query_image = None

    # --- Load Global Resources ---
    # These are loaded once and cached for all users.
    model = load_model()
    gallery_paths, gallery_embs = load_gallery_assets()

    # Gracefully handle the case where critical resources are missing.
    if model is None or gallery_embs.size == 0:
        st.error("Critical resources failed to load. The application cannot continue.")
        st.warning("Please ensure the model (`best_model.pth`) and gallery embeddings (`gallery_embeddings.npy`) are correctly generated and placed.")
        st.stop()
        
    # --- Sidebar UI for Controls and Information ---
    with st.sidebar:
        st.title("🆔 IDentrix Controls")
        
        # An expandable section to provide context to the user.
        with st.expander("ℹ️ About this App", expanded=True):
            st.info(
                """
                This app demonstrates a deep learning model for Person Re-Identification.
                - **Upload an image** or provide a URL to find matches.
                - **Use the Gallery Explorer** to browse the dataset.
                - **Try the Video Processor** for tracking in video clips.
                """
            )
        
        # TIER 1 FEATURE: User-adjustable search parameters.
        st.header("⚙️ Search Parameters")
        k_results = st.slider("Number of Matches (k)", min_value=1, max_value=20, value=5, step=1)
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        
        # TIER 3 FEATURE: Toggle for Explainable AI.
        st.header("🔬 Explainability (XAI)")
        show_gradcam = st.checkbox("Show Model Focus (Grad-CAM)")
        if show_gradcam:
            st.caption("Grad-CAM visualizes the focus of the model's CNN backbone (ResNet).")

    # --- Main Page Content ---
    st.title("🔍 Person Re-Identification Search")

    # TIER 2 FEATURE: Multiple ways to provide an input image.
    st.subheader("1. Provide a Query Image")
    input_tabs = st.tabs(["📤 Upload File", "🔗 From URL"])

    # Input Method 1: File Upload
    with input_tabs[0]:
        uploaded_file = st.file_uploader(
            "Select an image file.", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed"
        )
        if uploaded_file:
            # When a file is uploaded, store it in the session state to trigger a search.
            st.session_state.query_image = Image.open(uploaded_file).convert('RGB')
    
    # Input Method 2: URL
    with input_tabs[1]:
        url = st.text_input("Or paste an image URL here and press Enter.")
        if url:
            try:
                # Fetch the image from the URL and store it in the session state.
                response = requests.get(url)
                st.session_state.query_image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                st.error(f"Could not load image from URL: {e}")

    st.divider()

    # This block executes only if a query image exists in the session state.
    if st.session_state.query_image:
        query_image = st.session_state.query_image
        
        # Create a two-column layout for the query and results.
        query_col, results_col = st.columns([1, 4])
        
        with query_col:
            st.subheader("2. Your Query")
            st.image(query_image, caption="Current Query Image", use_container_width=True)
            # Display Grad-CAM if the user has enabled it.
            if show_gradcam:
                with st.spinner("Analyzing query focus..."):
                    gradcam_overlay = generate_gradcam_overlay(model, query_image)
                    st.image(gradcam_overlay, caption="Query Grad-CAM", use_container_width=True)

        with results_col:
            st.subheader(f"3. Top {k_results} Matches")
            
            with st.spinner("Analyzing image and searching gallery..."):
                # Perform inference on the query image to get its embedding.
                with torch.no_grad():
                    query_embedding = model(preprocess_image(query_image)).detach().numpy()[0]
                
                # Find the top-k matches from the pre-computed gallery.
                top_paths, top_scores = match_topk(query_embedding, gallery_embs, gallery_paths, k=k_results)

            # TIER 1 FEATURE: Filter results by the similarity threshold.
            strong_matches = [(p, s) for p, s in zip(top_paths, top_scores) if s >= similarity_threshold]
            weak_matches = [(p, s) for p, s in zip(top_paths, top_scores) if s < similarity_threshold]

            if not strong_matches and not weak_matches:
                st.info("No matches found for the given criteria.")
            else:
                # Display strong matches with a success message.
                if strong_matches:
                    st.success(f"Found {len(strong_matches)} matches above {similarity_threshold:.2f} similarity.")
                    # Create a dynamic grid for the results.
                    result_cols = st.columns(len(strong_matches))
                    for i, (path, score) in enumerate(strong_matches):
                        with result_cols[i]:
                            person_id = get_person_id_from_path(path) # TIER 1: Show Person ID
                            st.image(path, use_container_width=True, caption=f"ID: {person_id} | Score: {score:.3f}")
                            
                            # TIER 2: Interactive search chaining button.
                            if st.button("🔍 Search with this", key=f"search_{path}"):
                                st.session_state.query_image = Image.open(path).convert('RGB')
                                st.rerun() # Re-run the script to perform the new search.
                            
                            # TIER 3: Display Grad-CAM for the match.
                            if show_gradcam:
                                with st.spinner("Analyzing match..."):
                                    match_overlay = generate_gradcam_overlay(model, Image.open(path).convert('RGB'))
                                    st.image(match_overlay, caption="Match Grad-CAM", use_container_width=True)
                
                st.divider()
                
                # Display weak matches with a warning message.
                if weak_matches:
                    st.warning(f"Found {len(weak_matches)} potential matches below the threshold.")
                    result_cols = st.columns(len(weak_matches))
                    for i, (path, score) in enumerate(weak_matches):
                         with result_cols[i]:
                            person_id = get_person_id_from_path(path)
                            st.image(path, use_container_width=True, caption=f"ID: {person_id} | Score: {score:.3f}")


if __name__ == "__main__":
    # This is the entry point of the application.
    main()