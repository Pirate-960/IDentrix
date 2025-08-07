# pages/1_🖼️_Gallery_Explorer.py
"""
Streamlit page for exploring the person re-identification gallery dataset.

This page provides a paginated view of all images in the gallery, allowing
users to browse the dataset and select any image to be used as a query
on the main search page.

Features:
- Paginated grid view to handle large datasets.
- Interactive buttons to send an image to the main search page.
"""

import streamlit as st
import os
from PIL import Image

# --- Helper Functions ---

@st.cache_data
def load_gallery_paths():
    """Loads and caches the file paths of all images in the gallery."""
    gallery_dir = './data/gallery'
    if not os.path.isdir(gallery_dir):
        return []
    # Sort paths for consistent ordering across runs
    return sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')])

def get_person_id_from_path(path):
    """Extracts the person ID from the image filename."""
    try:
        return int(os.path.basename(path).split('_')[0])
    except:
        return "N/A"

# --- Main Page UI and Logic ---

st.set_page_config(page_title="Gallery Explorer", layout="wide")
st.title("🖼️ Gallery Explorer")
st.info("Browse the gallery dataset. Click 'Use as Query' to send an image to the main search page.")

gallery_paths = load_gallery_paths()

if not gallery_paths:
    st.error("Gallery data not found. Please run `prepare_dataset.py`.")
    st.stop()

# --- Pagination Controls ---
st.sidebar.header("Pagination")
items_per_page = st.sidebar.selectbox("Images per page", [25, 50, 100], index=1)
total_pages = (len(gallery_paths) + items_per_page - 1) // items_per_page
page_number = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)

# --- Display Logic ---
start_index = (page_number - 1) * items_per_page
end_index = start_index + items_per_page
page_paths = gallery_paths[start_index:end_index]

st.write(f"Showing images {start_index + 1} to {min(end_index, len(gallery_paths))} of {len(gallery_paths)}.")

# Grid layout
cols = st.columns(5) 
for i, path in enumerate(page_paths):
    with cols[i % 5]:
        person_id = get_person_id_from_path(path)
        st.image(path, caption=f"ID: {person_id}", use_container_width=True)
        if st.button("Use as Query", key=f"query_{path}"):
            # Set the selected image in session state
            st.session_state.query_image = Image.open(path).convert('RGB')
            # Switch back to the main search page
            st.switch_page("app.py")