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
import random
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


# --- Helper Functions ---

def get_sequential_id(original_id: str) -> int:
    """
    Generates a consistent, sequential ID for a given original person ID.

    This function uses st.session_state to store a mapping of original IDs
    to a simple, incrementing counter. This ensures that the same person
    always receives the same sequential ID throughout the user's session.

    For example:
    - First unique person ('0002') gets ID 1.
    - Second unique person ('0751') gets ID 2.
    - A new image of person '0002' will also get ID 1.

    Args:
        original_id (str): The true ID of the person from the filename.

    Returns:
        int: The assigned sequential ID.
    """
    # Initialize the mapping dictionary and a counter in the session state.
    # This will persist for the entire user session.
    if 'sequential_id_map' not in st.session_state:
        st.session_state.sequential_id_map = {}
        st.session_state.id_counter = 1

    # Check if we have already assigned a sequential ID to this person.
    if original_id not in st.session_state.sequential_id_map:
        # If not, assign the next available ID from the counter.
        current_counter = st.session_state.id_counter
        st.session_state.sequential_id_map[original_id] = current_counter
        # Increment the counter for the next new person.
        st.session_state.id_counter += 1
    
    # Return the consistent, sequential ID from our map.
    return st.session_state.sequential_id_map[original_id]


def get_person_id_from_path(path: str) -> str:
    """
    Extracts the original person ID from the path and then uses the
    get_sequential_id function to convert it to a consistent, anonymized ID.

    Args:
        path (str): The file path of the image.

    Returns:
        str: The anonymized sequential ID as a string.
    """
    try:
        # Extract the original, true ID from the filename.
        filename = os.path.basename(path)
        original_id = filename.split('_')[0]
        
        # Get the sequential ID for display.
        sequential_id = get_sequential_id(original_id)
        
        # Return it as a string.
        return str(sequential_id)
    except IndexError:
        # Handle malformed filenames gracefully.
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