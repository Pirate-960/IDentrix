# project_structure_script.py
"""
A utility script to automatically generate the directory structure for the
IDentrix person re-identification project.

This script is for convenience and is typically run only once at the beginning
of the project setup. It creates all the necessary folders and placeholder files.

Running this script is optional; you can also create the directories and
files manually.
"""

import os
import logging

# --- Basic logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Project Configuration ---
# 🔥 Amazing project name
project_name = "IDentrix"

# Define the complete directory and file structure.
# This dictionary maps directory paths to a list of files within them.
# An empty list means the directory should be created but will be empty initially.
structure = {
    # Main project directory containing all scripts
    project_name: [
        "model.py",
        "utils.py",
        "train.py",
        "evaluate.py",
        "app.py",
        "prepare_dataset.py",
        "requirements.txt"
    ],
    # Directory for saving model checkpoints
    os.path.join(project_name, "checkpoints"): [
        ".gitkeep" # A common convention to keep empty directories in git
    ],
    # Directory for the raw downloaded dataset
    os.path.join(project_name, "data", "raw"): [
        ".gitkeep"
    ],
    # Directory for gallery images (used for training and matching)
    os.path.join(project_name, "data", "gallery"): [
        ".gitkeep"
    ],
    # Directory for query images (used for evaluation and demo)
    os.path.join(project_name, "data", "query"): [
        ".gitkeep"
    ],
}

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Setting up project structure for '{project_name}'...")
    
    # Iterate over the defined structure
    for dir_path, files in structure.items():
        try:
            # Create the directory if it doesn't exist
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"✅ Created directory: {dir_path}")
            else:
                logging.info(f"⏩ Directory already exists: {dir_path}")

            # Create the files within the directory
            for file_name in files:
                full_path = os.path.join(dir_path, file_name)
                if not os.path.exists(full_path):
                    # Create an empty file with a placeholder comment
                    with open(full_path, "w") as f:
                        if file_name.endswith(".py"):
                            f.write(f"# {file_name}\n# This file is part of the {project_name} project.\n")
                        elif file_name == "requirements.txt":
                            f.write(f"# Dependencies for the {project_name} project\n")
                        # For other files like .gitkeep, just creating them is enough.
                    logging.info(f"✅ Created file: {full_path}")
                else:
                    logging.info(f"⏩ File already exists: {full_path}")

        except OSError as e:
            logging.error(f"❌ Failed to create directory {dir_path}. Error: {e}")
            break # Stop if a directory can't be created

    logging.info(f"\n🎉 Project '{project_name}' structure setup is complete.")