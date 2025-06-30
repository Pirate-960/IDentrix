import os

# 🔥 Amazing project name
project_name = "IDentrix"

# Directory and file structure
structure = {
    f"{project_name}/": [
        "model.py",
        "utils.py",
        "train.py",
        "evaluate.py",
        "app.py",
        "requirements.txt"
    ],
    f"{project_name}/checkpoints/": [
        "best_model.pth"
    ],
    f"{project_name}/data/gallery/": [],
    f"{project_name}/data/query/": []
}

# Create folders and files
for dir_path, files in structure.items():
    os.makedirs(dir_path, exist_ok=True)
    for file in files:
        full_path = os.path.join(dir_path, file)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                if file.endswith(".py"):
                    f.write(f"# {file}\n# This is part of the IDentrix person re-identification project.\n")
                elif file == "requirements.txt":
                    f.write("# Dependencies for the IDentrix project\n")
                elif file.endswith(".pth"):
                    pass  # Placeholder model file
print(f"✅ Project '{project_name}' structure created successfully.")
