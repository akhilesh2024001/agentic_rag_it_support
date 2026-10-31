from pathlib import Path
# current working directory 
root=Path(".")
# folder  create
folders=[
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "templates",
    "static",
    "uploads",
    "tests",
]
# Files to create
files = [
    "app/main.py",
    "ingest_sample_kb.py",
    "run.py",
    ".env"
]
# create folder
for folder in folders:
    folder_path=root/folder
    folder_path.mkdir(parents=True,exist_ok=True)
# create files

for file in files:
    (root/file).touch(exist_ok=True)
    print("done")
        