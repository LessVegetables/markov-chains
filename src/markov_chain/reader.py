from pathlib import Path


def read_texts_from_folder(folder_path: str):
    path = Path(folder_path)
    all_text = ""

    if path.is_file():
        with path.open("r", encoding="utf-8") as file:
            return file.read() + "\n"

    for file_path in sorted(path.glob("*.txt")):
        with file_path.open("r", encoding="utf-8") as file:
            all_text += file.read() + "\n"

    return all_text
