from pathlib import Path


def read_texts_from_folder(folder_path: str) -> str:
    path = Path(folder_path)
    all_text = ""

    if path.is_file():
        return _read_single_file(path)

    for file_path in sorted(path.glob("*.txt")):
        all_text += _read_single_file(file_path)

    return all_text


def _read_single_file(file_path: Path) -> str:
    """Читает файл с обработкой ошибок кодировки."""
    encodings = ["utf-8", "utf-8-sig", "cp1251", "iso-8859-5", "koi8-r"]

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding) as file:
                return file.read() + "\n"
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Не удалось прочитать файл {file_path} ни в одной из кодировок: {encodings}")
