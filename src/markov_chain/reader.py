"""Чтение текстовых корпусов из файлов и каталогов.

Функции ориентированы на связку с ``markov_core``: перебор кодировок для
русскоязычных текстов и объединение нескольких ``.txt`` из одной папки.
"""

from pathlib import Path


def read_texts_from_folder(folder_path: str) -> str:
    """Прочитать один файл или все ``*.txt`` из каталога и вернуть общий текст.

    Если ``folder_path`` указывает на файл — возвращается его содержимое
    (плюс символ новой строки в конце). Если на каталог — файлы читаются в
    лексикографическом порядке имени и склеиваются.

    Args:
        folder_path: Путь к ``.txt``-файлу или к папке с такими файлами.

    Returns:
        Объединённая строка; для каталога без ``.txt`` — пустая строка.

    Raises:
        ValueError: Ни одна из перечисленных кодировок не смогла декодировать файл.
    """
    path = Path(folder_path)
    all_text = ""

    if path.is_file():
        return _read_single_file(path)

    for file_path in sorted(path.glob("*.txt")):
        all_text += _read_single_file(file_path)

    return all_text


def _read_single_file(file_path: Path) -> str:
    """Прочитать один текстовый файл, перебирая распространённые кодировки.

    Args:
        file_path: Путь к файлу.

    Returns:
        Содержимое и завершающий символ новой строки.

    Raises:
        ValueError: Если ни ``utf-8``, ни ``cp1251``, ни другие типичные для RU
            кодировки не подошли.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1251", "iso-8859-5", "koi8-r"]

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding) as file:
                return file.read() + "\n"
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Не удалось прочитать файл {file_path} ни в одной из кодировок: {encodings}")
