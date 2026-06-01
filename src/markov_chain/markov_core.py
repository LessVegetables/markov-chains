from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math
from pathlib import Path
import pickle
import random
import re
from typing import DefaultDict

try:
    from .reader import read_texts_from_folder
except ImportError:
    from reader import read_texts_from_folder


class MarkovError(Exception):
    pass


class EmptyFileError(MarkovError):
    pass


class TooShortTextError(MarkovError):
    pass


class UnknownStateError(MarkovError):
    pass


class ModelNotTrainedError(MarkovError):
    pass


TransitionTable = dict[tuple[str, ...], dict[str, int]]

DEFAULT_DATASET_FILES = [
    "turgenev_mumu.txt",
    "turgenev_dvoryanskoe_gnezdo.txt",
    "tolstoy_voina_i_mir.txt",
]


class Tokenizer:
    """Улучшенная токенизация текста с дополнительными опциями."""

    def __init__(
        self,
        lowercase: bool = False,
        normalize_yo: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        preserve_case_for_sentences: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.normalize_yo = normalize_yo
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.preserve_case_for_sentences = preserve_case_for_sentences

    def preprocess(self, text: str) -> str:
        """
        Подготавливает текст перед токенизацией.

        Параметры
        ----------
        text : str
            Исходный текст.

        Возвращает
        ----------
        str
            Очищенный и нормализованный текст.
        """
        if self.remove_urls:
            text = self._remove_urls(text)
        if self.remove_emails:
            text = self._remove_emails(text)
        if self.normalize_yo:
            text = self._normalize_yo(text)
        if self.lowercase and not self.preserve_case_for_sentences:
            text = text.lower()
        return text.strip()

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Удалить URL из текста."""
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        return re.sub(url_pattern, '', text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        """Удалить email-адреса из текста."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    @staticmethod
    def _normalize_yo(text: str) -> str:
        """Нормализовать русские Ё/ё к Е/е."""
        return text.replace('ё', 'е').replace('Ё', 'Е')

    def tokenize(self, text: str) -> list[str]:
        """
        Разбивает текст на токены.

        Метод выделяет слова и отдельные знаки препинания. Перед разбиением
        текст проходит базовую предобработку.

        Параметры
        ----------
        text : str
            Исходный текст.

        Возвращает
        ----------
        list[str]
            Список токенов.
        """
        text = self.preprocess(text)
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

        if self.lowercase and self.preserve_case_for_sentences:
            # Приводим к lowercase, но сохраняем заглавные в начале предложений
            tokens = self._smart_lowercase(tokens)

        return tokens

    def _smart_lowercase(self, tokens: list[str]) -> list[str]:
        """Умное приведение к lowercase: сохраняет заглавные в начале предложений."""
        result = []
        sentence_starters = {'.', '!', '?', '…', '...'}

        for i, token in enumerate(tokens):
            if i == 0:
                # Первый токен оставляем как есть (вероятно, заглавная)
                result.append(token)
            elif i > 0 and tokens[i - 1] in sentence_starters:
                # После знака конца предложения - оставляем заглавную
                result.append(token)
            else:
                # Остальное в lowercase
                result.append(token.lower())

        return result

    def split_sentences(self, text: str) -> list[str]:
        """
        Разбивает текст на предложения.

        Метод учитывает распространенные сокращения, чтобы не разрывать
        предложение после точек внутри них.

        Параметры
        ----------
        text : str
            Исходный текст.

        Возвращает
        ----------
        list[str]
            Список предложений без пустых строк.
        """
        # Защищаем сокращения
        abbreviations = r'(?:т\.е|т\.к|и\.т\.д|и\.т\.п|др|г|гг|ул|пр|тел|etc|vs|e\.g|i\.e|mr|mrs|ms|dr|prof)'
        text = re.sub(rf'({abbreviations})\.', r'\1<DOT>', text, flags=re.IGNORECASE)

        # Разбиваем по концам предложений
        pattern = r'(?<=[.!?…])\s+(?=[А-ЯA-Z«"\'\(])'
        sentences = re.split(pattern, text)

        # Восстанавливаем точки в сокращениях и очищаем
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]

        return sentences


class MarkovTextGenerator:
    __slots__ = (
        "order",
        "_tokenizer",
        "transitions",
        "_total_transitions",
        "_token_frequencies",
        "_source_token_count",
        "used_random_start",
    )

    def __init__(self, order: int = 2, tokenizer: Tokenizer | None = None) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self._tokenizer = tokenizer if tokenizer else Tokenizer()
        self.transitions: TransitionTable = {}
        self._total_transitions: int = 0
        self._token_frequencies: dict[str, int] = {}
        self._source_token_count: int = 0
        self.used_random_start = False

    def __repr__(self) -> str:
        status = "trained" if self.transitions else "untrained"
        return f"MarkovTextGenerator(order={self.order}, {status}, states={len(self.transitions)})"

    def clear(self) -> None:
        """
        Сбрасывает обученную модель к начальному состоянию.

        После вызова очищаются таблица переходов, статистика токенов и флаг
        использования случайного старта.

        Возвращает
        ----------
        None
        """
        self.transitions = {}
        self._total_transitions = 0
        self._token_frequencies = {}
        self._source_token_count = 0
        self.used_random_start = False

    def is_trained(self) -> bool:
        """
        Проверяет, обучена ли модель.

        Возвращает
        ----------
        bool
            True, если таблица переходов уже построена, иначе False.
        """
        return bool(self.transitions)

    def get_random_start_state(self, seed: int = None) -> tuple[str, ...]:
        """
        Возвращает случайное начальное состояние из обученной модели.

        Параметры
        ----------
        seed: int = None
            Sead для генрации.

        Возвращает
        ----------
        tuple[str, ...]
            Случайное состояние длиной ``order``.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        """
        if seed is not None:
            random.seed(seed)

        if not self.is_trained():
            raise ModelNotTrainedError("Модель не обучена.")
        return random.choice(list(self.transitions.keys()))

    def read_text(self, file_path: str | Path) -> str:
        """
        Читает текстовый файл с подбором кодировки.

        Метод пробует несколько популярных кодировок и возвращает содержимое
        первого успешно прочитанного файла.

        Параметры
        ----------
        file_path : str | Path
            Путь к текстовому файлу.

        Возвращает
        ----------
        str
            Содержимое файла без пробелов по краям.

        Исключения
        ----------
        FileNotFoundError
            Если файл не найден.
        EmptyFileError
            Если файл пустой или содержит только пробельные символы.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        text = ""
        encodings = ("utf-8", "utf-8-sig", "cp1251", "iso-8859-5", "koi8-r")
        for encoding in encodings:
            try:
                text = path.read_text(encoding=encoding).strip()
                break
            except UnicodeDecodeError:
                continue

        if not text:
            raise EmptyFileError("Файл пустой или содержит только пробелы.")
        return text

    def tokenize(self, text: str) -> list[str]:
        """
        Токенизирует текст и проверяет, что он подходит для модели.

        Параметры
        ----------
        text : str
            Исходный текст для обучения или анализа.

        Возвращает
        ----------
        list[str]
            Список токенов.

        Исключения
        ----------
        TooShortTextError
            Если количество токенов меньше или равно ``order``.
        """
        tokens = self._tokenizer.tokenize(text)
        if len(tokens) <= self.order:
            raise TooShortTextError(
                f"Текст слишком короткий для order={self.order}. Нужно больше {self.order} токенов."
            )
        return tokens

    def split_sentences(self, text: str) -> list[str]:
        """
        Разбивает текст на предложения через внутренний токенизатор.

        Параметры
        ----------
        text : str
            Исходный текст.

        Возвращает
        ----------
        list[str]
            Список предложений.
        """
        return self._tokenizer.split_sentences(text)

    def build_transitions(self, tokens: list[str]) -> TransitionTable:
        """
        Строит таблицу переходов цепи Маркова по списку токенов.

        Каждое состояние состоит из ``order`` токенов, а значениями являются
        возможные следующие токены и количество их появлений.

        Параметры
        ----------
        tokens : list[str]
            Токены обучающего текста.

        Возвращает
        ----------
        TransitionTable
            Таблица переходов модели.
        """
        table: DefaultDict[tuple[str, ...], DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._source_token_count = len(tokens)

        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            next_token = tokens[i + self.order]
            table[state][next_token] += 1

        self.transitions = {state: dict(next_map) for state, next_map in table.items()}
        self._total_transitions = sum(
            sum(counts.values()) for counts in self.transitions.values()
        )
        self._build_token_frequencies()
        return self.transitions

    def _build_token_frequencies(self) -> None:
        self._token_frequencies = {}
        for next_map in self.transitions.values():
            for token, count in next_map.items():
                self._token_frequencies[token] = self._token_frequencies.get(token, 0) + count

    def train_from_text(self, text: str) -> TransitionTable:
        """
        Обучает модель на переданном тексте.

        Параметры
        ----------
        text : str
            Обучающий текст.

        Возвращает
        ----------
        TransitionTable
            Построенная таблица переходов.

        Исключения
        ----------
        TooShortTextError
            Если текст слишком короткий для выбранного ``order``.
        """
        tokens = self.tokenize(text)
        return self.build_transitions(tokens)

    def train_from_file(self, file_path: str | Path) -> TransitionTable:
        """
        Обучает модель на тексте из файла.

        Параметры
        ----------
        file_path : str | Path
            Путь к обучающему текстовому файлу.

        Возвращает
        ----------
        TransitionTable
            Построенная таблица переходов.

        Исключения
        ----------
        FileNotFoundError
            Если файл не найден.
        EmptyFileError
            Если файл пустой.
        TooShortTextError
            Если текст слишком короткий для выбранного ``order``.
        """
        text = self.read_text(file_path)
        return self.train_from_text(text)

    def train_from_multiple_files(self, file_paths: Iterable[str | Path]) -> TransitionTable:
        """
        Обучает модель на нескольких текстовых файлах.

        Все тексты объединяются в один корпус, после чего по нему строится
        общая таблица переходов.

        Параметры
        ----------
        file_paths : Iterable[str | Path]
            Пути к текстовым файлам.

        Возвращает
        ----------
        TransitionTable
            Построенная таблица переходов.
        """
        all_text = []
        for path in file_paths:
            text = self.read_text(path)
            all_text.append(text)
        combined_text = " ".join(all_text)
        return self.train_from_text(combined_text)

    def load_default_dataset(self) -> TransitionTable:
        """
        Загружает стандартный корпус проекта и обучает на нем модель.

        Файлы берутся из папки ``data/processed`` по списку
        ``DEFAULT_DATASET_FILES``.

        Возвращает
        ----------
        TransitionTable
            Построенная таблица переходов.
        """
        data_folder = Path(__file__).resolve().parents[2] / "data" / "processed"
        text = ""

        for file_name in DEFAULT_DATASET_FILES:
            text += read_texts_from_folder(str(data_folder / file_name))

        return self.train_from_text(text)

    def get_transition_snapshot(self, limit: int = 20) -> list[dict[str, object]]:
        """
        Возвращает часть таблицы переходов в удобном для интерфейса виде.

        Параметры
        ----------
        limit : int, optional
            Максимальное количество состояний в снимке.

        Возвращает
        ----------
        list[dict[str, object]]
            Список состояний и их возможных переходов с вероятностями.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        ValueError
            Если ``limit`` меньше 1.
        """
        if not self.transitions:
            raise ModelNotTrainedError("Модель не обучена.")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        snapshot = []
        for state, next_map in list(self.transitions.items())[:limit]:
            total = sum(next_map.values())
            transitions = [
                {
                    "token": token,
                    "count": count,
                    "probability": round(count / total, 4),
                }
                for token, count in sorted(next_map.items(), key=lambda item: item[1], reverse=True)
            ]
            snapshot.append({"state": state, "transitions": transitions})
        return snapshot

    def generate(
        self,
        start_state: tuple[str, ...] | list[str],
        max_tokens: int = 30,
        temperature: float = 1.0,
        stop_tokens: set[str] | None = None,
        min_tokens: int = 20,
        extra_tokens: int = 20,
    ) -> str:
        """
        Генерирует текст из точного начального состояния.

        Параметры
        ----------
        start_state : tuple[str, ...] | list[str]
            Начальное состояние длиной ``order``.
        max_tokens : int, optional
            Максимальное количество новых токенов.
        temperature : float, optional
            Температура генерации. Чем выше значение, тем случайнее выбор
            следующего токена.
        stop_tokens : set[str] | None, optional
            Токены, на которых генерация может остановиться после ``min_tokens``.
        min_tokens : int, optional
            Минимальное количество новых токенов перед остановкой на
            ``stop_tokens``.
        extra_tokens : int, optional
            Дополнительный запас токенов, чтобы дать генерации шанс завершиться
            знаком конца предложения.

        Возвращает
        ----------
        str
            Сгенерированный текст.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        ValueError
            Если длина ``start_state`` не равна ``order``.
        UnknownStateError
            Если начальное состояние отсутствует в таблице переходов.
        """
        if not self.transitions:
            raise ModelNotTrainedError("Модель не обучена. Сначала вызовите train_from_text(...).")

        state = tuple(start_state)

        if len(state) != self.order:
            raise ValueError(f"Начальное состояние должно содержать ровно {self.order} токенов.")

        if state not in self.transitions:
            raise UnknownStateError(f"Неизвестное начальное состояние: {state}")

        if stop_tokens is None:
            stop_tokens = {".", "!", "?"}

        result = list(state)
        current_state = state

        for step in range(max_tokens + extra_tokens):
            next_options = self.transitions.get(current_state)
            if not next_options:
                break

            next_token = self._weighted_choice(next_options, temperature)
            result.append(next_token)

            generated_tokens = len(result) - len(state)

            if generated_tokens >= min_tokens and next_token in stop_tokens:
                break

            current_state = tuple(result[-self.order :])

        return self._join_tokens(result)

    def generate_from_seed(
        self,
        seed_text: str,
        max_tokens: int = 30,
        temperature: float = 1.0,
        min_tokens: int = 20,
        extra_tokens: int = 20,
    ) -> str:
        """
        Генерирует текст из строгого начального текста.

        Текст разбивается пробелами и используется как точное начальное
        состояние. Если такого состояния нет в модели, будет вызвана ошибка.

        Параметры
        ----------
        seed_text : str
            Начальный текст, количество слов в котором должно совпадать с
            ``order``.
        max_tokens : int, optional
            Максимальное количество новых токенов.
        temperature : float, optional
            Температура генерации.
        min_tokens : int, optional
            Минимальное количество новых токенов перед остановкой.
        extra_tokens : int, optional
            Дополнительный запас токенов для завершения предложения.

        Возвращает
        ----------
        str
            Сгенерированный текст.

        Исключения
        ----------
        UnknownStateError
            Если начальное состояние отсутствует в таблице переходов.
        """
        start_state = tuple(seed_text.split())
        return self.generate(
            start_state=start_state,
            max_tokens=max_tokens,
            temperature=temperature,
            min_tokens=min_tokens,
            extra_tokens=extra_tokens,
        )

    def generate_text(
        self,
        start_text: str | None = None,
        max_tokens: int = 30,
        temperature: float = 1.0,
        min_tokens: int = 20,
        stop_tokens: set[str] | None = None,
        strict_start: bool = False,
    ) -> str:
        """
        Генерирует текст через удобный API для консоли и GUI.

        Если ``start_text`` не задан или не найден в модели, метод выбирает
        случайное начальное состояние и выставляет ``used_random_start=True``.
        При ``strict_start=True`` неизвестный старт вызывает ошибку.

        Параметры
        ----------
        start_text : str | None, optional
            Начальный текст для генерации.
        max_tokens : int, optional
            Максимальное количество новых токенов.
        temperature : float, optional
            Температура генерации.
        min_tokens : int, optional
            Минимальное количество новых токенов перед остановкой.
        stop_tokens : set[str] | None, optional
            Токены, на которых генерация может остановиться.
        strict_start : bool, optional
            Если True, неизвестный стартовый текст вызовет ошибку. Если False,
            будет выбран случайный старт.

        Возвращает
        ----------
        str
            Сгенерированный текст.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        UnknownStateError
            Если старт неизвестен и ``strict_start=True``.
        ValueError
            Если ``max_tokens`` меньше 1 или ``min_tokens`` меньше 0.
        """
        if not self.transitions:
            raise ModelNotTrainedError("Модель не обучена. Сначала вызовите train_from_text(...).")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")

        state: tuple[str, ...]
        self.used_random_start = False
        if start_text:
            tokens = self._tokenizer.tokenize(start_text)
            state = tuple(tokens[-self.order :])
            if len(state) != self.order or state not in self.transitions:
                if strict_start:
                    raise UnknownStateError(f"Неизвестное начальное состояние: {state}")
                self.used_random_start = True
                state = self.get_random_start_state()
        else:
            self.used_random_start = True
            state = self.get_random_start_state()

        return self.generate(
            state,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
            min_tokens=min_tokens,
        )

    def _weighted_choice(self, options: dict[str, int], temperature: float) -> str:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        tokens = list(options.keys())

        if temperature == 1.0 or len(tokens) == 1:
            weights = list(options.values())
            return random.choices(tokens, weights=weights, k=1)[0]

        log_weights = [math.log(count) for count in options.values()]
        scaled_weights = [w / temperature for w in log_weights]
        max_weight = max(scaled_weights)
        exp_weights = [math.exp(w - max_weight) for w in scaled_weights]

        return random.choices(tokens, weights=exp_weights, k=1)[0]

    def get_entropy(self) -> float:
        """
        Считает среднюю энтропию переходов модели.

        Возвращает
        ----------
        float
            Средняя энтропия переходов в битах.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        """
        if not self.transitions:
            raise ModelNotTrainedError("Модель не обучена.")

        total_entropy = 0.0
        for next_map in self.transitions.values():
            total = sum(next_map.values())
            state_entropy = 0.0
            for count in next_map.values():
                prob = count / total
                if prob > 0:
                    state_entropy -= prob * math.log2(prob)
            total_entropy += state_entropy

        return total_entropy / len(self.transitions)

    def get_perplexity(self) -> float:
        """
        Считает perplexity модели на основе энтропии.

        Возвращает
        ----------
        float
            Значение perplexity.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        """
        entropy = self.get_entropy()
        return math.pow(2, entropy)

    def get_statistics(self) -> dict:
        """
        Возвращает основные статистики обученной модели.

        Возвращает
        ----------
        dict
            Словарь со статистиками: порядок модели, количество состояний,
            количество токенов, число переходов, ветвление, энтропия,
            perplexity и покрытие словаря.

        Исключения
        ----------
        ModelNotTrainedError
            Если модель еще не обучена.
        """
        if not self.transitions:
            raise ModelNotTrainedError("Модель не обучена.")

        unique_states = len(self.transitions)
        unique_tokens = len(self._token_frequencies)
        avg_branching = sum(len(v) for v in self.transitions.values()) / unique_states
        max_branching = max(len(v) for v in self.transitions.values())

        return {
            "order": self.order,
            "unique_states": unique_states,
            "unique_tokens": unique_tokens,
            "source_tokens": self._source_token_count,
            "total_transitions": self._total_transitions,
            "avg_branching_factor": round(avg_branching, 2),
            "max_branching_factor": max_branching,
            "entropy_bits": round(self.get_entropy(), 4),
            "perplexity": round(self.get_perplexity(), 4),
            "vocabulary_coverage": round(unique_tokens / max(self._source_token_count, 1) * 100, 2),
        }

    def save_model(self, file_path: str | Path) -> None:
        """
        Сохраняет обученную модель в файл.

        Параметры
        ----------
        file_path : str | Path
            Путь к файлу, куда нужно сохранить модель.

        Возвращает
        ----------
        None
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "order": self.order,
                    "transitions": self.transitions,
                    "_total_transitions": self._total_transitions,
                    "_token_frequencies": self._token_frequencies,
                    "_source_token_count": self._source_token_count,
                },
                f,
            )

    @classmethod
    def load_model(cls, file_path: str | Path) -> MarkovTextGenerator:
        """
        Загружает модель из файла.

        Параметры
        ----------
        file_path : str | Path
            Путь к файлу сохраненной модели.

        Возвращает
        ----------
        MarkovTextGenerator
            Восстановленный экземпляр генератора.

        Исключения
        ----------
        FileNotFoundError
            Если файл модели не найден.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(order=data["order"])
        instance.transitions = data["transitions"]
        instance._total_transitions = data["_total_transitions"]
        instance._token_frequencies = data["_token_frequencies"]
        instance._source_token_count = data.get("_source_token_count", 0)
        return instance

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([«(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([»)\]}])", r"\1", text)
        return text
