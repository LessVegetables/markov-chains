"""Обучение и генерация текста по марковской модели n-грамм.

Состояние цепи — кортеж из ``order`` последовательных токенов; таблица переходов
хранит частоты следующего токена для каждого состояния. Токенизацию можно
настроить через ``Tokenizer`` (нормализация «ё», удаление URL/email, поведение
регистра).

Типовой сценарий:

    >>> from markov_core import MarkovTextGenerator, Tokenizer
    >>> gen = MarkovTextGenerator(order=2)
    >>> gen.train_from_text("Пример обучающего текста. Ещё одно предложение.")
    >>> text = gen.generate_text(max_tokens=40)

Исключения ``MarkovError`` и подклассы отражают ошибки обучения, генерации и ввода.
"""

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
    """Базовая ошибка ядра (обучение, генерация, ввод)."""


class EmptyFileError(MarkovError):
    """Файл пустой или после чтения не осталось значимого текста."""


class TooShortTextError(MarkovError):
    """В тексте недостаточно токенов для выбранного ``order`` (включая граничные случаи)."""


class UnknownStateError(MarkovError):
    """Начальное состояние отсутствует в таблице переходов (жёсткий режим или ``generate``)."""


class ModelNotTrainedError(MarkovError):
    """Операция требует обученной модели (таблица переходов пуста)."""


TransitionTable = dict[tuple[str, ...], dict[str, int]]
"""Таблица переходов: состояние из ``order`` токенов -> {следующий_токен: счётчик}."""


# Имена файлов в ``data/processed`` для :meth:`MarkovTextGenerator.load_default_dataset`.
DEFAULT_DATASET_FILES = [
    "turgenev_mumu.txt",
    "turgenev_dvoryanskoe_gnezdo.txt",
]


class Tokenizer:
    """Нормализация и разбиение текста на токены перед обучением марковской модели.

    Токенами считаются последовательности «словных» символов (``\\w+`` в Unicode)
    и одиночные знаки препинания. Опционально выполняется удаление URL и e-mail,
    подстановка «ё»→«е», а также режим приведения к нижнему регистру с сохранением
    заглавных букв в начале предложений.
    """

    def __init__(
        self,
        lowercase: bool = False,
        normalize_yo: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        preserve_case_for_sentences: bool = True,
    ) -> None:
        """
        Args:
            lowercase: Приводить регистр к нижнему (см. ``preserve_case_for_sentences``).
            normalize_yo: Заменить «ё»/«Ё» на «е»/«Е» до токенизации.
            remove_urls: Удалить URL из строки на этапе :meth:`preprocess`.
            remove_emails: Удалить адреса e-mail на этапе :meth:`preprocess`.
            preserve_case_for_sentences: Если ``lowercase=True``, оставлять заглавные
                после конца предложения и у первого токена (см. :meth:`_smart_lowercase`).
        """
        self.lowercase = lowercase
        self.normalize_yo = normalize_yo
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.preserve_case_for_sentences = preserve_case_for_sentences

    def preprocess(self, text: str) -> str:
        """Очистка и нормализация сырой строки перед :meth:`tokenize`.

        Args:
            text: Исходный текст.

        Returns:
            Строка после опциональных замен и ``strip()``; без разбиения на токены.
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
        """Разбить текст на список токенов после :meth:`preprocess`.

        Args:
            text: Входная строка.

        Returns:
            Список токенов (слова и отдельные знаки препинания).
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
        """Эвристически разбить текст на предложения (кириллица и латиница).

        Сокращения с точками временно маскируются, чтобы не резать по ним.

        Args:
            text: Исходный текст.

        Returns:
            Список непустых предложений без лишних пробелов по краям.
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
    """Марковский генератор текста по частотам *n-грамм* с фиксированным ``order``.

    После обучения в ``transitions`` лежит таблица: состояние из ``order``
    подряд идущих токенов отображается на счётчики возможных продолжений.
    Сэмплирование следующего токена учитывает эти счётчики; параметр
    ``temperature`` смягчает или усиливает пики распределения
    (см. :meth:`_weighted_choice`).

    Для ввода из файлов и сериализации см. :meth:`read_text`, :meth:`save_model`,
    :meth:`load_model`.
    """

    __slots__ = (
        "order",
        "_tokenizer",
        "transitions",
        "_total_transitions",
        "_token_frequencies",
        "_source_token_count",
    )

    def __init__(self, order: int = 2, tokenizer: Tokenizer | None = None) -> None:
        """
        Args:
            order: Длина контекста (число токенов в состоянии цепи). Должно быть >= 1.
            tokenizer: Кастомный :class:`Tokenizer`; иначе создаётся экземпляр по умолчанию.

        Raises:
            ValueError: Если ``order < 1``.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self._tokenizer = tokenizer if tokenizer else Tokenizer()
        self.transitions: TransitionTable = {}
        self._total_transitions: int = 0
        self._token_frequencies: dict[str, int] = {}
        self._source_token_count: int = 0

    def __repr__(self) -> str:
        status = "trained" if self.transitions else "untrained"
        return f"MarkovTextGenerator(order={self.order}, {status}, states={len(self.transitions)})"

    def clear(self) -> None:
        """Очистить таблицу переходов и вспомогательную статистику (как у нового экземпляра)."""
        self.transitions = {}
        self._total_transitions = 0
        self._token_frequencies = {}
        self._source_token_count = 0

    def is_trained(self) -> bool:
        """True, если в ``transitions`` есть хотя бы одно состояние."""
        return bool(self.transitions)

    def get_random_start_state(self) -> tuple[str, ...]:
        """Случайно выбрать состояние из ключей ``transitions``.

        Raises:
            ModelNotTrainedError: Если модель не обучена.
        """
        if not self.is_trained():
            raise ModelNotTrainedError("Модель не обучена.")
        return random.choice(list(self.transitions.keys()))

    def read_text(self, file_path: str | Path) -> str:
        """Прочитать файл в строку, перебирая типичные кодировки (UTF-8, CP1251, …).

        Args:
            file_path: Путь к текстовому файлу.

        Returns:
            Текст без ведущих и завершающих пробелов.

        Raises:
            FileNotFoundError: Если файл не найден.
            EmptyFileError: Если ни одна кодировка не подошла или файл пустой/пробельный.
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
        """Токенизировать текст и проверить, что токенов больше ``order``.

        Raises:
            TooShortTextError: Если длина последовательности токенов <= ``order``.
        """
        tokens = self._tokenizer.tokenize(text)
        if len(tokens) <= self.order:
            raise TooShortTextError(
                f"Текст слишком короткий для order={self.order}. Нужно больше {self.order} токенов."
            )
        return tokens

    def split_sentences(self, text: str) -> list[str]:
        """То же, что :meth:`Tokenizer.split_sentences` у встроенного токенизатора."""
        return self._tokenizer.split_sentences(text)

    def build_transitions(self, tokens: list[str]) -> TransitionTable:
        """Построить и записать в ``self.transitions`` таблицу частот переходов.

        Для каждого среза ``tokens[i : i + order]`` увеличивается счётчик
        следующего токена ``tokens[i + order]``. Предыдущее содержимое таблицы
        заменяется.

        Args:
            tokens: Уже сегментированный текст (длина должна быть > ``order``).

        Returns:
            Новая таблица переходов (тот же объект, что и ``self.transitions``).
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
        """Обучить модель на одной строке: токенизация и :meth:`build_transitions`."""
        tokens = self.tokenize(text)
        return self.build_transitions(tokens)

    def train_from_file(self, file_path: str | Path) -> TransitionTable:
        """Прочитать файл и обучить модель (:meth:`read_text` + :meth:`train_from_text`)."""
        text = self.read_text(file_path)
        return self.train_from_text(text)

    def train_from_multiple_files(self, file_paths: Iterable[str | Path]) -> TransitionTable:
        """Склеить тексты файлов через пробел и обучить модель одним вызовом."""
        all_text = []
        for path in file_paths:
            text = self.read_text(path)
            all_text.append(text)
        combined_text = " ".join(all_text)
        return self.train_from_text(combined_text)

    def load_default_dataset(self) -> TransitionTable:
        """Обучить на встроенном корпусе: каталог ``data/processed`` у корня репозитория и :data:`DEFAULT_DATASET_FILES`."""
        data_folder = Path(__file__).resolve().parents[2] / "data" / "processed"
        text = ""

        for file_name in DEFAULT_DATASET_FILES:
            text += read_texts_from_folder(str(data_folder / file_name))

        return self.train_from_text(text)

    def get_transition_snapshot(self, limit: int = 20) -> list[dict[str, object]]:
        """Первые ``limit`` состояний с отсортированными исходящими переходами и вероятностями.

        Удобно для отладки и UI (просмотр «куда можно пойти» из состояния).

        Args:
            limit: Сколько состояний вернуть (в современном Python порядок ключей
                ``dict`` соответствует порядку вставки при обучении).

        Returns:
            Список словарей с ключами ``state`` и ``transitions``; внутри — ``token``, ``count``, ``probability``.

        Raises:
            ModelNotTrainedError: Если модель пуста.
            ValueError: Если ``limit < 1``.
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
        """Продолжить текст из заданного *обученного* состояния длины ``order``.

        На каждом шаге выбирается следующий токен по весам из ``transitions``;
        при ``temperature != 1`` см. :meth:`_weighted_choice`. Остановка: нет
        исходящих рёбер, либо после ``min_tokens`` сгенерированных токенов
        встретился токен из ``stop_tokens``.

        Args:
            start_state: Ровно ``order`` токенов; должно быть ключом ``transitions``.
            max_tokens: Верхняя оценка числа *новых* токенов (логика цикла см. ниже).
            temperature: Положительное число; >1 — более «равномерный» выбор,
                <1 — ближе к жадному к частым токенам.
            stop_tokens: Множество токенов-остановов (по умолчанию конец предложения).
            min_tokens: Минимум сгенерированных токенов до допустимой остановки по ``stop_tokens``.
            extra_tokens: Доп. шаги после ``max_tokens`` в цикле (запас на длинные хвосты).

        Returns:
            Строка из склеенных токенов с правками пробелов у пунктуации (:meth:`_join_tokens`).

        Raises:
            ModelNotTrainedError: Нет таблицы переходов.
            ValueError: Неверная длина ``start_state``.
            UnknownStateError: ``start_state`` отсутствует в ``transitions``.
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
        """То же, что :meth:`generate`, но начальное состояние — пробельный ``split`` строки (без :class:`Tokenizer`).

        Длина ``seed_text.split()`` должна совпадать с ``order``; иначе будет
        :exc:`ValueError` или :exc:`UnknownStateError` из :meth:`generate`.

        Note:
            Для согласованности с обучением предпочтительнее :meth:`generate_text`,
            где старт парсится через :meth:`Tokenizer.tokenize`.
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
        """Высокоуровневая генерация: случайный или производный от строки старт.

        ``start_text`` токенизируется встроенным :class:`Tokenizer`; последние
        ``order`` токенов становятся состоянием. Если состояния нет в таблице и
        ``strict_start`` ложно, берётся :meth:`get_random_start_state`.

        Args:
            start_text: Начало фразы или ``None`` (полностью случайное состояние).
            max_tokens: Ограничение для :meth:`generate` (должно быть >= 1).
            temperature: Температура сэмплирования.
            min_tokens: Минимум новых токенов до остановки по ``stop_tokens``.
            stop_tokens: Пользовательские стоп-токены; ``None`` — поведение :meth:`generate`.
            strict_start: Если True, неизвестное состояние из ``start_text`` → :exc:`UnknownStateError`.

        Returns:
            Сгенерированная строка.

        Raises:
            ModelNotTrainedError: Модель не обучена.
            ValueError: Неверные ``max_tokens`` или ``min_tokens``.
            UnknownStateError: Неизвестный старт при ``strict_start=True``.
        """
        if not self.transitions:
            raise ModelNotTrainedError("Модель не обучена. Сначала вызовите train_from_text(...).")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")

        state: tuple[str, ...]
        if start_text:
            tokens = self._tokenizer.tokenize(start_text)
            state = tuple(tokens[-self.order :])
            if len(state) != self.order or state not in self.transitions:
                if strict_start:
                    raise UnknownStateError(f"Неизвестное начальное состояние: {state}")
                state = self.get_random_start_state()
        else:
            state = self.get_random_start_state()

        return self.generate(
            state,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
            min_tokens=min_tokens,
        )

    def _weighted_choice(self, options: dict[str, int], temperature: float) -> str:
        """Выбрать ключ из ``options`` пропорционально весам с учётом ``temperature``.

        При ``temperature == 1`` или одном ключе — сэмплирование по сырым счётчикам.
        Иначе — логарифм весов, деление на ``temperature`` и стабилизированный softmax.
        """
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
        """Средняя по состояниям энтропия исходящего распределения (в битах).

        Для каждого состояния считается :math:`H = -\\sum p \\log_2 p` по нормализованным
        счётчикам; возвращается среднее арифметическое по числу состояний.

        Raises:
            ModelNotTrainedError: Модель не обучена.
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
        """Возвращает :math:`2^{H}`, где :math:`H` — :meth:`get_entropy`."""
        entropy = self.get_entropy()
        return math.pow(2, entropy)

    def get_statistics(self) -> dict:
        """Агрегированные метрики обученной модели (состояния, ветвление, энтропия и пр.).

        Raises:
            ModelNotTrainedError: Модель не обучена.
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
        """Сериализовать order, переходы и счётчики в один pickle-файл.

        Warning:
            Формат pickle не безопасен для недоверенных файлов при загрузке.
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
        """Восстановить экземпляр из файла, записанного :meth:`save_model`.

        Warning:
            Не загружайте pickle из ненадёжных источников.

        Raises:
            FileNotFoundError: Файл отсутствует.
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
        """Склеить токены пробелом и убрать лишние пробелы у пунктуации и скобок."""
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([«(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([»)\]}])", r"\1", text)
        return text
