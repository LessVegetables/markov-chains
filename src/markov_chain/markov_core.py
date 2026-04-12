from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import re
from typing import DefaultDict


class MarkovError(Exception):
    pass


class EmptyFileError(MarkovError):
    pass


class TooShortTextError(MarkovError):
    pass


class UnknownStateError(MarkovError):
    pass


TransitionTable = dict[tuple[str, ...], dict[str, int]]


class MarkovTextGenerator:
    def __init__(self, order: int = 2) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: TransitionTable = {}

    def read_text(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise EmptyFileError("Файл пустой или содержит только пробелы.")
        return text

    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        if len(tokens) <= self.order:
            raise TooShortTextError(
                f"Текст слишком короткий для order={self.order}. Нужно больше {self.order} токенов."
            )
        return tokens

    def build_transitions(self, tokens: list[str]) -> TransitionTable:
        table: DefaultDict[tuple[str, ...], DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            next_token = tokens[i + self.order]
            table[state][next_token] += 1

        self.transitions = {state: dict(next_map) for state, next_map in table.items()}
        return self.transitions

    def train_from_text(self, text: str) -> TransitionTable:
        tokens = self.tokenize(text)
        return self.build_transitions(tokens)

    def train_from_file(self, file_path: str | Path) -> TransitionTable:
        text = self.read_text(file_path)
        return self.train_from_text(text)

    def generate(self, start_state: tuple[str, ...] | list[str], max_tokens: int = 30) -> str:
        if not self.transitions:
            raise MarkovError("Модель не обучена. Сначала вызовите train_from_text(...).")

        state = tuple(start_state)

        if len(state) != self.order:
            raise ValueError(f"Начальное состояние должно содержать ровно {self.order} токенов.")

        if state not in self.transitions:
            raise UnknownStateError(f"Неизвестное начальное состояние: {state}")

        result = list(state)
        current_state = state

        for _ in range(max_tokens):
            next_options = self.transitions.get(current_state)
            if not next_options:
                break

            next_tokens = list(next_options.keys())
            weights = list(next_options.values())
            next_token = random.choices(next_tokens, weights=weights, k=1)[0]

            result.append(next_token)
            current_state = tuple(result[-self.order :])

        return self._join_tokens(result)

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([«(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([»)\]}])", r"\1", text)
        return text
