from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math
from pathlib import Path
import pickle
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


class ModelNotTrainedError(MarkovError):
    pass


TransitionTable = dict[tuple[str, ...], dict[str, int]]


class MarkovTextGenerator:
    __slots__ = ("order", "transitions", "_total_transitions", "_token_frequencies")

    def __init__(self, order: int = 2) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: TransitionTable = {}
        self._total_transitions: int = 0
        self._token_frequencies: dict[str, int] = {}

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
        tokens = self.tokenize(text)
        return self.build_transitions(tokens)

    def train_from_file(self, file_path: str | Path) -> TransitionTable:
        text = self.read_text(file_path)
        return self.train_from_text(text)

    def train_from_multiple_files(self, file_paths: Iterable[str | Path]) -> TransitionTable:
        all_text = []
        for path in file_paths:
            text = self.read_text(path)
            all_text.append(text)
        combined_text = " ".join(all_text)
        return self.train_from_text(combined_text)

    def generate(
        self,
        start_state: tuple[str, ...] | list[str],
        max_tokens: int = 30,
        temperature: float = 1.0,
        stop_tokens: set[str] | None = None,
    ) -> str:
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

        for _ in range(max_tokens):
            next_options = self.transitions.get(current_state)
            if not next_options:
                break

            next_token = self._weighted_choice(next_options, temperature)
            result.append(next_token)

            if next_token in stop_tokens:
                break

            current_state = tuple(result[-self.order :])

        return self._join_tokens(result)

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
        entropy = self.get_entropy()
        return math.pow(2, entropy)

    def get_statistics(self) -> dict:
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
            "total_transitions": self._total_transitions,
            "avg_branching_factor": round(avg_branching, 2),
            "max_branching_factor": max_branching,
            "entropy_bits": round(self.get_entropy(), 4),
            "perplexity": round(self.get_perplexity(), 4),
            "coverage": round(unique_tokens / max(unique_tokens, 1) * 100, 2),
        }

    def save_model(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "order": self.order,
                    "transitions": self.transitions,
                    "_total_transitions": self._total_transitions,
                    "_token_frequencies": self._token_frequencies,
                },
                f,
            )

    @classmethod
    def load_model(cls, file_path: str | Path) -> MarkovTextGenerator:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(order=data["order"])
        instance.transitions = data["transitions"]
        instance._total_transitions = data["_total_transitions"]
        instance._token_frequencies = data["_token_frequencies"]
        return instance

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([«(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([»)\]}])", r"\1", text)
        return text
