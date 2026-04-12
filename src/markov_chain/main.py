from pathlib import Path
import random

from markov_core import MarkovTextGenerator
from reader import read_texts_from_folder


def choose_start_state(generator: MarkovTextGenerator) -> tuple[str, ...]:
    states = list(generator.transitions.keys())

    sentence_starts = [
        state
        for state in states
        if state[0] and state[0][0].isupper()
    ]

    if sentence_starts:
        return random.choice(sentence_starts)

    return random.choice(states)


def main() -> None:
    data_folder = Path(__file__).resolve().parents[2] / "data" / "processed"
    included_files = [
        "turgenev_mumu.txt",
        "turgenev_dvoryanskoe_gnezdo.txt",
        "turgenev_nov.txt",
    ]

    text = ""
    for file_name in included_files:
        text += read_texts_from_folder(str(data_folder / file_name))

    generator = MarkovTextGenerator(order=3)
    generator.train_from_text(text)
    start_state = choose_start_state(generator)

    print("Файлы прочитаны.")
    print(f"Количество символов: {len(text)}")
    print(f"Количество состояний: {len(generator.transitions)}")
    print(generator.generate(start_state, max_tokens=80))


if __name__ == "__main__":
    main()
