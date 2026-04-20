import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text generator based on Markov chains.")
    parser.add_argument("--order", type=int, default=3, help="Order of the Markov chain.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Maximum generated tokens.")
    parser.add_argument("--min-tokens", type=int, default=20, help="Minimum generated tokens before stopping.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--runs", type=int, default=3, help="Number of generated texts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_folder = Path(__file__).resolve().parents[2] / "data" / "processed"
    included_files = [
        "turgenev_mumu.txt",
        "turgenev_dvoryanskoe_gnezdo.txt",
        #"turgenev_nov.txt",
    ]

    text = ""
    for file_name in included_files:
        text += read_texts_from_folder(str(data_folder / file_name))

    generator = MarkovTextGenerator(order=args.order)
    generator.train_from_text(text)

    print("Файлы прочитаны.")
    print(f"Количество символов: {len(text)}")
    print(f"Количество состояний: {len(generator.transitions)}")

    stats = generator.get_statistics()
    print("\nСтатистика модели:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nПараметры запуска:")
    print(f"  order: {args.order}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  min_tokens: {args.min_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  runs: {args.runs}")
    print(f"  seed: {args.seed if args.seed is not None else 'random'}")

    for run_number in range(1, args.runs + 1):
        start_state = choose_start_state(generator)
        print(f"\nГенерация {run_number}:")
        print(
            generator.generate(
                start_state,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
            )
        )


if __name__ == "__main__":
    main()
