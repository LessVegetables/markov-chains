import tempfile
from pathlib import Path

import pytest

from markov_core import (
    EmptyFileError,
    MarkovTextGenerator,
    ModelNotTrainedError,
    Tokenizer,
    TooShortTextError,
    UnknownStateError,
)

"""Базовые тесты для markov_core.py."""

class TestInitialization:
    """Тесты инициализации генератора."""

    def test_default_order(self) -> None:
        gen = MarkovTextGenerator()
        assert gen.order == 2
        assert not gen.is_trained()

    def test_custom_order(self) -> None:
        gen = MarkovTextGenerator(order=3)
        assert gen.order == 3

    def test_invalid_order(self) -> None:
        with pytest.raises(ValueError, match="order must be >= 1"):
            MarkovTextGenerator(order=0)

    def test_repr_untrained(self) -> None:
        gen = MarkovTextGenerator(order=3)
        assert "order=3" in repr(gen)
        assert "untrained" in repr(gen)

    def test_repr_trained(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")
        assert "trained" in repr(gen)
        assert "states=" in repr(gen)


class TestTokenization:
    """Тесты токенизации текста."""

    def test_basic_tokenization(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = gen.tokenize("the cat sat")
        assert tokens == ["the", "cat", "sat"]

    def test_tokenization_with_punctuation(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = gen.tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "," in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_russian_text_tokenization(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = gen.tokenize("Мама мыла раму.")
        assert "Мама" in tokens
        assert "мыла" in tokens
        assert "раму" in tokens
        assert "." in tokens

    def test_text_too_short(self) -> None:
        gen = MarkovTextGenerator(order=5)
        with pytest.raises(TooShortTextError):
            gen.tokenize("короткий текст")

    def test_empty_text(self) -> None:
        gen = MarkovTextGenerator(order=2)
        with pytest.raises(TooShortTextError):
            gen.tokenize("a")


class TestTransitions:
    """Тесты построения таблицы переходов."""

    def test_build_transitions(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        transitions = gen.build_transitions(tokens)

        assert ("the", "cat") in transitions
        assert ("cat", "sat") in transitions
        assert transitions[("the", "cat")]["sat"] == 1

    def test_train_from_text(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")

        assert gen.is_trained()
        assert ("the", "cat") in gen.transitions
        assert gen._total_transitions > 0

    def test_train_clears_previous(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat with a hat")
        first_states = len(gen.transitions)
        first_total = gen._total_transitions

        gen.train_from_text("completely different text here now")
        second_states = len(gen.transitions)
        second_total = gen._total_transitions

        # Проверяем, что модель действительно перезаписалась
        assert (first_states, first_total) != (second_states, second_total)

    def test_clear(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")
        assert gen.is_trained()

        gen.clear()
        assert not gen.is_trained()
        assert gen._total_transitions == 0


class TestGeneration:
    """Тесты генерации текста."""

    def test_generate_basic(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat the cat sat")

        result = gen.generate(("the", "cat"), max_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_untrained_error(self) -> None:
        gen = MarkovTextGenerator(order=2)
        with pytest.raises(ModelNotTrainedError):
            gen.generate(("the", "cat"), max_tokens=5)

    def test_generate_unknown_state(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")

        with pytest.raises(UnknownStateError):
            gen.generate(("unknown", "state"), max_tokens=5)

    def test_generate_wrong_state_length(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")

        with pytest.raises(ValueError, match="ровно 2 токенов"):
            gen.generate(("the",), max_tokens=5)

    def test_generate_with_temperature(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat the dog ran fast")

        result = gen.generate(("the", "cat"), max_tokens=5, temperature=0.5)
        assert isinstance(result, str)

    def test_generate_with_min_tokens(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat. the dog ran fast.")

        result = gen.generate(("the", "cat"), max_tokens=20, min_tokens=5, stop_tokens={"."})
        tokens = result.split()
        # Должно быть минимум 5 токенов (2 начальных + 3 сгенерированных)
        assert len(tokens) >= 5

    def test_get_random_start_state(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")

        state = gen.get_random_start_state()
        assert isinstance(state, tuple)
        assert len(state) == 2
        assert state in gen.transitions

    def test_get_random_start_state_untrained(self) -> None:
        gen = MarkovTextGenerator(order=2)
        with pytest.raises(ModelNotTrainedError):
            gen.get_random_start_state()


class TestWeightedChoice:
    """Тесты взвешенного выбора."""

    def test_weighted_choice_basic(self) -> None:
        gen = MarkovTextGenerator(order=2)
        options = {"a": 10, "b": 1}

        # С очень низкой температурой должны почти всегда выбирать "a"
        results = [gen._weighted_choice(options, temperature=0.1) for _ in range(50)]
        assert results.count("a") > results.count("b")

    def test_weighted_choice_temperature_one(self) -> None:
        gen = MarkovTextGenerator(order=2)
        options = {"a": 1, "b": 1}

        result = gen._weighted_choice(options, temperature=1.0)
        assert result in options

    def test_weighted_choice_single_option(self) -> None:
        gen = MarkovTextGenerator(order=2)
        options = {"only": 5}

        result = gen._weighted_choice(options, temperature=1.0)
        assert result == "only"

    def test_weighted_choice_invalid_temperature(self) -> None:
        gen = MarkovTextGenerator(order=2)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            gen._weighted_choice({"a": 1}, temperature=0)


class TestStatistics:
    """Тесты статистических методов."""

    def test_get_entropy(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat")

        entropy = gen.get_entropy()
        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_get_entropy_untrained(self) -> None:
        gen = MarkovTextGenerator(order=2)
        with pytest.raises(ModelNotTrainedError):
            gen.get_entropy()

    def test_get_perplexity(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat the dog ran")

        perplexity = gen.get_perplexity()
        assert isinstance(perplexity, float)
        assert perplexity >= 1

    def test_get_statistics(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat on the mat the cat ran the dog sat")

        stats = gen.get_statistics()
        assert "order" in stats
        assert "unique_states" in stats
        assert "unique_tokens" in stats
        assert "total_transitions" in stats
        assert "entropy_bits" in stats
        assert "perplexity" in stats


class TestSerialization:
    """Тесты сохранения и загрузки модели."""

    def test_save_and_load(self) -> None:
        gen = MarkovTextGenerator(order=3)
        gen.train_from_text("the quick brown fox jumps over the lazy dog")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            gen.save_model(path)

            loaded = MarkovTextGenerator.load_model(path)
            assert loaded.order == gen.order
            assert loaded.transitions == gen.transitions

    def test_save_creates_directory(self) -> None:
        gen = MarkovTextGenerator(order=2)
        gen.train_from_text("the cat sat")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "model.pkl"
            gen.save_model(path)
            assert path.exists()

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            MarkovTextGenerator.load_model("/nonexistent/path/model.pkl")


class TestFileOperations:
    """Тесты операций с файлами."""

    def test_read_text(self) -> None:
        gen = MarkovTextGenerator(order=2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello world test")
            temp_path = f.name

        try:
            text = gen.read_text(temp_path)
            assert text == "Hello world test"
        finally:
            Path(temp_path).unlink()

    def test_read_text_nonexistent(self) -> None:
        gen = MarkovTextGenerator(order=2)
        with pytest.raises(FileNotFoundError):
            gen.read_text("/nonexistent/file.txt")

    def test_read_text_empty(self) -> None:
        gen = MarkovTextGenerator(order=2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("   ")
            temp_path = f.name

        try:
            with pytest.raises(EmptyFileError):
                gen.read_text(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_train_from_file(self) -> None:
        gen = MarkovTextGenerator(order=2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("the cat sat on the mat")
            temp_path = f.name

        try:
            gen.train_from_file(temp_path)
            assert gen.is_trained()
        finally:
            Path(temp_path).unlink()

    def test_train_from_multiple_files(self) -> None:
        gen = MarkovTextGenerator(order=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "text1.txt"
            file2 = Path(tmpdir) / "text2.txt"
            file1.write_text("the cat sat", encoding='utf-8')
            file2.write_text("on the mat", encoding='utf-8')

            gen.train_from_multiple_files([file1, file2])
            assert gen.is_trained()


class TestJoinTokens:
    """Тесты склеивания токенов обратно в текст."""

    def test_join_basic(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = ["Hello", "world", "!"]
        result = gen._join_tokens(tokens)
        assert result == "Hello world!"

    def test_join_with_punctuation(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = ["Hello", ",", "world", "!"]
        result = gen._join_tokens(tokens)
        assert "Hello, world!" == result

    def test_join_with_quotes(self) -> None:
        gen = MarkovTextGenerator(order=2)
        tokens = ["Он", "сказал", ":", "«", "привет", "»", "."]
        result = gen._join_tokens(tokens)
        assert "Он сказал: «привет»." == result


class TestTokenizer:
    """Тесты класса Tokenizer."""

    def test_basic_tokenization(self) -> None:
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello world!")
        assert "Hello" in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_normalize_yo(self) -> None:
        tokenizer = Tokenizer(normalize_yo=True)
        text = "Ёжик ест ягоды"
        result = tokenizer._normalize_yo(text)
        assert "Ёжик" not in result
        assert "Ежик" in result

    def test_remove_urls(self) -> None:
        tokenizer = Tokenizer(remove_urls=True)
        text = "Visit https://example.com for more info"
        result = tokenizer.preprocess(text)
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_remove_emails(self) -> None:
        tokenizer = Tokenizer(remove_emails=True)
        text = "Contact test@example.com please"
        result = tokenizer.preprocess(text)
        assert "test@example.com" not in result
        assert "Contact" in result

    def test_lowercase(self) -> None:
        tokenizer = Tokenizer(lowercase=True)
        tokens = tokenizer.tokenize("Hello WORLD")
        # Первое слово может остаться с заглавной (начало предложения)
        # А последующие в lowercase
        assert "WORLD" not in tokens

    def test_split_sentences(self) -> None:
        tokenizer = Tokenizer()
        text = "Первое предложение. Второе предложение! Третье?"
        sentences = tokenizer.split_sentences(text)
        assert len(sentences) == 3
        assert "Первое" in sentences[0]
        assert "Второе" in sentences[1]
        assert "Третье" in sentences[2]

    def test_split_sentences_with_abbreviations(self) -> None:
        tokenizer = Tokenizer()
        text = "См. т.е. это пример. Итд."
        sentences = tokenizer.split_sentences(text)
        # t.е. не должно разбить предложение
        assert len(sentences) == 1 or "См. т.е." in sentences[0]


class TestTokenizerWithGenerator:
    """Тесты интеграции Tokenizer с MarkovTextGenerator."""

    def test_generator_with_custom_tokenizer(self) -> None:
        tokenizer = Tokenizer(normalize_yo=True, lowercase=True)
        gen = MarkovTextGenerator(order=2, tokenizer=tokenizer)

        # Обучаем на тексте с Ё
        gen.train_from_text("Ёжик бежит быстро. Ёжик спит.")

        # Проверяем, что токены нормализованы
        for state in gen.transitions.keys():
            for token in state:
                assert 'ё' not in token.lower(), "Ё не нормализована к е"

    def test_generator_with_url_removal(self) -> None:
        tokenizer = Tokenizer(remove_urls=True)
        gen = MarkovTextGenerator(order=2, tokenizer=tokenizer)

        gen.train_from_text("Check https://example.com for info and more text here")

        # URL должен быть удален до токенизации
        for state in gen.transitions.keys():
            for token in state:
                assert "http" not in token

    def test_split_sentences_via_generator(self) -> None:
        gen = MarkovTextGenerator(order=2)
        text = "Первое. Второе. Третье."
        sentences = gen.split_sentences(text)
        assert len(sentences) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
