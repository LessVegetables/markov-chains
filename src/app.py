import streamlit as st
from markov_chain.markov_core import (
    MarkovError,
    MarkovTextGenerator,
    ModelNotTrainedError,
    TooShortTextError,
)

st.html("""<script defer src="https://stat.gehrman.me/script.js" data-website-id="025ade9a-2a56-494f-a07a-21d6e4b07019"></script>""")


def generate_text():
    seed = st.session_state.seed
    if seed is not None:
        start_state = generator.get_random_start_state(seed)

    user_input = st.session_state.user_input
    max_tokens = st.session_state.max_tokens
    min_tokens = st.session_state.min_tokens
    temperature = st.session_state.temperature

    if len(user_input.split()) >= 3:
        try:
            st.session_state.output = generator.generate_text(
                                                    start_text=user_input,
                                                    max_tokens=max_tokens,
                                                    min_tokens=min_tokens,
                                                    temperature=temperature,
                                                )
            if generator.used_random_start:
                st.warning(
                    "Такого начала нет в обученной модели (или оно разбивается на токены иначе). "
                    "Текст начат со случайного состояния из корпуса."
                )
        except ModelNotTrainedError:
            st.error(
                "Модель не удалось обучить: проверьте, что в data/processed есть "
                "тексты для load_default_dataset."
            )
        except TooShortTextError as exc:
            st.error(f"Недостаточно токенов для выбранного порядка цепи: {exc}")
        except ValueError as exc:
            msg = str(exc)
            if "order must be >= 1" in msg:
                st.error("Некорректный порядок цепи: введите непустую строку.")
            else:
                st.error(f"Некорректные параметры генерации: {msg}")
        except MarkovError as exc:
            st.error(f"Ошибка модели Маркова: {exc}")
        except OSError as exc:
            st.error(f"Не удалось прочитать данные или сохранить временные файлы: {exc}")


default = "И он начал"

if "user_input" not in st.session_state:
    st.session_state.user_input = default

if len(st.session_state.user_input) == 0:
        st.error("Введите хотя бы три слова в начале цепи.")
else:
    generator = MarkovTextGenerator(order=len(st.session_state.user_input.split()))
    generator.load_default_dataset()

with st.sidebar:
    st.header("Parameters")
    st.session_state.max_tokens = st.slider("Max tokens", 10, 200, 80, 10)
    st.session_state.min_tokens = st.slider("Min tokens", 5, 100, 20, 5)
    st.session_state.temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    use_random_seed = st.checkbox("Random seed", value=True)
    st.session_state.seed = None if use_random_seed else st.number_input("Seed", min_value=0, value=42)


st.text_input(
    "Start of Markov Chain",
    value=default,
    label_visibility="hidden",
    key="user_input",
    on_change=generate_text
)

if st.button("Generate"):
    st.html("<script>umami.track('generate-clicked')</script>")
    generate_text()

st.write(st.session_state.get("output", ""))
