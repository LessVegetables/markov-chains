import streamlit as st
from markov_chain.markov_core import (
    MarkovError,
    MarkovTextGenerator,
    ModelNotTrainedError,
    TooShortTextError,
)

st.html("""<script defer src="https://stat.gehrman.me/script.js" data-website-id="025ade9a-2a56-494f-a07a-21d6e4b07019"></script>""")

default = "И он начал" #ему объяснять
userInput = st.text_input("Start of Markov Chain", value=default, label_visibility="hidden")

if st.button("Generate"):

    st.html("<script>umami.track('generate-clicked')</script>")
    
    generator = MarkovTextGenerator(order=len(userInput.split()))
    generator.load_default_dataset()
    output = generator.generate_from_seed(userInput)

    words = userInput.split()
    if not words:
        st.error("Введите хотя бы одно слово в начале цепи.")
    else:
        try:
            generator = MarkovTextGenerator(order=len(words))
            generator.load_default_dataset()
            output = generator.generate_text(userInput)
            if generator.used_random_start:
                st.warning(
                    "Такого начала нет в обученной модели (или оно разбивается на токены иначе). "
                    "Текст начат со случайного состояния из корпуса."
                )
            st.write(output)
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
