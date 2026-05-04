import streamlit as st
from markov_chain.markov_core import (
    MarkovError,
    MarkovTextGenerator,
    ModelNotTrainedError,
    TooShortTextError,
    UnknownStateError,
)

st.html("""
    <script defer src="https://stat.gehrman.me/script.js" data-website-id="025ade9a-2a56-494f-a07a-21d6e4b07019"></script>
""")

default = "И он начал" #ему объяснять
userInput = st.text_input("Start of Markov Chain", value=default, label_visibility="hidden")

if st.button("Generate"):
    st.html('<script>umami.track("generate-clicked")</script>')

    words = userInput.split()
    if not words:
        st.error("Введите хотя бы одно слово в начале цепи.")
    else:
        try:
            generator = MarkovTextGenerator(order=len(words))
            generator.load_default_dataset()
            output = generator.generate_from_seed(userInput)
            st.write(output)
        except UnknownStateError:
            st.error(
                "Такого начала нет в обученной модели: этих слов подряд нет в корпусе "
                "или они сегментируются иначе. Попробуйте фразу из стиля текста "
                "(например, как в примере поля) или добавьте свой корпус и переобучите модель."
            )
        except ModelNotTrainedError:
            st.error("Модель не удалось обучить: проверьте, что в data/processed есть тексты для load_default_dataset.")
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
