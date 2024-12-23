# Proyecto Final: Deep Learning con TensorFlow y Keras

- **Curso**: 8_PT MAD_ APLICACIONES INDUSTRIALES DEL APRENDIZAJE AUTOMÁTICO Y LA INTELIGENCIA ARTIFICIAL
- **Alumno**: David Carvajal Garrido

## Índice

- [Requisitos previos](#requisitos-previos)
- [Metodología de trabajo](#metodología-de-trabajo)
- [Resultados obtenidos](#resultados-obtenidos)

## Requisitos previos

El objetivo del proyecto es crear una aplicación a la que se le pida componer una letra original de una canción en base a dos parámetros de entrada:

1. Artista similar
2. Palabras que deban aparecer en el texto generado

Como entorno de trabajo se ha trabajado localmente en una máquina que tenía instalados Visual Studio Code y Python.

Se ha añadido un plugin para VSCode para leer y ejecutar notebooks, así como se han instalado las siguientes librerías necesarias:

- **TensorFlow y Keras**: Para la creación del modelo de deep learning.
- **Pandas**: Para manipular y explorar el dataset.
- **NumPy**: Para cálculos numéricos.
- **scikit-learn**: Para procesamiento y vectorización.
- **nltk**: Para trabajar con texto y preprocesar las letras.
- **Matplotlib**: Para visualizar datos durante la fase de análisis.

Se ha utilizado un dataset en CSV de información sobre canciones, concretamente con los campos siguientes:

- **Name**: El título de la canción.
- **Lyrics**: La letra de la canción.
- **Singer**: El nombre del cantante o artista.
- **Movie**: La película o álbum asociado a la canción (si procede).
- **Genre**: El género o géneros a la que pertenece la canción.
- **Rating**: La puntuación o popularidad de la canción en Spotify.

El dataset utilizado ha sido el siguiente:

[https://www.kaggle.com/datasets/suraj520/music-dataset-song-information-and-lyrics](https://www.kaggle.com/datasets/suraj520/music-dataset-song-information-and-lyrics)

## Metodología de trabajo

### Preparación del entorno

1. **Instalar dependencias necesarias**:

    ```bash
    pip install tensorflow keras pandas numpy scikit-learn nltk jupyter matplotlib
    ```

2. **Instalar el plugin de Jupyter para Visual Studio Code**:

    [https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

3. **Ficheros del proyecto**:

    ```
    songs_dataset.csv         # Datos
    songs_composer.ipynb      # Análisis exploratorios y pruebas
    song_generator.h5         # Modelo entrenado
    tokenizer_config.json     # Tokenizer del modelo
    generate_song.py          # Código de la aplicación
    ```

### Exploración y preprocesamiento del dataset

1. **Carga inicial del dataset**: Mediante la librería `pandas` se han cargado los datos del CSV y se ha comprobado que se han importado correctamente en un `dataframe`.

    ```python
    import pandas as pd

    # Cargar dataset
    df = pd.read_csv("songs_dataset.csv")

    # Exploración
    print(df.head())
    print(df.info())
    print(df['Lyrics'].iloc[0])  # Ejemplo de letra
    ```

2. **Limpieza de datos**: Como puede haber filas con datos incompletos, se ha decidido descartarlas.

    Además, en el campo que contiene la letra de la canción se han suprimido algunos caracteres no deseados.
    
    Esta limpieza se ha hecho con métodos de `dataframe` así como los métodos de clases primitivas de Python como los del tipo `string`.

    ```python
    df = df.dropna()
    
    df['Lyrics'] = df['Lyrics'].str.replace(r'\n', ' ')
    ```

3. **Tokenización y preprocesamiento de texto**: Una vez limpiado el texto, se ha realizado la tokenización del mismo así como el procesamiento de las letras, que dado el dataset de partida se trata de temas en inglés.

    Para ello se ha empleado la librería `nltk`, concretamente los métodos `word_tokenize` y `stopwords`.

    ```python
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    nltk.download('punkt')
    nltk.download('stopwords')

    # Tokenización y limpieza
    stop_words = set(stopwords.words('english'))
    
    def preprocess_lyrics(lyrics):
        tokens = word_tokenize(lyrics.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        return ' '.join(tokens)

    df['Processed_Lyrics'] = df['Lyrics'].apply(preprocess_lyrics)
    ```

4. **Análisis exploratorio**: A partir de la tokenización se ha analizado la distribución de artistas, géneros y palabras más comunes, utilizando la función `Counter` del módulo de Python de `collections`.

    Como ayuda a la visualización mental de dicha distribución, se ha generado una gráfica con la librería `matplotlib`.

    ```python
    from collections import Counter
    import matplotlib.pyplot as plt

    word_counts = Counter(" ".join(df['Processed_Lyrics']).split())

    # Visualización
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.show()
    ```

### Modelado con Deep Learning

1. **Vectorización del texto**: De cara al entrenamiento del modelo, el texto debe ser vectorizado debidamente.

    Para ello, se ha hecho uso de la librería `keras` dentro de `tensorflow`, concretamente de `Tokenizer` y `pad_sequences`, obteniendo así las secuencias con las que se realizará posteriormente el entrenamiento.

    ```python
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['Processed_Lyrics'])
    sequences = tokenizer.texts_to_sequences(df['Processed_Lyrics'])

    # Padding
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    ```

2. **Crear el modelo de generación**: Con todos los elementos necesarios se ha procedido a crear el modelo, utilizando una arquitectura de tipo LSTM (red neuronal recurrente de memoria a corto-largo plazo), que está especialmente indicada para el tipo de tareas que se pretenden realizar en este proyecto.

    De nuevo `keras` ofrece las herramientas para implementarla.

    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense

    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=max_length),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(len(tokenizer.word_index) + 1, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    ```

3. **Entrenamiento**: El último paso es la división de los datos en entrenamiento y validación, para lo que se utilizar la librería `scikit-learn`.

    Se ha hecho también una conversión intermedia de etiquetas con `keras` para facilitar el proceso de entrenamiento.
    
    Por último, se ha guardado el modelo y el tokenizer en su carpeta correspondiente para poderlos usar después en la aplicación.

    ```python
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, df['Genre'], test_size=0.2)

    # Convertir etiquetas a formato categórico (si es necesario)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Guardar el modelo entrenado y el tokenizer
    model.save("song_generator.h5")

    tokenizer_config = {
        "word_index": tokenizer.word_index,
        "max_length": max_length
    }
    with open("tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)
    ```

### Implementación de la aplicación

Se ha creado un fichero `generate_song.py` con el código necesario para pedir al usuario por consola los parámetros de entrada (artista similar y palabras clave) y mediante el uso del modelo entrenado generar así el texto de salida.

```python
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Cargar el modelo y el tokenizer
model = load_model("song_generator.h5")
tokenizer = Tokenizer()
tokenizer_config_path = "tokenizer_config.json"

# Cargar configuración del tokenizer
import json

with open(tokenizer_config_path, "r") as f:
    tokenizer_config = json.load(f)
tokenizer.word_index = tokenizer_config['word_index']
max_length = tokenizer_config['max_length']

# Función para generar letras
def generate_lyrics(model, tokenizer, seed_text, num_words=50):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = tokenizer.index_word.get(predicted[0], "")
        if not output_word:
            break
        seed_text += " " + output_word
    return seed_text

# Interacción con el usuario
def main():
    print("¡Bienvenido al generador de letras de canciones!")
    print("Por favor, introduce un artista similar:")
    artist = input("> ").strip()
    
    print("Ahora introduce algunas palabras clave separadas por espacios:")
    seed_text = input("> ").strip()

    print("\nGenerando letra de la canción...\n")
    generated_lyrics = generate_lyrics(model, tokenizer, seed_text, num_words=50)
    print("Letra generada:\n")
    print(generated_lyrics)

if __name__ == "__main__":
    main()
```

La ejecución del script debe hacerse en una máquina con Python instalado, ejecutando el siguiente comando desde la terminal:

```bash
python generate_song.py
```

## Resultados obtenidos

Tras realizar todo el proyecto, se ha lanzado el script `generate_song.py` probando con el artista `Queen` y las palabras `magic time`, con este resultado:

    [WIP]

Se han hecho otras pruebas similares con otros artistas y palabras clave, generando más letras, que con la música adecuada podrían llegar a convertirse en grandes éxitos.