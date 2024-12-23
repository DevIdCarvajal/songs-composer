# Proyecto Final: Deep Learning con TensorFlow y Keras

- **Curso**: 8_PT MAD_ APLICACIONES INDUSTRIALES DEL APRENDIZAJE AUTOMÁTICO Y LA INTELIGENCIA ARTIFICIAL
- **Alumno**: David Carvajal Garrido

## Índice

- [Requisitos previos](#requisitos-previos)
- [Metodología de trabajo](#metodología-de-trabajo)
- [Resultados obtenidos](#resultados-obtenidos)

## Requisitos previos

El objetivo del proyecto es crear una aplicación a la que se le pida componer una letra original de una canción en base a unas palabras clave proporcionadas por el usuario, que deberán aparecer en el texto generado.

Inicialmente como entorno de trabajo se empezó a trabajar localmente en una máquina que tiene instalados **Visual Studio Code** con el plugin **Jupyter** para leer y ejecutar notebooks, así como **Python** con las siguientes librerías:

- **TensorFlow y Keras**: Para la creación del modelo de aprendizaje profundo.
- **Pandas**: Para manipular y explorar el dataset.
- **NumPy**: Para cálculos numéricos.
- **scikit-learn**: Para procesamiento y vectorización.
- **nltk**: Para trabajar con texto y preprocesar las letras.
- **Matplotlib**: Para visualizar datos durante la fase de análisis.

Sin embargo y debido a insuficiencia de recursos hardware y problemas con la puesta a punto de la máquina, finalmente se ha entrenado el modelo en Kaggle en el siguiente notebook:

[https://www.kaggle.com/code/davidcarvajalgarrido/generador-de-letras-de-canciones](https://www.kaggle.com/code/davidcarvajalgarrido/generador-de-letras-de-canciones)

Se ha utilizado también el siguiente dataset en CSV de información sobre canciones de todos los géneros de 1950 a 2019:

[https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)

Sin embargo, para este proyecto se ha tomado un subconjunto de esos datos, tomando solamente las canciones del género rock y los siguientes campos:

- **artist_name**: El nombre del cantante.
- **track_name**: El título de la canción.
- **release_date**: El año de publicación.
- **genre**: El género de la canción.
- **lyrics**: La letra de la canción.
- **len**: Los segundos de duración.
- **topic**: El tema general de la letra.

## Metodología de trabajo

### Preparación del entorno (en local)

**Aclaración previa**: Como se ha indicado anteriormente, aunque la puesta a punto de la máquina de trabajo se haya llevado a cabo siguiendo los siguientes pasos, finalmente el entrenamiento del modelo se ha hecho desde un notebook de Kaggle.

1. **Instalar dependencias necesarias**:

    ```bash
    pip install tensorflow keras pandas numpy scikit-learn nltk jupyter matplotlib
    ```

2. **Instalar el plugin de Jupyter para Visual Studio Code**:

    [https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

3. **Ficheros del proyecto**:

    ```
    data/tcc_ceds_music_rock.csv   # Datos
    models/song_generator.h5       # Modelo entrenado
    models/tokenizer_config.json   # Tokenizer del modelo
    songs_composer.ipynb           # Generación del modelo
    generate_song.py               # Aplicación de usuario
    ```

### Exploración y preprocesamiento del dataset

1. **Carga inicial del dataset**: Mediante la librería `pandas` se han cargado los datos del CSV y se ha comprobado que se han importado correctamente en un `dataframe`.

    ```python
    import pandas as pd

    # Cargar dataset
    df = pd.read_csv("data/tcc_ceds_music_rock.csv")

    # Exploración
    print(df.head())
    print(df.info())
    print(df['lyrics'].iloc[0])
    ```

2. **Limpieza de datos**: Como puede haber filas con datos incompletos, se ha decidido descartarlas.

    Además, en el campo que contiene la letra de la canción se han suprimido algunos caracteres no deseados.
    
    Esta limpieza se ha hecho con métodos de `dataframe` así como los métodos de clases primitivas de Python como los del tipo `string`.

    ```python
    # Filas con datos incompletos
    df = df.dropna()

    # Eliminación de caracteres no deseados
    df['Lyrics'] = df['lyrics'].str.replace(r'\n', ' ')
    ```

3. **Tokenización y preprocesamiento de texto**: Una vez limpiado el texto, se ha realizado la tokenización del mismo así como el procesamiento de las letras, que dado el dataset de partida se trata de temas en inglés.

    Para ello se ha empleado la librería `nltk`, concretamente los métodos `word_tokenize` y `stopwords`.

    ```python
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')

    # Tokenización y limpieza
    stop_words = set(stopwords.words('english'))

    def preprocess_lyrics(lyrics):
        tokens = word_tokenize(lyrics.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        return ' '.join(tokens)

    df['Processed_Lyrics'] = df['lyrics'].apply(preprocess_lyrics)
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
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense

    model = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(len(tokenizer.word_index) + 1, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    ```

2. **Crear el modelo de generación**: Con todos los elementos necesarios se ha procedido a crear el modelo, utilizando una arquitectura de tipo LSTM (red neuronal recurrente de memoria a corto-largo plazo), que está especialmente indicada para el tipo de tareas que se pretenden realizar en este proyecto.

    De nuevo `keras` ofrece las herramientas para implementarla.

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

3. **Entrenamiento**: El último paso es la división de los datos en entrenamiento y validación, para lo que se utilizar la librería `scikit-learn`.

    Se ha hecho también una conversión intermedia de etiquetas con `keras` para facilitar el proceso de entrenamiento.
    
    Por último, se ha guardado el modelo y el tokenizer en su carpeta correspondiente para poderlos usar después en la aplicación.

    ```python
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    import json

    # Crear etiquetas y convertir etiquetas a formato categórico
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    # Dividir los datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Guardar el modelo entrenado y el tokenizer
    model.save("models/song_generator.h5")

    tokenizer_config = {
        "word_index": tokenizer.word_index,
        "max_length": max_length
    }
    with open("models/tokenizer_config.json", "w") as f:
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
model = load_model("models/song_generator.h5")
tokenizer = Tokenizer()
tokenizer_config_path = "models/tokenizer_config.json"

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
    # print("Por favor, introduce un artista similar:")
    # artist = input("> ").strip()
    
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

Tras realizar todo el proyecto, se ha lanzado el script `generate_song.py` probando con las palabras `magic time`, con este resultado:

```bash
magic time
```

Se han hecho otras pruebas similares con otras palabras clave con los mismos resultados. Esto claramente significa que aún queda trabajo por hacer, ya que el modelo no es capaz de crear sus propias letras adecuadamente. 

Por tanto, es obvio que hay que seguir trabajando en el proyecto, bien en los datos, bien en el proceso, hasta que sea capaz de generar letras que con la música adecuada puedan llegar a convertirse en grandes éxitos.
