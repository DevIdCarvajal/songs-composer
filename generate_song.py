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
    print("Por favor, introduce algunas palabras clave separadas por espacios:")
    seed_text = input("> ").strip()

    print("\nGenerando letra de la canción...\n")
    generated_lyrics = generate_lyrics(model, tokenizer, seed_text, num_words=50)
    print("Letra generada:\n")
    print(generated_lyrics)

if __name__ == "__main__":
    main()
