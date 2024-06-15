import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Descargar los recursos necesarios
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging_nltk(sentence):
    # Tokenizar la oración
    words = word_tokenize(sentence)
    # Etiquetado de partes del habla
    pos_tags = pos_tag(words)
    return pos_tags

# Ejemplo de uso
sentence = "Hello, World! This is an example sentence."
pos_tags = pos_tagging_nltk(sentence)
print(pos_tags)
