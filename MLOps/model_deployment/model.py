# fonctions chargement et prediction du modele
# fonction de pretraitement des donnes
# création du model


import re

POSITIVE_WORDS = {
    'bon', 'super', 'génial', 'excellent', 'parfait', 'love', 'great'

}

NEGATIVE_WORDS = {
    'mauvais', 'nul', 'horrible', 'terrible', 'hate', 'bad'

}


def _normalized_text(text:str) :
    text = text.lower()
    words = re.findall(r"\w+",text, flags=re.UNICODE)
    return words

def analyse_sentiment(text):
    words = _normalized_text(text)

    has_positive = any(word in POSITIVE_WORDS for word in words)
    has_negative = any(word in NEGATIVE_WORDS for word in words)


    # logique de la prediction
    sentiment =''

    return {
            'sentiment':"",
            'mot_imporant':"",
            'proba':""
            }