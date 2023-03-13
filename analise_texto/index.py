import pandas as pd 
import nltk
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')

class SentimentalText:
    """
    A class that preprocesses text and calculates its sentiment score using the
    SentimentIntensityAnalyzer class from the NLTK library.
    """
    def __init__(self, language='english', translate=False):
        """
        Initializes the SentimentalText object.

        Parameters:
            language (str): The language of the text to be analyzed. Defaults to 'english'.
            translate (bool): A boolean indicating whether the text should be translated to the
            specified language using the GoogleTranslator class from the deep_translator package.
            Defaults to False.
        """
        self.language = language
        self.translate = translate
        self.sia = SentimentIntensityAnalyzer()

    def pre_processamento(self, frase):
        """
        Preprocesses a text by removing stop words, tokenizing it, and converting all words to
        lowercase.

        Parameters:
            frase (str): The text to be preprocessed.

        Returns:
            A string with the preprocessed text.
        """
        if self.translate:
            frase = GoogleTranslator(source='auto', target=self.language).translate(frase)

        stop_words = set(stopwords.words(self.language))
        word_tokens = word_tokenize(frase.lower())
        filtered_sentence = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_sentence)
    
    def sentimento(self, frase=''):
        """
        Calculates the sentiment score of a text using the SentimentIntensityAnalyzer class from the
        NLTK library.

        Parameters:
            frase (str): The text to be analyzed.

        Returns:
            A float between -1 and 1 representing the sentiment score of the text.
        """
        preprocessed_sentence = self.pre_processamento(frase)
        sentiment_score = self.sia.polarity_scores(preprocessed_sentence)
        return sentiment_score['compound']


if __name__ == '__main__':
    df = pd.read_csv('amazon.csv')
    df['text'] = df['reviewTitle'] + ' ' + df['reviewDescription']
    df['sentiment_score'] = df['text'].apply(lambda x: SentimentalText(language='english', translate=True).sentimento(x))
    df.to_csv('comments_with_sentiment_scores.csv', index=False)
    print(df)
