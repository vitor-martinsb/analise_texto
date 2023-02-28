import pandas as pd 
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import nltk
import json

nltk.download('stopwords')

class sentimental_text:
    
    def __init__(self, language='english', translate=False):
        self.language = language
        self.translate = translate
        self.sia = SentimentIntensityAnalyzer()
    
    def pre_processamento(self, frase):
        '''
        Retorna o sentimento negativo/positivo de uma frase
        
                Parameters:
                        frase (str): Frase a ser analisada
        
                Returns:
                        result (float): -1 -> Negativo / 0 -> Neutro / 1 -> Positivo
        '''
        if self.translate:
            frase = GoogleTranslator(source='auto', target=self.language).translate(frase)
        
        stop_words = set(stopwords.words(self.language))
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        frase = emoji_pattern.sub(r'', frase) # remove emojis
        word_tokens = word_tokenize(frase.lower())
        filtered_sentence = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_sentence)

    def sentimento(self, frase=''):
        '''
        Retorna um decimal -1 a 1 
        
                Parameters:
                        frase (str): Frase a ser analisada
        
                Returns:
                        result (str): Binary string of the sum of a and b
        '''
        preprocessed_sentence = self.pre_processamento(frase)
        sentiment_score = self.sia.polarity_scores(preprocessed_sentence)
        return sentiment_score['compound']

if __name__ == '__main__':
    df = pd.read_csv('review.csv')
    df['text'] = df['title'] + ' ' + df['body_text']
    df['sentiment_score'] = df['text'].apply(lambda x: sentimental_text(language='english', translate=True).sentimento(x))
    df.to_csv('comments_with_sentiment_scores.csv', index=False)
    print(df)
