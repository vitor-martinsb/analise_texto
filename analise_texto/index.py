import pandas as pd
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import nltk
from collections import Counter

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
        stop_words.update(['de', 'a', 'e', 'que', 'dir="auto"', 'class="style-scope'
                           'o', 'em', 'da', 'na', 'por', 'para', 'os', 'dos', 'um', 'com', 'é'])
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        frase = emoji_pattern.sub(r'', frase) # remove emojis
        frase = re.sub('span dir=""auto"" class=""style-scope yt-formatted-string"">', frase)
        frase = re.sub('/span><span dir=""auto"" class=""style-scope yt-formatted-string"">', frase)
        sentence_tokens = sent_tokenize(frase.lower())
        filtered_sentence = []
        if self.translate:
            sentence_tokens = [GoogleTranslator(source='auto', target=self.language).translate(sentence) for sentence in sentence_tokens]
        for sentence in sentence_tokens:
            for sentence in sentence_tokens:
                word_tokens = word_tokenize(sentence)
                filtered_sentence.extend([word for word in word_tokens if word.isalnum() and word not in stop_words])
            return ' '.join(filtered_sentence)

    def sentimento(self, frase=''):
        '''
        Retorna um decimal -1 a 1 
        
                Parameters:
                        frase (str): Frase a ser analisada
        
                Returns:
                        result (str): Binary string of the sum of a and b
        '''
        preprocessed_comment = self.pre_processamento(frase)
        sentence_tokens = sent_tokenize(preprocessed_comment)
        if len(sentence_tokens) == 0:
            return 0
        sentiment_scores = [self.sia.polarity_scores(sentence)['compound'] for sentence in sentence_tokens]
        return sum(sentiment_scores)/len(sentiment_scores)

if __name__ == '__main__':
    df = pd.read_csv('output.csv')
    comment_cols = [col for col in df.columns if col.startswith('comments')]
    cols = comment_cols
    comments_df = df[cols]
    comments_df = df.melt(value_vars=comment_cols, value_name='comment')
    comments_df = comments_df.dropna()
    comments_df = comments_df.reset_index(drop=True)

    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(['de', 'a', 'e', 'que', 'dir="auto"', 'class="style-scope'
                           'o', 'em', 'da', 'na', 'por', 'para', 'os', 'dos', 'um', 'com', 'é'])

    words = []
    for comment in comments_df['comment']:
        words += [word.lower() for word in comment.split() if word.lower() not in stop_words]

    word_counts = Counter(words)

    print("Palavras mais comuns:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

    comments_df['sentiment_score'] = comments_df.apply(lambda x: sentimental_text(language='english', translate=True).sentimento(x['comment']), axis=1)
    df.to_csv('outputter.csv', index=True)
    print(comments_df.head())


