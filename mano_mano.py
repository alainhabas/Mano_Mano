import pandas as pd
import numpy as np
# from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#import spacy
import nltk

import matplotlib.pyplot as plt
# import seaborn as sns

# Importe la librairie des mots anglais
#nlp = spacy.load("en_core_web_sm")


pd.set_option('display.width', 7000000)
pd.set_option('display.max_columns', None)

df_manomano = pd.read_csv('D:/Downloads/df_manomano.csv')

#print(df_manomano['reason'].value_counts().tail(25))


def code():

    var_comment = df_manomano[['comment', 'score', 'device']][~pd.isna(df_manomano.comment)]
    df_comment = pd.DataFrame(var_comment)

    var_reason = df_manomano.reason[~pd.isna(df_manomano.reason)]
    df_reason = pd.DataFrame(var_reason)


    df_filtered = df_manomano.copy()
    df_filtered = df_filtered[~pd.isna(df_filtered.reason)]


    print(df_manomano.head(5))


    #print(df_filtered['reason'].value_counts().tail(25))



    #print(df_filtered['reason_en'])

    #print(df_reason.apply(lambda x: translator.translate(x, dest='en').text))

    #df_manomano['reason_en'] = df_manomano['reason'].apply(lambda x: translator.translate(x, dest='en').text)


def filtering(sentence):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens_no_punctuation = tokenizer.tokenize(sentence)

    tokens_clean = []
    for word in tokens_no_punctuation:
        if word.lower() not in nltk.corpus.stopwords.words("english"):
            tokens_clean.append(word.lower())

    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatizer_tokens = " ".join(lemmatizer.lemmatize(token) for token in tokens_clean)

    # Transforme les mots du même radical en leurs mots d'origine ('am' en 'be', 'versions' en 'version'...)
    #sent_tokens = nlp(lemmatizer_tokens)
    #lemmatizer_tokens = []
    #for token in sent_tokens:
    #    lemmatizer_tokens.append(token.lemma_)

    return lemmatizer_tokens


#comment_score_filtered = df_manomano[(~pd.isna(df_manomano.comment)) & (df_manomano.score < 6)]
#df_comment_score_filtered = comment_score_filtered.copy()
#df_comment_score_filtered['filtered_comment'] = comment_score_filtered.comment.apply(filtering)
#print('Application des filtres terminée')




#print(df_comment_score_filtered.shape)

a = "I think the delivery is too expensive|Los gastos de envío me han parecido demasiado elevados|I costi di spedizione erano troppo elevati|Ich fand, dass die Versandkosten zu teuer waren|J’ai trouvé que les frais de port étaient trop chers"


#df_shipment = df_comment_score_filtered[(df_comment_score_filtered['filtered_comment'].str.contains('shipping|shipment')) |
#                                        (df_comment_score_filtered['reason'].str.contains(a))]

#df_manomano.reason = df_manomano.reason[~pd.isna(df_manomano.reason)]

#print(df_manomano.value_counts())

#df_manomano = df_manomano[~pd.isna(df_manomano.tags)]
#df_shipment = df_manomano[df_manomano['tags'].str.contains('Delivery Fees')]

#print(df_manomano['tags'].value_counts())

def zearytu():
    #z = df_manomano[pd.isna(df_manomano.tags)]

    df_manomano['tags'] = df_manomano['tags'].replace(np.nan, 'None')

    df_manomano['tags'] = df_manomano['tags'].apply(lambda line: [word.strip().title() for word in line.replace("Detractor '- ", "").replace(' Detractor - ', "").split(';')])

    w = pd.concat([df_manomano, df_manomano['tags'].apply(lambda x: '|'.join(x)).str.get_dummies()], axis=1)

    #c = pd.DataFrame({'tag': z.index, 'count': z}).reset_index(drop=True)

    #w.to_csv("C:/Users/simax/Desktop/df_tags.csv")


    w['bv_transaction'].hist(bins=100)
    plt.show()


zearytu()





comment_score_filtered = df_manomano[(~pd.isna(df_manomano.comment)) & (df_manomano.score < 6)]
df_comment_score_filtered = comment_score_filtered.copy()
print('Filtrage des nans et notes en dessous de 6')

#print(df_comment_score_filtered['reason'].value_counts().head(45))

def display_plot():
    a = nltk.FreqDist(sum(df_comment_score_filtered['filtered_comment'].map(nltk.word_tokenize), []))
    print('Cumul des mots terminé')
    b = pd.DataFrame(a.items(), columns=['words', 'frequency']).sort_values(by='frequency', ascending=False).set_index('words')
    b.head(40).plot(kind='bar')
    plt.show()


def tfidf_model():
    vectorizer = TfidfVectorizer()
    values = vectorizer.fit_transform(df_comment_score_filtered['filtered_comment'])

    feature_names = vectorizer.get_feature_names_out()

    print(pd.DataFrame(values.toarray(), columns=feature_names))


def word_cloud():



    df_comment_score_filtered['filtered_comment'] = comment_score_filtered.comment.apply(filtering)
    print("Suppression des stopwords et application d'un lemmatazing")

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    values = vectorizer.fit_transform(df_comment_score_filtered['filtered_comment'])
    feature_names = vectorizer.get_feature_names_out()
    print('Prise en charge des mots par 1, 2 et 3')

    c = pd.DataFrame(values.toarray(), columns=feature_names)
    print("Création d'un df par le CV")

    p = pd.DataFrame(c.describe().loc['max'])
    print('Df créé')

    h = p[p['max'] != 1.0].sort_values(by='max', ascending=False).head(45)
    print('Sorting')

    print(h)


#h = nltk.FreqDist(sum(df_comment_score_filtered['filtered_comment'].map(nltk.word_tokenize), []))
#b = pd.DataFrame(h.items(), columns=['words', 'frequency']).sort_values(by='frequency', ascending=False).set_index('words')
#print(b)
