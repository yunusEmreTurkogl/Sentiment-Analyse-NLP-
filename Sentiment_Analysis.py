################################
# Sentiment Analysis
################################

# Elimizde bulunan metinlerin duygu durumunu matematiksel olarak ifade etmek demektir.
# Elimizde bulunan metindeki her bir kelimenin duygusal olarak tasimis olduklari bir anlami vardir dusuncesiyyle hareket
# edip, bunlari butuncul olarak degerlendirerek bu metnin pozitif mi yoksa negatif mi olduguna iliskin degerlndirmesidir.

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL as Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk

filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option("display.float_format", lambda x: "%2f" % x)

#######################################################
# Text Preprocessing
#######################################################

df = pd.read_csv("NLP/nlp/datasets/amazon_reviews.csv", sep=",")
df["reviewText"] = df["reviewText"].str.lower()
df["reviewText"] = df["reviewText"].str.replace("[^\w\s]", ' ')
df["reviewText"] = df["reviewText"].str.replace("\d", '') # \d ifadesi ilgili degiskendeki sayilari yakala
df.head()
sw = stopwords.words("english")
# Metne gidip her bir satiri gezdiktenten sonra cok fazla kullanilan edat baglac gibi kelimeleri silmek icin kullanacagiz.
df["reviewText"] = df["reviewText"].apply(lambda x:" ".join(x for x in str(x).split() if x not in sw)) # split() on tanimli deger bosluk

# Bazi NLP projelerinde ilgili metinlerde nadir olarak gecen kelimeleri cikarmak isteriz. Bu her zaman yapilmasi gereken
# bir islem degildir o zamanki ihtiyaca gore yapilabilir. Cunku bu kelimelerin belirli bir oruntu olusturamayacagini
# dusunur, bu yuzden disarida birakmak isteriz
temp_df = pd.Series(" ".join(df["reviewText"]).split()).value_counts()
drops = temp_df[temp_df <= 1]
df["reviewText"] = df["reviewText"].apply(lambda x:" ".join(x for x in str(x).split() if x not in drops))

# Lemmatize (Kelimeleri koklerine ayirma)
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df["reviewText"].head()

# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")
#  {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}
# Yukaridaki ciktida asil odaklanilmasi gereken deger compound degeridir. Bu skor (-1,1) arasinda olan bir skordur.
# Buradan cikacak skor eger ki 0 dan buyukse bunun pozitif oldugu 0 dan kucukse bunun negatif oldugu cikariminda
# bulunacagiz.

sia.polarity_scores("I liked this music but it is not good as the other one")
# {'neg': 0.207, 'neu': 0.666, 'pos': 0.127, 'compound': -0.298}
# Compound degeri:-0.29 olarak gelmis buradan su cikarimi yapiyoruz cumle neagtif bir duygu ifade ediyor.

# Simdi df["reviewText"] deki her bir deger uzerinde gezecigiz ve bunlarin duygu durumunu cikaracagiz.

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])
df.head()
df.shape
df["reviewerName"].loc[(df['overall']>4) & (df['polarity_score']>0.5)]

##########################################
# Sentiment Modelling
##########################################

# Bir metin siniflandrma modeli olusturup bu modele cesitli yorumlar soruldugu zaman bu yorumun pozitif mi negatif mi
# olup olmadigini tahmin etmek.

# 1. Unsupervised bir sekilde olusturmus oldugum polarity_score degiskeninden supervised learning gecis yaparak bullmus
#    oldugum skorlarin belirli bir threshold dan buyuk oldugu olanlari 1 digerlerine 0 dersem bu benim label im olur.
#    Unsupervised obir sekilde cikarmis oldugum skora gore labellar olustururarak bir gecis yapmis olurum.
# 2. Makine ogrenmesi yontemleri ile bir siniflandirma modeli gelistirmek istedigimizi dusunelim. 0 ve 1 lerin oldugu
#    bir label dusunelim ve bunu uzerinden bir siniflandirma problemi olusturalim.


##########################################
# Feature Engineering
##########################################

# Siniflandirma modeli yapmak istedigimiz icin target degiskenimizi olusturacagiz. Bu degiskeni de polarity_score a
# gore yapacagiz.

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"]>0 else "neg")

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"]>0 else "neg")
df.head()

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

# Target degiskenimiz su anda string bir ifade oldugu ici LabelEncoder dan gecirecegiz.

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"]) # 1 ler pos 0 lar neg
df["sentiment_label"].value_counts()

y = df["sentiment_label"]
X = df['reviewText']
# Reviewtext degiskenimiz binary bir halde degil bunu binary hale getirecegiz NLP konusunun en onemli bolumu bu olacak.

##########################################
# Count Vector
##########################################

# Bagimli degiskenimizi yukarida olusturduk simdi amacimiz bagimli degiskenleri olusturmak
# Bu bagimsiz degiskenleri sadece elimizde olan metinden uretmek durumundayiz.
# NLP nin en onemli kismi elimizde bulunan metinlerin vectorlestirilmesi ve bundan sonra yapilacak olan adimlardir.
# Elde bulunan metni matematiksellestirmek icin kullanilan en yaygin yontemler
# 1. Count Vector: frekans temsilleri
# 2. TF-IDF Vectors: Normalize edilmis frekans temsilleri
# 3. Word Embedding (Word2Vec, Glove, BERT vs)

# Kelimeleri temsil yontemleri bu noktada onemli olacak. Word leri mi, characters leri mi yoksa ngram lari mi temsil edecek.
a = """Bu ornegin anlasilabilmesi icin daha uzun bir metin uzerinden gosterecegim.
N-gram'lar birlikte kullanilan kelimelerin kombinasyonlarini gosterir ve feature uretmek icin kullanilir."""

TextBlob(a).ngrams(3)

from sklearn.feature_extraction.text import CountVectorizer

corpus = ["This is the first document.",
          "This document is the second document.",
          "And this is the third one.",
          "Is this the first document?"]

# Amacimiz yukaridaki cumleleri vectorize etmek
# Word Frekans

# Count Vector

vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X_c.toarray()

# N-gram frekans

vectorizer2 = CountVectorizer(analyzer="word", ngram_range=(2,2))
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names()
X_n.toarray()

# Simdi kendi ornegimize uygulayacagiz.

vectorizer = CountVectorizer() # On tanimli degeri kelime
X_count = vectorizer.fit_transform(X)
vectorizer.get_feature_names()[0:5]

################################
# TF-IDF
################################

# Count vectorun ortaya cikarabilecegi yanlilik adina normalize edilmis standardlastirilmis bir vectoru olusturma yontemi.
# Kelimlerin document larda gecme frekansini ve butun corpusta gecme odaginda bir standardlastirma islemi yapilir.

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer() # On tanimli degeri kelime
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)

################################
# Sentiment Modelling
################################

# 1. Text Preprocessing
# 2. Text Visulaization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modelling

################################
# Logistic Regression
################################

log_model = LogisticRegression().fit(X_tf_idf_word, y)
cross_val_score(log_model, X_tf_idf_word, y, scoring="accuracy", cv=5).mean()

new_comment = pd.Series("this product is great")
new_comment = pd.Series("look at that shit very bad")
new_comment = TfidfVectorizer().fit(X).transform(new_comment)
log_model.predict(new_comment)

random_review = pd.Series(df['reviewText'].sample(1).values)
random_review = TfidfVectorizer().fit(X).transform(random_review)
log_model.predict(random_review)

################################
# Random Forest
################################

# Count Vector

rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1, scoring="accuracy").mean()
# 0.8451678535096642

# TF-IDF Word-Level

rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()
#  0.8295015259409968

# TF-IDF N-Gram

rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()
# 0.7865717192268565

################################
# LightGBM
################################

from lightgbm import LGBMClassifier

X_count = X_count.astype("float32")
y = y.astype("float32")
lgm_model = LGBMClassifier().fit(X_count, y)
cross_val_score(lgm_model, X_count, y, cv=5, scoring="accuracy").mean()
# 0.8905391658189217

######################################
# Hyperparametre Optimizasyonu(RF)
######################################

rf_model = RandomForestClassifier(random_state=17)
rf_params = {"max_depth": [8,None],
             "max_features": [5,7,"auto"],
             "min_samples_split": [2,5,8],
             "n_estimators": [100,200]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=50).fit(X_count, y)
rf_best_grid.best_params_
# {'max_depth': None,
#  'max_features': 'auto',
#  'min_samples_split': 2,
#  'n_estimators': 100}

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count,y)
cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()
# 0.8463886063072227

rf_final = RandomForestClassifier(max_depth=None, max_features="auto",
                                  min_samples_split=2, n_estimators=1000).fit(X_count, y)
cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()
# 0.8467955239064089

########################################
# Hyperparametre Optimizasyonu(LightGBM)
########################################

lgb_params = {"learning_rate" : [ 0.1, 0.01, 0.02],
               "n_estimators": [100,500],
               "subsample": [0.6, 0.8],
               "max_depth": [3,4],
               "min_child_samples": [5,10]}

lgb = LGBMClassifier()

lgb_final = GridSearchCV(lgb, lgb_params, cv=5, verbose=50).fit(X_count, y)
lgb_final.best_params_
# {'learning_rate': 0.1,
#  'max_depth': 4,
#  'min_child_samples': 5,
#  'n_estimators': 500,
#  'subsample': 0.6}

lgbm_final_model = LGBMClassifier(learning_rate=0.1, max_depth=4,
                                  min_child_samples=5, n_estimators=500, subsample=0.6).fit(X_count, y)
cross_val_score(lgbm_final_model, X_count, y, cv=5, n_jobs=-1).mean()
