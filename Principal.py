#Importação das bibliotecas utilizadas
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from string import punctuation
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Criação de um dataframe com colunas vazias
df_select = pd.DataFrame(columns=['text', 'label'])

#Importação de dados do Excel
file_name = 'Monografia.xlsx'
sheet_name = 'BASE_NEW'
df = pd.read_excel(file_name, sheet_name)
df['Subtítulo'] = df['Subtítulo'].replace(np.nan, "")

#Função de eliminação de caracteres numéricos
def drop_digits(in_str):
    digit_list = "1234567890"
    for char in digit_list:
        in_str = in_str.replace(char, "")

    return in_str
 
# Função de Stemming
def Stemming(sentence):
    stemmer = RSLPStemmer()
    phrase = []
    for word in sentence:
        phrase.append(stemmer.stem(word.lower()))
        phrase.append(" ")
    return "".join(phrase)

#Dados de treinamento
# 1 título
# 2 subtítulo
# 3 título + subtítulo
tipo_analise = 1

df_temp = df[0:277]
if tipo_analise == 1:
    df_select['text'] = df_temp['Título']
if tipo_analise == 2:
    df_select['text'] = df_temp['Subtítulo']
    
if tipo_analise == 3:
    df_select['text'] = df_temp['Título'] + ' ' + df_temp['Subtítulo']
    
df_select['label'] = df_temp['Index']

#Considerando o processo de Stemming
df_select['tokenized_sents'] = df_select.apply(lambda row: drop_digits(Stemming(nltk.word_tokenize(row['text']))), axis=1)

# Sem considerar o processo de Stemming
#df_select['tokenized_sents'] = df_select.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
#df_select['tokenized_sents'] = df_select.apply(lambda row: drop_digits(row['text']), axis=1)

stopwords = set(stopwords.words('portuguese') + list(punctuation))
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df_select['tokenized_sents'],df_select['label'],test_size = 0.10,shuffle=False)

#Criando a matriz TF-IDF
Tfidf_vect = TfidfVectorizer(stop_words =  stopwords)
Tfidf_vect = Tfidf_vect.fit(df_select['tokenized_sents'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

cv=CountVectorizer()
word_count_vector=cv.fit_transform(df_select['tokenized_sents'])
word_count_vector.shape
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

# Imprimindo valor TF-IDF
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
df_idf.sort_values(by=['idf_weights'])

#------------------------------------------------------------------------------
#Utilizando o classificador SVM
#SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
#SVM = svm.SVC(C=1.0, kernel='poly', degree=5, gamma='auto')
SVM = svm.SVC(kernel='rbf', gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
#Predições
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)
confusion_matrix(Test_Y, predictions_SVM)

#------------------------------------------------------------------------------
# Utilizando o classificador Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
#Predições
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
confusion_matrix(Test_Y, predictions_NB)

#------------------------------------------------------------------------------
#Utilizando classificador Ranodm Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
# random forest model creation
rfc = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
rfc.fit(Train_X_Tfidf,Train_Y)
#Predições
rfc_predict = rfc.predict(Test_X_Tfidf)
print("RandomForestClassifier Score -> ",accuracy_score(rfc_predict, Test_Y)*100)
confusion_matrix(Test_Y, rfc_predict)
