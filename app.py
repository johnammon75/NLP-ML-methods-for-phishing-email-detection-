import pandas as pd
import numpy as np
import re
import gensim
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_score, recall_score, mean_squared_error

import flask as fl

nltk.download('stopwords')
nltk.download('wordnet')

def lemmatizeSentence(sentence, lemmatizer):
    token_words= sentence.split()
    lemma_sentence=[]
    for word in token_words:
        if( word not in stopwords.words("english")):
            lemma_sentence.append(lemmatizer.lemmatize(word))
            lemma_sentence.append(" ")
    return "".join(lemma_sentence)

def getDFcsv(fname):
    df = pd.read_csv(fname)
    for w in range(len(df.Subject)): #index through each row of emailsubject
        #convert to lowercase
        sub = str(df['Subject'][w]).lower()
        cont = str(df['Text'][w]).lower()
        #remove punctuation
        sub = re.sub('[^a-zA-Z]', ' ', sub)
        cont = re.sub('[^a-zA-Z]', ' ', cont)
        #remove tags
        sub=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",sub)
        cont=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",cont)
        #remove digits and special chars
        sub = re.sub("(\\d|\\W)+"," ",sub)
        cont = re.sub("(\\d|\\W)+"," ",cont)
        #remove stopwords, get their base/stem representation, and convert to lower
        #Instantiate a lemmatizer object. Question: will this impact/ruin the word vector association?
        lemmatizer = WordNetLemmatizer() #lemmatizer does this: running = run
        sub = lemmatizeSentence(sub, lemmatizer)
        cont = lemmatizeSentence(cont, lemmatizer)
        df['Subject'][w] = sub
        df['Text'][w] = cont

    return df



df = getDFcsv("phish_with_labels.csv")
df['Type']=df['Type'].replace('Fraud', 0)
df['Type']=df['Type'].replace('Phishing', 0)
df['Type']=df['Type'].replace('Commercial Spam', 0)
df['Type']=df['Type'].replace('False Positives ', 1)



def createBagOfWords(df):
    X_train_sms, X_test_sms, y_train, y_test = train_test_split(df['Text'], 
                                                    df['Type'], 
                                                    random_state=42)
    #test data distribution worked correctly
    #print('Number of rows in the total set: {}'.format(df.shape[0]))
    #print('Number of rows in the training set: {}'.format(y_train.shape[0]))
    #print('Number of rows in the test set: {}'.format(y_test.shape[0]))
    
    # Instantiate the CountVectorizer method
    count_vector = CountVectorizer(ngram_range=(1, 1), lowercase = True , stop_words =  'english')

    # Fit the training data and then return the matrix
    X_train = count_vector.fit_transform(X_train_sms)


    # Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
    X_test = count_vector.transform(X_test_sms)

    #get a list of the words, or the column labels, that correspond to each word
    X_train_feature_list = count_vector.get_feature_names_out()

    #convert the training count vector to an array and then turn that array into a matrix
    doc_array =  X_train.toarray()
    frequency_matrix_X_train = pd.DataFrame((doc_array),columns = X_train_feature_list)
    return X_train, X_test, y_train, y_test
def createWord2Vec(df):
    X_train_cont, X_test_cont, y_train, y_test = train_test_split(df['Text'], 
                                                    df['Type'], 
                                                    random_state=42)
    #create a word2vec model with those words
    w2v_model = gensim.models.Word2Vec(X_train_cont,
                                   vector_size=100,
                                   window=2,
                                   min_count=0)

    #print(w2v_model.wv.most_similar('will', topn=10))

    vocab = set(w2v_model.wv.index_to_key )

    #replace the word with the word2vec vector in the training set
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in vocab]) for ls in X_train_cont], dtype = object)
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in vocab]) for ls in X_test_cont], dtype = object)

    # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
        
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    X_train_vect_avg = np.array(X_train_vect_avg)
    X_test_vect_avg = np.array(X_test_vect_avg)
    return X_train_vect_avg, X_test_vect_avg, y_train, y_test
def createTfIdf(df):
    X_train_sms, X_test_sms, y_train, y_test = train_test_split(df['Text'], 
                                                    df['Type'], 
                                                    random_state=42)
    tfIdfVectorizer = TfidfVectorizer(use_idf=True, lowercase = True , stop_words =  'english')
    X_train = tfIdfVectorizer.fit_transform(X_train_sms)
    X_test = tfIdfVectorizer.transform(X_test_sms)
    return X_train, X_test, y_train, y_test

def trainCombo(nlp, ml, df):
    valid_nlp = ["word2vec", "bagofwords","tfidf"]
    valid_ml = ["naivebayes", "randomforest"]

    if((nlp not in valid_nlp == True) | (ml not in valid_ml == True)):
        return

    #get training datasets using the specified nlp model
    if(nlp == "word2vec"):
        X_train, X_test, y_train, y_test = createWord2Vec(df)
    elif(nlp == "bagofwords"):
        X_train, X_test, y_train, y_test = createBagOfWords(df)
    elif(nlp == "tfidf"):
        X_train, X_test, y_train, y_test = createTfIdf(df)

    if(ml == "naivebayes"):
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train , y_train)
        predictions = naive_bayes.predict(X_test)
        return mean_squared_error(y_test, predictions)
    elif(ml == "randomforest"):
        rf = RandomForestClassifier()
        rf_model = rf.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

def findBestCombo(nlpmodels, mlmodels, df):
    best_MSE = 2
    best_combo = ""
    for nlp in nlpmodels:
        for ml in mlmodels:
            if(nlp == "word2vec" and ml == "naivebayes"):
                continue
            print( "training: " + nlp + " " + ml)
            curr_MSE = trainCombo(nlp, ml, df)
            print("Mean Squared Error: " + str(curr_MSE))
            if(curr_MSE < best_MSE):
                best_MSE = curr_MSE
                best_combo = nlp + " and " + ml
    return best_combo

bc = findBestCombo(["word2vec", "bagofwords", "tfidf"], ["randomforest", "naivebayes"], df)
print("best: " + bc)

#docker stuff
app = fl.Flask(__name__)

@app.route('/')
def hello_world():
    return "best: " + bc


"""""
#use chi squared test to select most important feature 
#might not work
X = df['Text', 'Subject']
y = df['Type']
chi2_selector = SelectKBest(chi2, k=1)
X_kbest = chi2_selector.fit_transform(X, y)
print(X_kbest)
selected_feature_indices = chi2_selector.get_support(indices=True)
# Create a new dataframe with only the selected features
selected_features_df = X.iloc[:, selected_feature_indices]
print("chisquared test")
print(selected_features_df)
"""

"""
#create rf model with word2vec
X_train_vect_avg_w2v, X_test_vect_avg_w2v, y_train_w2v, y_test_w2v = createWord2Vec(df)
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg_w2v, y_train_w2v)

y_pred = rf_model.predict(X_test_vect_avg_w2v)

precision = precision_score(y_test_w2v, y_pred)
recall = recall_score(y_test_w2v, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test_w2v).sum()/len(y_pred), 3)))


#create NaivesBayes model with BagofWords
X_train_bow, X_test_bow, X_train_sms_bow, X_train_sms_bow, y_train_bow, y_test_bow = createBagOfWords(df)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_bow , y_train_bow)

predictions = naive_bayes.predict(X_test_bow)
print('\nWord Count results using Naive Bayes:')
print('Accuracy score: ', format(accuracy_score(predictions,y_test_bow)))
print('Precision score: ', format(precision_score(predictions,y_test_bow)))
print('Recall score: ', format(recall_score(predictions,y_test_bow)))
print('F1 score: ', format(f1_score(predictions,y_test_bow)) + "\n \n")
"""

