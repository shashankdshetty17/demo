import pandas as pd
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


models = {
 "SGDClassifier":"newmodel/MODEL/SGDClassifier_model.pkl",
 "CatBoostClassifier":"newmodel/MODEL/CatBoostClassifier_model.pkl",
 "LinearSVC":"newmodel/MODEL/LinearSVC_model.pkl",
 "RidgeClassifier":"newmodel/MODEL/RidgeClassifier_model.pkl",
 "LGBMClassifier":"newmodel/MODEL/LGBMClassifier_model.pkl",
 "XGBClassifier":"newmodel/MODEL/XGBClassifier_model.pkl",
 "RandomForestClassifier":"newmodel/MODEL/RandomForestClassifier_model.pkl",
 "MultinomialNB":"newmodel/MODEL/MultinomialNB_model.pkl",
 "BaggingClassifier":"newmodel/MODEL/BaggingClassifier_model.pkl",
 "AdaBoostClassifier":"newmodel/MODEL/AdaBoostClassifier_model.pkl"
}
#All Methods
def punctuation_to_features(df, column):
    df[column] = df[column].replace('!', ' exclamation ')
    df[column] = df[column].replace('?', ' question ')
    df[column] = df[column].replace('\'', ' quotation ')
    df[column] = df[column].replace('\"', ' quotation ')
    
    return df[column]

def tokenize(column):
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]

def remove_stopwords(tokenized_column):
    stops = set(stopwords.words("english"))
    return [word for word in tokenized_column if not word in stops]

def apply_stemming(tokenized_column):
    
    stemmer = PorterStemmer() 
    return [stemmer.stem(word).lower() for word in tokenized_column]

def rejoin_words(tokenized_column):
    return ( " ".join(tokenized_column))

#Main Code
def predictnew(filename,model):
    df=pd.read_csv("scraper/FILE/"+filename)

    df['text'] = df['review_text'].str.replace('\n', ' ')
    
    df['text'] = punctuation_to_features(df, 'text')
    
    df['tokenized'] = df.apply(lambda x: tokenize(x['text']), axis=1)
    
    
    df['stopwords_removed'] = df.apply(lambda x: remove_stopwords(x['tokenized']), axis=1)
    
    
    df['porter_stemmed'] = df.apply(lambda x: apply_stemming(x['stopwords_removed']), axis=1)
    
    df['all_text'] = df.apply(lambda x: rejoin_words(x['porter_stemmed']), axis=1)
    print(df[['all_text']].head())
    
    models_count = model
    for i, (model_name, model_file) in enumerate(models.items()):
        model = pickle.load(open(model_file, 'rb'))
        predictions = model.predict(df["text"])
        count_or = np.count_nonzero(predictions == 0)
        count_cg = np.count_nonzero(predictions == 1)
        print(f"Model {i+1} - {model_name}:", predictions, count_cg, count_or)
        if model_name in df.columns:
            df = df.drop(model_name, axis=1)
        df[model_name] = predictions
    df = df.drop(df.columns[0], axis=1)
    last_columns = df.iloc[:, -models_count:]
    df['Result'] = last_columns.apply(lambda x: x.value_counts().idxmax(), axis=1)
    df.to_csv("predict/"+filename)
    return df
    