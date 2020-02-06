from nltk.corpus import wordnet
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import pandas as pd

def word2vec(x):
    try:
        token=word_tokenize(x)
        token=[word for word in token if word not in english_punctuations]
        token=[porter_stemmer.stem(word.lower()) for word in token]
        temlist=[]
        for word in token:
            if word in allword:
                temlist.append(word)
            else:
                for sys in wordnet.synsets(word):
                    if sys.name().split('.')[0] in allword:
                        temlist.append(sys.name().split('.')[0])
        token=temlist
        vec=onehot_encoder.transform(label_encoder.transform(token).reshape(len(token),1))
        vec=np.array(vec)
        vec=vec.sum(axis=0)
        vec=vec.tolist()
        return vec
    except(TypeError):
        return None

porter_stemmer = PorterStemmer()
train_data=pd.read_csv('incident_label.csv',sep='\t')
alltitle=' '.join(train_data['title'].values)
allword=word_tokenize(alltitle)
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\'','|','-']
stops = set(stopwords.words("english"))
allword = [word.lower() for word in allword if word.lower() not in stops and word not in english_punctuations]
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(allword)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
train_data['vec']=train_data['title'].apply(word2vec)
X=train_data['vec'].values
Y=train_data['lbael'].values

model = tree.DecisionTreeClassifier(max_depth=18)
y_train_pred = cross_val_predict(model, X, Y, n_jobs=16,cv=10)
print(classification_report(Y,y_train_pred,digits=4))
model.fit(X,Y)
classify_data=pd.read_csv('reuter_incident.csv',sep='\t')
classify_data['vec']=classify_data['title'].apply(word2vec)
Y_p=model.predict(classify_data['vec'].values)
classify_data['Label']=Y_p
classify_data.to_csv('reuter_incident_classify.csv',sep='\t')