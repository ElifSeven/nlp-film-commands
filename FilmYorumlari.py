import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


#dataset yükle
df = pd.read_csv("IMDB Dataset.csv")
df.head()
'''
#önişlem
from bs4 import BeautifulSoup

def removeHTML(text):
    soup = BeautifulSoup(text,"html.parser")
    return soup.get_text()

#print(removeHTML(df.iloc[0][0])) # dataframeler listeler gibi çalışmıyor. İlk ssütun sonra satır seçiyoruz

maxObs = 20
observations = ''
for n in range(maxObs):
    observations += removeHTML(df.iloc[n][0])
observationsClean = ','.join(i.lower() for i in observations.split() if i.isalnum())
setOfWords = set(observationsClean.split(','))
print(setOfWords)
'''
'''
# kelime dağarcığı
from tqdm import tqdm

dictList= []
for n in tqdm(range(len(df))):
    observations = removeHTML(df.iloc[n][0])
    clean = ','.join(i.lower() for i in observations.split() if i.isalnum())
    dictOfWords = dict.fromkeys(setOfWords,0)
    for word in clean.split(','):
        if word in dictOfWords:
            dictOfWords[word] += 1
    dictList.append(dictOfWords)

pd.DataFrame(dictList).sample(10)
'''
# library ile kelime dağarcığı hazırlama
y = df.sentiment.replace({"positive":1,"negative":0})
x = df.review
bag = CountVectorizer()
X = bag.fit_transform(x)

# Train
X_train, X_test, y_train, y_test = train_test_split(X,y)
clf = RandomForestClassifier(n_jobs = -1) # hepsini kullanacak
clf.fit(X_train,y_train)

# predict
pred = clf.predict(X_test)
cm = confusion_matrix(y_test,pred)


test1 = "I really did not enjoy watching this. Very disappointed"
test2 = "What a wonderful movie. I enjoyed watching this with my kids"
clf.predict(bag.transform([test1]))
clf.predict(bag.transform([test2]))



