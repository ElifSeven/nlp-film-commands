import pandas as pd

#dataset yükle
df = pd.read_csv("IMDB Dataset.csv")
df.head()
#önişlem
from bs4 import BeautifulSoup

def removeHTML(text):
    soup = BeautifulSoup(text,"html.parser")
    return soup.get_text()

print(removeHTML(df.iloc[0][0])) # dataframeler listeler gibi çalışmıyor. İlk ssütun sonra satır seçiyoruz
