from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# split into sentences
sentences = sent_tokenize(text)
print(sentences[0])

# split into words
tokens = word_tokenize(text)

#stop_words = stopwords.words('english')
#print(stop_words)

## remove all tokens that are not alphanumeric
#words = [word for word in tokens if word.isalpha()]
#print(words[:100])

# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])
