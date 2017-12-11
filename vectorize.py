from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# load data
text = ["The quick brown fox jumped over the lazy dog."]

# create the transform
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
#vectorizer = HashingVectorizer(n_features=20)

# tokenize and build vocab (not necessary for HashingVectorizer)
vectorizer.fit(text)

# summarize
print(vectorizer.vocabulary_)
#print(vectorizer.idf_)

# encode document
vector = vectorizer.transform(text)

# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
