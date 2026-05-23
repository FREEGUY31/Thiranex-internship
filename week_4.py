import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Data
df = pd.read_csv("B:\\thiranex\\week 4\\Womens Clothing E-Commerce Reviews.csv")


df = df.dropna(subset=['Review Text'])

X = df['Review Text']

y = np.where(df['Rating'] >= 4, 1, 0)

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_vectors = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\n--- Project 4 Complete ---")
print(f"Model Classification Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


plt.figure(figsize=(6, 5))
sns.countplot(x=y, palette='viridis')
plt.title('Review Sentiments (0 = Bad/Neutral, 1 = Good)')
plt.xlabel('Sentiment Class')
plt.ylabel('Total Count')
plt.tight_layout()


plt.savefig('clothing_sentiment_plot.png')
plt.close()
print("Graph saved successfully as 'clothing_sentiment_plot.png'!")


test_review = ["The fit of this dress is terrible and the fabric feels extremely cheap."]
test_vector = vectorizer.transform(test_review)
pred = model.predict(test_vector)
print(f"\nLive Testing String: '{test_review[0]}'")
print(f"Predicted Tag: {'Positive (1)' if pred[0] == 1 else 'Negative (0)'}")