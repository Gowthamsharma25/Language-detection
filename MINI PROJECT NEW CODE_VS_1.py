import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tabulate import tabulate

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values. They will be dropped.\n")
    data.dropna(inplace=True)

# Function to truncate long text with dots (for display only)
def truncate_text(text, length=40):
    return text if len(text) <= length else text[:length] + '...'

# Show dataset overview
print("\nDataset Overview:")
preview_data = data.copy()
preview_data['Text'] = preview_data['Text'].apply(lambda x: truncate_text(x))
print(tabulate(preview_data.head(), headers='keys', tablefmt='fancy_grid'))

# Show language distribution
print("\nLanguage Distribution:")
print(tabulate(data["language"].value_counts().reset_index(), headers=['Language', 'Count'], tablefmt='fancy_grid'))

# Prepare data for training (use full text)
x = np.array(data["Text"])
y = np.array(data["language"])

# Use TF-IDF Vectorizer for better text representation
cv = TfidfVectorizer()
X = cv.fit_transform(x)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Show model accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

# Continuous language detection loop
while True:
    user_input = input("Enter a text to detect its language (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("\nExiting language detection. Goodbye!\n")
        break

    data = cv.transform([user_input]).toarray()
    output = model.predict(data)[0]

    # Show the result
    print("\nPrediction Result:")
    print(tabulate([[output]], headers=['Detected Language'], tablefmt='fancy_grid'))
    print("\nEnter another text or type 'exit' to stop.\n")