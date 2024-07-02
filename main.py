import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from datetime import datetime
import spacy
from textblob import TextBlob

# Download NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for NER
nlp = spacy.load('en_core_web_sm')

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords, then stem the tokens
    tokens = [stemmer.stem(word) for word in tokens if word not in string.punctuation and word not in stop_words]
    return tokens

def named_entity_recognition(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

def chatbot_response(user_input):
    tokens = preprocess_text(user_input)
    entities = named_entity_recognition(user_input)
    sentiment = analyze_sentiment(user_input)

    if any(token in tokens for token in ["hello", "hi", "hey"]):
        return "Hello! How can I help you today?"
    if any(token in tokens for token in ["how", "you","are"]):
        if sentiment == "positive":
            return "I'm glad to hear that! How can I assist you?"
        elif sentiment == "negative":
            return "I'm sorry to hear that. How can I assist you?"
        else:
            return "I'm just a bot, but I'm doing great! How can I assist you?"
    elif any(token in tokens for token in ["name"]):
        if "PERSON" in entities:
            return f"Nice to meet you, {entities['PERSON']}! I'm ChatNLP, your friendly chatbot."
        else:
            return "I'm ChatNLP, your friendly chatbot. What's your name?"
    elif "time" in tokens:
        now = datetime.now()
        return f"The current time is {now.strftime('%H:%M:%S')}."
    elif "date" in tokens:
        today = datetime.today()
        return f"Today's date is {today.strftime('%Y-%m-%d')}."
    elif any(token in tokens for token in ["bye", "goodbye"]):
        return "Goodbye! Have a great day!"
    elif any(token in tokens for token in ["thank", "thanks"]):
        return "You're welcome! If you have any more questions, feel free to ask."
    elif any(token in tokens for token in ["weather"]):
        return "I'm not connected to the internet, so I can't provide real-time weather updates, but it's always a good idea to check a reliable weather website or app."
    elif "joke" in tokens:
        return "Why don't scientists trust atoms? Because they make up everything!"
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase?"

# Main loop
if __name__ == "__main__":
    print("ChatNLP: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "goodbye"]:
            print("ChatNLP: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print(f"ChatNLP: {response}")
