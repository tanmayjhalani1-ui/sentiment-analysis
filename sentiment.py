import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Sentiment Analysis")
st.markdown("Analyze the sentiment of your text - Positive, Negative, or Neutral")
st.markdown("---")

# Training data - Sample dataset
training_data = [
    # Positive sentiments
    ("I love this product, it's amazing!", 1),
    ("This is the best day ever!", 1),
    ("I'm so happy with my purchase", 1),
    ("Great quality and excellent service", 1),
    ("Absolutely wonderful experience", 1),
    ("I'm very satisfied with this", 1),
    ("This is fantastic!", 1),
    ("Love it! Highly recommend", 1),
    ("Best decision ever", 1),
    ("Excellent quality", 1),
    ("Perfect!", 1),
    ("Amazing product", 1),
    ("Very happy", 1),
    ("Great job", 1),
    ("Superb", 1),
    
    # Negative sentiments
    ("I hate this, it's terrible", 0),
    ("This is the worst day of my life", 0),
    ("I'm very disappointed", 0),
    ("Poor quality and bad service", 0),
    ("Absolutely horrible experience", 0),
    ("I'm not satisfied at all", 0),
    ("This is awful!", 0),
    ("Don't buy this, waste of money", 0),
    ("Terrible decision", 0),
    ("Horrible quality", 0),
    ("Useless!", 0),
    ("Terrible product", 0),
    ("Very unhappy", 0),
    ("Bad job", 0),
    ("Awful", 0),
    
    # Neutral sentiments
    ("The weather is cloudy today", 2),
    ("This costs $50", 2),
    ("The book has 300 pages", 2),
    ("The meeting is at 3 PM", 2),
    ("The color is blue", 2),
    ("It's a regular day", 2),
    ("The price is $25", 2),
    ("It has three parts", 2),
    ("The time is 5 o'clock", 2),
    ("This is okay", 2),
]

# Create and train model (cached)
@st.cache_resource
def create_model():
    # Prepare training data
    texts = [text for text, _ in training_data]
    labels = [label for _, label in training_data]
    
    # Create pipeline with TF-IDF and Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    # Train model
    model.fit(texts, labels)
    
    return model

model = create_model()

# Input section
st.subheader("📝 Enter Your Text")
user_text = st.text_area("Type something to analyze:", placeholder="Example: I love this product!", height=100)

st.markdown("---")

# Prediction
if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
    if user_text.strip():
        # Get prediction and probability
        prediction = model.predict([user_text])[0]
        probabilities = model.predict_proba([user_text])[0]
        
        # Map prediction to sentiment
        sentiments = {0: "😞 Negative", 1: "😊 Positive", 2: "😐 Neutral"}
        sentiment = sentiments[prediction]
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        st.markdown("---")
        st.subheader("📊 Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment", sentiment)
        
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")
        
        st.markdown("---")
        st.write("**Sentiment Distribution:**")
        
        # Show probabilities
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Negative", f"{probabilities[0]*100:.2f}%")
        
        with result_col2:
            st.metric("Positive", f"{probabilities[1]*100:.2f}%")
        
        with result_col3:
            st.metric("Neutral", f"{probabilities[2]*100:.2f}%")
        
    else:
        st.warning("⚠️ Please enter some text to analyze!")

st.markdown("---")

# Examples
st.subheader("💡 Try These Examples:")

examples = [
    "I absolutely love this! Best purchase ever!",
    "This is terrible and doesn't work at all",
    "The product costs $29.99"
]

for example in examples:
    if st.button(example, use_container_width=True):
        st.session_state.example = example