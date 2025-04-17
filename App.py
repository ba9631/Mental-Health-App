import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# Load the dataset
df = pd.read_csv('sen_1k.csv')  # Replace 'sen_1k.csv' with your actual CSV file path

# Data Preprocessing
def clean_text(text):
    # Clean the text by removing anything that is not a Hindi character
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return text

# Apply cleaning to the text column
df['cleaned_text'] = df['text'].apply(clean_text)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features based on your dataset size
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Target labels (sentiment: positive, negative, neutral)
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()  # Initialize the model
model.fit(X_train, y_train)  # Train the model on training data

# Stress-reduction measures in Hindi
stress_free_tips = {
    "yoga": "योग के आसन जैसे शवासन, अनुलोम-विलोम और बालासन तनाव कम करने में मदद करते हैं।",
    "meditation": "दिन में 10-15 मिनट ध्यान करने से मन को शांति और स्थिरता मिलती है।",
    "breathing": "गहरी सांस लेने की तकनीकें जैसे प्राणायाम मन को शांत करती हैं।",
    "physical_activity": "नियमित व्यायाम जैसे पैदल चलना या हल्का व्यायाम तनाव को दूर करता है।",
    "hobbies": "अपने पसंदीदा शौक जैसे चित्रकारी, संगीत सुनना, या बागवानी में समय बिताएं।",
}

# Function to predict sentiment and provide stress-free measures
def predict_sentiment(user_input):
    cleaned_input = clean_text(user_input)  # Clean the input text
    vectorized_input = vectorizer.transform([cleaned_input]).toarray()  # Vectorize the input text
    prediction = model.predict(vectorized_input)[0]  # Predict sentiment
    
    # Custom messages in Hindi
    if prediction == "Positive":
        return "व्यक्ति तनाव मुक्त है।"
    elif prediction == "Negative":
        tips = f"{stress_free_tips['yoga']}\n{stress_free_tips['meditation']}\n{stress_free_tips['breathing']}\n{stress_free_tips['physical_activity']}\n{stress_free_tips['hobbies']}"
        return f"व्यक्ति तनाव में है। तनाव को कम करने के लिए सुझाव:\n\n{tips}"
    elif prediction == "Neutral":
        return "कोई विशेष भावना नहीं पहचानी गई।"
    else:
        return "त्रुटि: भावना का अनुमान नहीं लगाया जा सका।"

# Create the Gradio Interface
def sentiment_analysis_gui(input_text):
    return predict_sentiment(input_text)

# Define the Gradio interface with improvements
interface = gr.Interface(
    fn=sentiment_analysis_gui,
    inputs=gr.Textbox(
        label="हिंदी में एक वाक्य या पैराग्राफ दर्ज करें",
        placeholder="उदाहरण: मैं आज बहुत खुश हूं।",
        lines=3,
    ),
    outputs=gr.Textbox(
        label="भावना विश्लेषण परिणाम",
        lines=8,
    ),
    title="हिंदी भावना विश्लेषण",
    description=(
        "यह उपकरण आपके हिंदी पाठ का विश्लेषण करता है और सकारात्मक, नकारात्मक, या तटस्थ भावना की पहचान करता है।"
        " यदि भावना नकारात्मक है, तो तनाव कम करने के सुझाव प्रदान करता है।\n\n"
        "उदाहरण:\n"
        "- मैं बहुत खुश हूं।\n"
        "- मुझे चिंता महसूस हो रही है।"
    ),
    theme="default",  # Default theme or any other valid theme
)

# Add footer manually using gr.Markdown inside the layout
footer_text = "यह उपकरण भावनात्मक विश्लेषण के लिए बनाया गया है। लेखक: यश शर्मा"
footer = gr.Markdown(footer_text)

# Launch the Gradio interface with footer inside the layout
interface.launch(inline=True, share=True)
