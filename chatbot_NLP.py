# ===========================================
# Activity 3: AI-based Symptom Checker Chatbot
# Tool: Google Colab (NLP + Simple Chatbot)
# ===========================================

# Step 1: Install HuggingFace Transformers (if not already installed)
!pip install transformers

# Step 2: Import Libraries
from transformers import pipeline

# We'll use a lightweight NLP zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Step 3: Define possible medical conditions (educational demo)
possible_conditions = [
    "Common Cold",
    "Flu",
    "COVID-19",
    "Asthma",
    "Pneumonia",
    "Migraine",
    "Food Poisoning",
    "Heart Disease",
    "Allergy"
]

print("✅ Symptom Checker Chatbot is ready!")
print("Type your symptoms (e.g., 'fever, cough, chest pain')\n")

# Step 4: Chat loop (interactive input)
while True:
    symptoms = input("📝 Enter symptoms (or type 'exit' to quit): ")
    if symptoms.lower() in ["exit", "quit"]:
        print("👋 Thanks for trying the AI Symptom Checker. Remember: This is not a medical diagnosis tool!")
        break
    
    # AI predicts most likely condition(s)
    result = classifier(symptoms, candidate_labels=possible_conditions)
    
    print("\n🤖 AI Symptom Checker Results:")
    for cond, score in zip(result['labels'][:3], result['scores'][:3]):
        print(f"   {cond} → Likelihood: {score:.2f}")
    
    print("\n⚠️ Disclaimer: This is an educational demo, not medical advice.\n")
