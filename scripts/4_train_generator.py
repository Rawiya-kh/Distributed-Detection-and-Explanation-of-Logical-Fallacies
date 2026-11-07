# ============================================================================
# SIMPLE EXPLANATION GENERATOR - 
# This uses a template-based approach with your trained classifier
# ============================================================================

import json
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Set paths
BASE_DIR = "/content"
DATA_PROCESSED = f"{BASE_DIR}/data/processed"
CLASSIFIER_DIR = f"{BASE_DIR}/models/classifier"  
OUTPUT_DIR = f"{BASE_DIR}/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("SIMPLE EXPLANATION GENERATOR (Using Trained distilbert)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# LOAD YOUR TRAINED CLASSIFIER
# ============================================================================
print("\n📂 Loading your trained distilbert classifier...")

if not os.path.exists(f"{CLASSIFIER_DIR}/label_map.json"):
    print(f"❌ Classifier not found at {CLASSIFIER_DIR}")
    print("Please train the classifier first!")
    raise FileNotFoundError(f"Classifier not found at {CLASSIFIER_DIR}")

tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
model = model.to(device)
model.eval()

with open(f"{CLASSIFIER_DIR}/label_map.json", 'r') as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}
print(f"Loaded classifier with {len(label_map)} classes")
print(f"Classes: {list(label_map.keys())}")

# ============================================================================
# EXPLANATION TEMPLATES (Same as the paper's style)
# ============================================================================
EXPLANATION_TEMPLATES = {
    'ad_hominem': {
        'template': "This argument commits the AD HOMINEM fallacy. Instead of addressing the argument about {topic}, it attacks the character of {person}, which is logically irrelevant to whether the claim is true.",
        'example': "Example: 'You can't trust Dr. Smith's research because he drives an SUV' attacks the person, not the research."
    },
    'appeal_to_emotion': {
        'template': "This argument commits the APPEAL TO EMOTION fallacy. It uses {emotion} to manipulate the audience instead of providing logical evidence for {claim}.",
        'example': "Example: 'Think of the starving children!' evokes pity instead of providing evidence."
    },
    'appeal_to_popularity': {
        'template': "This argument commits the APPEAL TO POPULARITY fallacy. It assumes that because {many_people} believe or do {thing}, it must be true or correct.",
        'example': "Example: 'Everyone is buying this product, so it must be the best' uses popularity as evidence."
    },
    'circular_reasoning': {
        'template': "This argument commits CIRCULAR REASONING. It uses {claim} as both a premise and the conclusion, creating a logical loop without external evidence.",
        'example': "Example: 'The Bible is true because it says so in the Bible' proves nothing."
    },
    'equivocation': {
        'template': "This argument commits EQUIVOCATION. It uses the term '{term}' in two different senses, leading to a misleading conclusion.",
        'example': "Example: 'A feather is light, so it can't be heavy' uses 'light' in two different ways."
    },
    'fallacy_of_credibility': {
        'template': "This argument commits the FALLACY OF CREDIBILITY (Appeal to False Authority). It relies on {source} who is not a reliable expert on {topic}.",
        'example': "Example: 'My actor friend says this medicine works' - actors aren't medical experts."
    },
    'false_cause': {
        'template': "This argument commits the FALSE CAUSE fallacy (Post Hoc Ergo Propter Hoc). It assumes that because {event_a} happened before {event_b}, {event_a} caused {event_b}.",
        'example': "Example: 'I wore my lucky socks and we won, so the socks caused the win' confuses correlation with causation."
    },
    'false_dilemma': {
        'template': "This argument commits the FALSE DILEMMA fallacy. It presents only {option_a} and {option_b} as options, ignoring other valid possibilities.",
        'example': "Example: 'You're either with us or against us' ignores neutral positions."
    },
    'faulty_generalization': {
        'template': "This argument commits the FAULTY GENERALIZATION (Hasty Generalization) fallacy. It draws a conclusion about all {group} based on only {count} example(s).",
        'example': "Example: 'I met two rude New Yorkers, so everyone there is rude' - insufficient sample size."
    },
    'intentional': {
        'template': "This argument appears INTENTIONALLY MISLEADING. The reasoning is deliberately structured to deceive rather than inform.",
        'example': "Example: Cherry-picking data to support a predetermined conclusion."
    },
    'logical_fallacy': {
        'template': "This argument contains a LOGICAL FALLACY. The conclusion does not logically follow from the premises provided.",
        'example': "Example: 'If A then B, B is true, therefore A is true' - affirming the consequent."
    },
    'relevance_fallacy': {
        'template': "This argument commits a RELEVANCE FALLACY (Red Herring). It introduces irrelevant information about {distraction} to divert attention from the main issue.",
        'example': "Example: 'Why worry about pollution when people are starving?' - changes the subject."
    },
    'straw_man': {
        'template': "This argument commits the STRAW MAN fallacy. It misrepresents the original position on {topic} as {misrepresentation}, then attacks this distorted version.",
        'example': "Example: 'You want to lower military spending? So you want us defenseless!' - exaggerates the position."
    }
}

# ============================================================================
# SIMPLE TEMPLATE-BASED GENERATOR
# ============================================================================
class SimpleExplanationGenerator:
    def __init__(self, model, tokenizer, label_map, templates):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.id_to_label = {v: k for k, v in label_map.items()}
        self.templates = templates
        self.device = next(model.parameters()).device
    
    def extract_keywords(self, text):
        """Extract simple keywords from text for template filling"""
        words = text.lower().split()
        
        # Find potential topics (nouns after "about", "regarding", etc.)
        topics = []
        for i, word in enumerate(words):
            if word in ['about', 'regarding', 'concerning', 'on'] and i + 1 < len(words):
                topics.append(words[i + 1])
        
        # Find potential persons (names starting with capital letters in original)
        import re
        persons = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', text)
        
        return {
            'topic': topics[0] if topics else "the issue",
            'person': persons[0] if persons else "the speaker",
            'emotion': "fear or pity",
            'claim': text[:50] + "...",
            'many_people': "many people",
            'thing': "something",
            'event_a': "one event",
            'event_b': "another event",
            'option_a': "one option",
            'option_b': "another option",
            'group': "the group",
            'count': "a few",
            'term': "a key term",
            'source': "an unqualified source",
            'distraction': "something irrelevant",
            'misrepresentation': "an exaggerated version"
        }
    
    def generate_explanation(self, text, fallacy_label, confidence):
        """Generate explanation using templates"""
        
        if fallacy_label not in self.templates:
            fallacy_label = 'logical_fallacy'
        
        template_info = self.templates[fallacy_label]
        keywords = self.extract_keywords(text)
        
        try:
            explanation = template_info['template'].format(**keywords)
        except KeyError:
            explanation = template_info['template']
            for key, value in keywords.items():
                explanation = explanation.replace(f"{{{key}}}", str(value))
        
        # Add confidence level
        if confidence > 0.9:
            confidence_text = " (Very high confidence)"
        elif confidence > 0.7:
            confidence_text = " (High confidence)"
        elif confidence > 0.5:
            confidence_text = " (Moderate confidence)"
        else:
            confidence_text = " (Low confidence)"
        
        return f"{explanation}{confidence_text}\n\n{template_info['example']}"
    
    def predict_and_explain(self, text):
        """Predict fallacy and generate explanation"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = outputs.logits.argmax().item()
            confidence = probs[0][pred_id].item()
        
        fallacy = self.id_to_label[pred_id]
        explanation = self.generate_explanation(text, fallacy, confidence)
        
        return {
            'text': text,
            'predicted_fallacy': fallacy,
            'confidence': confidence,
            'explanation': explanation
        }


# ============================================================================
# LOAD TEST DATA AND EVALUATE
# ============================================================================
print("\n📂 Loading test data...")

test_df = pd.read_csv(f"{DATA_PROCESSED}/test.csv")
test_labels = test_df['label'].map(label_map).fillna(0).astype(int).values
test_texts = test_df['text'].astype(str).tolist()

print(f"Test examples: {len(test_df)}")

# Initialize generator
generator = SimpleExplanationGenerator(model, tokenizer, label_map, EXPLANATION_TEMPLATES)

# ============================================================================
# GENERATE EXPLANATIONS FOR TEST SET
# ============================================================================
print("\n🚀 Generating explanations for test set...")
print("="*60)

results = []
correct = 0

for i, (text, true_label) in enumerate(tqdm(zip(test_texts, test_labels), total=len(test_texts))):
    result = generator.predict_and_explain(text)
    results.append(result)
    
    if result['predicted_fallacy'] == id_to_label[true_label]:
        correct += 1
    
    # Print first 10 examples
    if i < 10:
        print(f"\n{'='*50}")
        print(f"📝 Text: {text[:150]}...")
        print(f"🎯 Predicted: {result['predicted_fallacy']} (conf: {result['confidence']:.3f})")
        print(f"💡 Explanation: {result['explanation'][:200]}...")

accuracy = correct / len(test_texts)
print(f"\n📊 Accuracy on test set: {accuracy:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n💾 Saving results...")

output_path = f"{OUTPUT_DIR}/explanations.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} explanations to {output_path}")

# Save predictions CSV
predictions_df = pd.DataFrame([{
    'text': r['text'],
    'predicted_fallacy': r['predicted_fallacy'],
    'confidence': r['confidence'],
    'explanation': r['explanation']
} for r in results])

predictions_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
print(f"Saved predictions to {OUTPUT_DIR}/predictions.csv")

# ============================================================================
# DEMO: Interactive Mode
# ============================================================================
print("\n" + "="*60)
print("💬 INTERACTIVE MODE")
print("="*60)

while True:
    user_input = input("\n📝 Enter an argument (or 'quit'): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    result = generator.predict_and_explain(user_input)
    
    print(f"\n🎯 Fallacy: {result['predicted_fallacy']}")
    print(f"📊 Confidence: {result['confidence']:.3f}")
    print(f"\n💡 Explanation:")
    print(f"{result['explanation']}")
    print("-"*40)

print("\n" + "="*60)
print("✅ COMPLETE!")
print(f"Explanations saved to: {OUTPUT_DIR}")
print("="*60)