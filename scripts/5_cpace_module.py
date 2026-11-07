# ============================================================================
# FILE: 05_cpace_module.py
# CPACE Contrastive Explanation Module
# ============================================================================

import json
import os
import torch
import spacy
from typing import List, Dict, Tuple
import numpy as np

class CPACEModule:
    """
    Contrastive Explanation Module based on concept extraction
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load spaCy for NER and concept extraction
        print("Loading spaCy model for CPACE...")
        self.nlp = spacy.load("en_core_web_lg")
        
        # Contrastive explanation templates
        self.templates = {
            'ad_hominem': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. Instead of addressing the logical content about {topic}, it attacks the character or personal attributes of {person}, which is irrelevant to whether the claim is true.",
            
            'appeal_to_popularity': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It assumes that because {many_people} believe or do something, it must be true or correct, whereas a sound argument would provide actual evidence.",
            
            'false_cause': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It assumes that {event_a} caused {event_b} solely because they occurred together, without demonstrating a causal mechanism.",
            
            'false_dilemma': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It presents only {option_a} and {option_b} as options, ignoring the full spectrum of alternatives that exist.",
            
            'faulty_generalization': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It draws a conclusion about all {group} based on only {count} example(s), which is insufficient evidence.",
            
            'equivocation': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It uses the term '{term}' in two different senses, leading to a misleading conclusion.",
            
            'straw_man': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It misrepresents the original position on {topic} by exaggerating or distorting it, then attacks this distorted version.",
            
            'appeal_to_emotion': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It uses emotional manipulation about {emotion_topic} instead of providing logical evidence.",
            
            'circular_reasoning': "This argument commits the {fallacy} fallacy rather than a {alt_fallacy} fallacy. It uses the conclusion about {claim} as one of its own premises.",
            
            'default': "This argument contains a logical fallacy. Unlike {alt_fallacy} which would have valid reasoning, the logic here is flawed because {reason}."
        }
    
    def extract_concepts(self, text: str) -> Dict:
        """Extract key concepts and named entities from text"""
        doc = self.nlp(text[:2000])
        
        # Extract various entity types
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        
        # Extract key clauses (first few sentences)
        clauses = [sent.text[:100] for sent in list(doc.sents)[:2]]
        
        # Extract main noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks][:3]
        
        return {
            "persons": persons[:2] if persons else ["the speaker"],
            "organizations": orgs[:2],
            "locations": gpes[:2],
            "dates": dates[:1],
            "clauses": clauses,
            "topics": noun_phrases[:2] if noun_phrases else ["the issue"],
            "count": 1
        }
    
    def get_alt_fallacy_description(self, alt_fallacy: str) -> str:
        """Get a brief description of an alternative fallacy for contrast"""
        descriptions = {
            'ad_hominem': "a fallacy that attacks the person",
            'appeal_to_popularity': "a fallacy based on popularity",
            'false_cause': "a false causal claim",
            'false_dilemma': "an oversimplified either-or argument",
            'faulty_generalization': "a hasty generalization",
            'equivocation': "an ambiguous term fallacy",
            'straw_man': "a misrepresentation fallacy",
            'circular_reasoning': "circular logic",
            'appeal_to_emotion': "an emotional appeal"
        }
        return descriptions.get(alt_fallacy, "a logical fallacy based on evidence")
    
    def generate_contrastive_explanation(self, text: str, fallacy: str, alternatives: List[str]) -> str:
        """Generate contrastive explanation using extracted concepts"""
        
        # Extract concepts from text
        concepts = self.extract_concepts(text)
        
        # Choose template
        template = self.templates.get(fallacy, self.templates['default'])
        
        # Get alternative fallacy for contrast
        alt_fallacy = alternatives[0] if alternatives else "fallacies that rely on evidence"
        alt_desc = self.get_alt_fallacy_description(alt_fallacy)
        
        # Prepare fill values
        fill_values = {
            'fallacy': fallacy.replace('_', ' ').title(),
            'alt_fallacy': alt_fallacy.replace('_', ' ') if isinstance(alt_fallacy, str) else alt_fallacy,
            'person': concepts['persons'][0] if concepts['persons'] else "the person",
            'topic': concepts['topics'][0] if concepts['topics'] else "the issue",
            'many_people': "many people",
            'event_a': concepts['clauses'][0][:50] if concepts['clauses'] else "one event",
            'event_b': concepts['clauses'][1][:50] if len(concepts['clauses']) > 1 else "another event",
            'option_a': concepts['topics'][0] if concepts['topics'] else "one option",
            'option_b': concepts['topics'][1] if len(concepts['topics']) > 1 else "another option",
            'group': concepts['organizations'][0] if concepts['organizations'] else "the group",
            'count': 1,
            'term': concepts['topics'][0][:30] if concepts['topics'] else "a term",
            'emotion_topic': concepts['topics'][0] if concepts['topics'] else "emotions",
            'claim': concepts['clauses'][0][:50] if concepts['clauses'] else "the claim",
            'reason': "the premises do not support the conclusion"
        }
        
        # Fill template
        try:
            explanation = template.format(**fill_values)
        except KeyError as e:
            # Fallback to simpler template
            explanation = f"This argument contains a {fallacy} fallacy. Unlike {alt_desc}, the reasoning here is flawed."
        
        return explanation
    
    def generate_standard_explanation(self, fallacy: str) -> str:
        """Generate simple non-contrastive explanation"""
        standard_templates = {
            'ad_hominem': "This argument attacks the person making the claim rather than addressing the claim itself.",
            'appeal_to_popularity': "This argument relies on popularity rather than evidence to support its conclusion.",
            'false_cause': "This argument assumes correlation implies causation without sufficient evidence.",
            'false_dilemma': "This argument presents only two options when more possibilities exist.",
            'faulty_generalization': "This argument draws a broad conclusion from insufficient evidence.",
            'equivocation': "This argument uses an ambiguous term in different ways.",
            'straw_man': "This argument misrepresents the opponent's position to make it easier to attack.",
            'appeal_to_emotion': "This argument uses emotional manipulation instead of logical reasoning.",
            'circular_reasoning': "This argument uses its conclusion as one of its premises.",
        }
        return standard_templates.get(fallacy, "This argument contains a logical fallacy.")


# ============================================================================
# FUSION MODULE - Combines RAG and CPACE explanations
# ============================================================================

class FusionModule:
    """Fusion module that selects the best explanation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use a small model for similarity scoring
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
    
    def compute_similarity(self, explanation: str, retrieved_passages: List[str]) -> float:
        """Compute semantic similarity between explanation and retrieved passages"""
        if not retrieved_passages:
            return 0.0
        
        # Encode explanation and passages
        exp_embedding = self.encoder.encode([explanation], normalize_embeddings=True)[0]
        
        passage_embeddings = self.encoder.encode(retrieved_passages, normalize_embeddings=True)
        
        # Compute average cosine similarity
        similarities = np.dot(passage_embeddings, exp_embedding)
        return float(similarities.mean())
    
    def select_best_explanation(self, rag_explanation: str, cpace_explanation: str, 
                                retrieved_passages: List[str]) -> Tuple[str, str]:
        """Select the best explanation based on grounding similarity"""
        
        rag_score = self.compute_similarity(rag_explanation, retrieved_passages)
        cpace_score = self.compute_similarity(cpace_explanation, retrieved_passages)
        
        if rag_score >= cpace_score:
            return rag_explanation, "RAG"
        else:
            return cpace_explanation, "CPACE"


def main():
    """Test CPACE module"""
    
    print("="*60)
    print("CPACE MODULE TEST")
    print("="*60)
    
    cpace = CPACEModule()
    
    test_cases = [
        ("You can't trust Dr. Smith's research because he drives an SUV!", "ad_hominem", ["straw_man", "appeal_to_emotion"]),
        ("Everyone is buying this product, so it must be the best.", "appeal_to_popularity", ["false_cause", "ad_hominem"]),
        ("Since the new mayor took office, crime increased. The mayor caused this.", "false_cause", ["false_dilemma", "faulty_generalization"]),
        ("You're either with us or against us.", "false_dilemma", ["ad_hominem", "appeal_to_emotion"]),
        ("I met two rude people from New York, so everyone there is rude.", "faulty_generalization", ["false_cause", "straw_man"]),
    ]
    
    for text, fallacy, alternatives in test_cases:
        print(f"\n{'='*50}")
        print(f"Text: {text}")
        print(f"Fallacy: {fallacy}")
        print(f"Alternatives: {alternatives}")
        
        explanation = cpace.generate_contrastive_explanation(text, fallacy, alternatives)
        print(f"\nCPACE Explanation:\n{explanation}")
        
        # Show extracted concepts
        concepts = cpace.extract_concepts(text)
        print(f"\nExtracted concepts: {concepts}")
    
    print("\n✅ CPACE module ready!")

if __name__ == "__main__":
    main()