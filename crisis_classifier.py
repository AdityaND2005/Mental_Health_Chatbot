"""
Crisis Classifier Model for detecting mental health crises
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Tuple
import config


class CrisisClassifier:
    """
    Crisis detection model for identifying high-risk mental health situations
    Uses DistilBERT with custom classification head
    """
    
    def __init__(self):
        self.device = config.DEVICE
        print(f"[CrisisClassifier] Initializing on device: {self.device}")
        
        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.CRISIS_CLASSIFIER_MODEL,
            cache_dir=config.MODELS_DIR
        )
        
        self.model = AutoModel.from_pretrained(
            config.CRISIS_CLASSIFIER_MODEL,
            cache_dir=config.MODELS_DIR
        )
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.model.to(self.device)
        self.classifier.to(self.device)
        self.model.eval()
        self.classifier.eval()
        
        print("[CrisisClassifier] Model loaded successfully")
    
    def keyword_check(self, text: str) -> bool:
        """
        Quick keyword-based check for crisis indicators
        
        Args:
            text: User input text
        
        Returns:
            True if crisis keywords detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in config.CRISIS_KEYWORDS)
    
    def classify(self, text: str) -> Tuple[bool, float]:
        """
        Classify if text indicates a mental health crisis
        
        Args:
            text: User input text to classify
        
        Returns:
            Tuple of (is_crisis: bool, confidence: float)
        """
        # First do keyword check (fast path)
        if self.keyword_check(text):
            print("[CrisisClassifier] Crisis keywords detected")
            return True, 0.95
        
        # Deep learning classification
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                # Classify
                crisis_score = self.classifier(cls_embedding).item()
            
            is_crisis = crisis_score >= config.CRISIS_CONFIDENCE_THRESHOLD
            
            if is_crisis:
                print(f"[CrisisClassifier] Crisis detected with confidence: {crisis_score:.3f}")
            
            return is_crisis, crisis_score
            
        except Exception as e:
            print(f"[CrisisClassifier] Error during classification: {e}")
            # Fallback to keyword check only
            return self.keyword_check(text), 0.8 if self.keyword_check(text) else 0.3
    
    def batch_classify(self, texts: list) -> list:
        """
        Classify multiple texts in batch
        
        Args:
            texts: List of text strings
        
        Returns:
            List of (is_crisis, confidence) tuples
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results


# Simpler rule-based classifier for very lightweight deployment
class LightweightCrisisClassifier:
    """
    Lightweight rule-based crisis classifier using keywords and patterns
    Much smaller memory footprint for mobile deployment
    """
    
    def __init__(self):
        # Extended keyword patterns with severity levels
        self.high_risk_keywords = [
            "suicide", "kill myself", "end my life", "want to die",
            "suicide plan", "kill me", "end it all"
        ]
        
        self.medium_risk_keywords = [
            "self harm", "hurt myself", "cutting", "overdose",
            "no reason to live", "better off dead", "can't go on",
            "give up", "hopeless", "worthless"
        ]
        
        self.low_risk_keywords = [
            "depressed", "anxious", "sad", "lonely", "scared",
            "worried", "stressed", "overwhelmed"
        ]
        
        print("[LightweightCrisisClassifier] Initialized rule-based classifier")
    
    def classify(self, text: str) -> Tuple[bool, float]:
        """
        Rule-based classification
        
        Args:
            text: User input text
        
        Returns:
            Tuple of (is_crisis: bool, confidence: float)
        """
        text_lower = text.lower()
        
        # Check high risk
        for keyword in self.high_risk_keywords:
            if keyword in text_lower:
                return True, 0.95
        
        # Check medium risk
        medium_count = sum(1 for kw in self.medium_risk_keywords if kw in text_lower)
        if medium_count >= 2:
            return True, 0.85
        elif medium_count == 1:
            return True, 0.75
        
        # Check low risk patterns
        low_count = sum(1 for kw in self.low_risk_keywords if kw in text_lower)
        if low_count >= 3:
            return False, 0.65  # Concerning but not immediate crisis
        
        return False, 0.3


def get_crisis_classifier(lightweight: bool = False):
    """
    Factory function to get appropriate crisis classifier
    
    Args:
        lightweight: If True, use rule-based classifier (smaller memory)
    
    Returns:
        Crisis classifier instance
    """
    if lightweight:
        return LightweightCrisisClassifier()
    else:
        try:
            return CrisisClassifier()
        except Exception as e:
            print(f"[Warning] Could not load full classifier: {e}")
            print("[Info] Falling back to lightweight classifier")
            return LightweightCrisisClassifier()
