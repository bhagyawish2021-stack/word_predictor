"""
Next Word Prediction Engine
Implements N-gram language model with frequency-based prediction
"""

import pickle
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class NextWordPredictor:
    """
    N-gram based language model for predicting next words
    Uses frequency analysis of word sequences to provide predictions
    """
    
    def __init__(self, n_gram_size: int = 3):
        """
        Initialize the predictor
        
        Args:
            n_gram_size: The size of n-grams to use (default: 3 for trigrams)
        """
        self.n_gram_size = n_gram_size
        self.n_grams = defaultdict(Counter)
        self.vocabulary = set()
        self.total_tokens = 0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess and tokenize text
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of tokens
        """
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        return [t for t in tokens if t]
    
    def _build_n_grams(self, tokens: List[str]) -> None:
        """
        Build n-gram model from tokens
        
        Args:
            tokens: List of preprocessed tokens
        """
        for i in range(len(tokens) - self.n_gram_size):
            context = tuple(tokens[i:i + self.n_gram_size - 1])
            next_word = tokens[i + self.n_gram_size - 1]
            
            self.n_grams[context][next_word] += 1
            self.vocabulary.add(next_word)
        
        self.total_tokens = len(tokens)
    
    def train(self, corpus_path: str) -> None:
        """
        Train the model on a text corpus
        
        Args:
            corpus_path: Path to the training text file
        """
        try:
            logger.info(f"Training on corpus: {corpus_path}")
            
            with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            tokens = self._preprocess_text(text)
            
            if len(tokens) < self.n_gram_size:
                logger.warning("Corpus too small, generating default predictions")
                self._create_default_model()
                return
            
            self._build_n_grams(tokens)
            logger.info(f"Training complete. Vocabulary size: {len(self.vocabulary)}")
            
        except FileNotFoundError:
            logger.warning(f"Corpus file not found: {corpus_path}")
            self._create_default_model()
    
    def _create_default_model(self) -> None:
        """Create a default model with common words if no corpus is available"""
        common_sequences = [
            (("i", "am"), {"going": 5, "learning": 4, "trying": 3, "happy": 3, "sorry": 2}),
            (("how", "are"), {"you": 10, "they": 3, "we": 2, "things": 2}),
            (("i", "like"), {"python": 5, "coding": 4, "music": 3, "you": 3, "it": 2}),
            (("the", "next"), {"day": 4, "time": 3, "step": 3, "chapter": 2, "one": 2}),
            (("would", "you"), {"like": 5, "prefer": 3, "mind": 2, "help": 2}),
            (("please", "let"), {"me": 8, "us": 2}),
            (("is", "very"), {"important": 5, "useful": 3, "good": 3, "bad": 2}),
            (("what", "is"), {"your": 4, "this": 4, "that": 3, "it": 2}),
            (("do", "you"), {"know": 5, "think": 4, "understand": 3, "see": 2}),
            (("going", "to"), {"sleep": 5, "work": 4, "school": 4, "eat": 3, "run": 2}),
        ]
        
        for context, words in common_sequences:
            self.n_grams[context] = Counter(words)
            self.vocabulary.update(words.keys())
        
        self.total_tokens = 1000
        logger.info("Default model created")
    
    def predict(self, text: str, num_predictions: int = 5) -> List[Dict]:
        """
        Predict next words based on input text
        
        Args:
            text: Input text to predict from
            num_predictions: Number of predictions to return
            
        Returns:
            List of dicts with 'word' and 'confidence' keys
        """
        tokens = self._preprocess_text(text)
        
        if len(tokens) < self.n_gram_size - 1:
            return self._get_frequent_words(num_predictions)
        
        context = tuple(tokens[-(self.n_gram_size - 1):])
        
        if context in self.n_grams:
            candidates = self.n_grams[context]
        else:
            fallback_context = tuple(tokens[-1:])
            candidates = self._find_fallback_predictions(fallback_context)
        
        if not candidates:
            return self._get_frequent_words(num_predictions)
        
        total = sum(candidates.values())
        predictions = [
            {
                'word': word,
                'confidence': round(count / total, 3)
            }
            for word, count in candidates.most_common(num_predictions)
        ]
        
        return predictions
    
    def _find_fallback_predictions(self, context: Tuple) -> Counter:
        """
        Find predictions using shorter context
        
        Args:
            context: Tuple of words
            
        Returns:
            Counter of candidate words
        """
        candidates = Counter()
        
        for n_gram_context, words in self.n_grams.items():
            if n_gram_context[-1:] == context:
                candidates.update(words)
        
        return candidates
    
    def _get_frequent_words(self, num_words: int = 5) -> List[Dict]:
        """
        Get most frequent words in vocabulary
        
        Args:
            num_words: Number of words to return
            
        Returns:
            List of dicts with 'word' and 'confidence' keys
        """
        all_words = Counter()
        for words_counter in self.n_grams.values():
            all_words.update(words_counter)
        
        if not all_words:
            return []
        
        total = sum(all_words.values())
        return [
            {
                'word': word,
                'confidence': round(count / total, 3)
            }
            for word, count in all_words.most_common(num_words)
        ]
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    @staticmethod
    def load(filepath: str) -> 'NextWordPredictor':
        """
        Load a model from a file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded NextWordPredictor instance
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return NextWordPredictor()
