"""
NLTK Preprocessing Pipeline
============================
Core NLP module — ALL text normalization happens here.

NLTK functions used (explicitly):
  - nltk.sent_tokenize     → sentence boundary detection
  - nltk.word_tokenize     → word-level tokenization
  - nltk.pos_tag            → Part-of-Speech tagging
  - stopwords.words('english') → stopword removal
  - WordNetLemmatizer      → lemmatization with POS awareness
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from typing import List, Dict

# Download required NLTK data
NLTK_PACKAGES = [
    "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]


def ensure_nltk_data():
    """Download all required NLTK data packages."""
    for package in NLTK_PACKAGES:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass


class NLTKProcessor:
    """
    Text preprocessing pipeline using NLTK.

    Pipeline:
        1. Sentence tokenization (sent_tokenize)
        2. Word tokenization (word_tokenize)
        3. POS tagging (pos_tag)
        4. Stopword removal
        5. Lemmatization (WordNetLemmatizer)
    """

    def __init__(self, config: dict):
        ensure_nltk_data()
        self.remove_stopwords = config.get("remove_stopwords", True)
        self.apply_lemmatization = config.get("apply_lemmatization", True)
        self.apply_pos_tagging = config.get("apply_pos_tagging", True)

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert POS tag to WordNet POS for better lemmatization."""
        from nltk.corpus import wordnet
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN  # default

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK sent_tokenize."""
        return sent_tokenize(text)

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words using NLTK word_tokenize."""
        return word_tokenize(text)

    def process(self, text: str) -> str:
        """
        Run the full NLTK preprocessing pipeline.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text string.
        """
        # Step 1: Word tokenization
        tokens = word_tokenize(text.lower())

        # Step 2: POS tagging
        if self.apply_pos_tagging:
            tagged = pos_tag(tokens)
        else:
            tagged = [(token, "NN") for token in tokens]

        # Step 3: Stopword removal
        if self.remove_stopwords:
            tagged = [(word, tag) for word, tag in tagged
                      if word.isalnum() and word not in self.stop_words]
        else:
            tagged = [(word, tag) for word, tag in tagged if word.isalnum()]

        # Step 4: Lemmatization
        if self.apply_lemmatization:
            tokens = [
                self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
                for word, tag in tagged
            ]
        else:
            tokens = [word for word, tag in tagged]

        return " ".join(tokens)

    def process_to_tokens(self, text: str) -> List[str]:
        """Run pipeline and return token list instead of string."""
        processed = self.process(text)
        return processed.split()
