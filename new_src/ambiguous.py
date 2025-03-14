import nltk
import spacy
from nltk.corpus import wordnet as wn

class Ambiguous():
    def __init__(self):
        self.spacyModel = spacy.load("en_core_web_sm")

    def find_main_pos(self, sentence):
        doc = self.spacyModel(sentence)
        main_verb, main_noun = None, None
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                main_verb = token.text
            if token.dep_ in ("nsubj", "nsubjpass"):
                main_noun = token.text
        return main_noun, main_verb

    def make_ambiguous(self, sentence):
        doc = self.spacyModel(sentence)
        main_noun, main_verb = self.find_main_pos(sentence)
        
        def replace_token(token, replace_func):
            wn_pos = self._get_wordnet_pos(token.pos_)
            return replace_func(token.text, wn_pos) if wn_pos else token.text
        
        transformations = {
            "subject_synonym": lambda token: replace_token(token, self._get_synonym) if token.text == main_noun else token.text,
            "subject_hypernym": lambda token: replace_token(token, self._get_hypernym) if token.text == main_noun else token.text,
            "subject_homonym": lambda token: replace_token(token, self._get_homonym) if token.text == main_noun else token.text,
            "verb_synonym": lambda token: replace_token(token, self._get_synonym) if token.text == main_verb else token.text,
            "verb_hypernym": lambda token: replace_token(token, self._get_hypernym) if token.text == main_verb else token.text,
            "verb_homonym": lambda token: replace_token(token, self._get_homonym) if token.text == main_verb else token.text,
            "nouns_except_subject_synonym": lambda token: replace_token(token, self._get_synonym) if token.pos_ == "NOUN" and token.text != main_noun else token.text,
            "nouns_except_subject_hypernym": lambda token: replace_token(token, self._get_hypernym) if token.pos_ == "NOUN" and token.text != main_noun else token.text,
            "nouns_except_subject_hyponym": lambda token: replace_token(token, self._get_hyponym) if token.pos_ == "NOUN" and token.text != main_noun else token.text,
            "verbs_except_main_synonym": lambda token: replace_token(token, self._get_synonym) if token.pos_ == "VERB" and token.text != main_verb else token.text,
            "verbs_except_main_hypernym": lambda token: replace_token(token, self._get_hypernym) if token.pos_ == "VERB" and token.text != main_verb else token.text,
            "verbs_except_main_hyponym": lambda token: replace_token(token, self._get_hyponym) if token.pos_ == "VERB" and token.text != main_verb else token.text,
        }
        
        results = {key: ' '.join([transform(token) for token in doc]) for key, transform in transformations.items()}
        return results

    def _get_wordnet_pos(self, spacy_pos):
        return {'NOUN': wn.NOUN, 'VERB': wn.VERB}.get(spacy_pos, None)

    def _get_synonym(self, word, pos):
        synsets = wn.synsets(word, pos=pos)
        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    return lemma.name().replace('_', ' ')
        return word

    def _get_hypernym(self, word, pos):
        synsets = wn.synsets(word, pos=pos)
        if synsets and synsets[0].hypernyms():
            return synsets[0].hypernyms()[0].lemmas()[0].name().replace('_', ' ')
        return word

    def _get_hyponym(self, word, pos):
        synsets = wn.synsets(word, pos=pos)
        if synsets and synsets[0].hyponyms():
            return synsets[0].hyponyms()[0].lemmas()[0].name().replace('_', ' ')
        return word

    def _get_homonym(self, word, pos):
        synsets = wn.synsets(word, pos=pos)
        return synsets[1].lemmas()[0].name().replace('_', ' ') if len(synsets) > 1 else word

