import pytest
from collections import Counter
from scipy.sparse.linalg import svds
import en_core_web_lg

from project3 import (extract_state_city_name, get_most_common_words,
                              remove_most_common_words,
                              remove_city_state_names, correct_words)


def test_extract_state_city_name():
    assert extract_state_city_name("NY Albany Troy Schenectady Saratoga Springs.pdf") == ("NY", "Albany Troy Schenectady Saratoga Springs")
    assert extract_state_city_name("DC Washington.pdf") == ("DC", "Washington, D.C")
    assert extract_state_city_name("TX Lubbock.pdf") == ("TX", "Lubbock")

def test_get_most_common_words():
    text = "the rings the face the love the fairy the rings the rings"
    assert get_most_common_words(text, n=2) == ["the", "rings"]

def test_remove_most_common_words():
    text = "the rings the face the love the fairy the rings the rings"
    most_common_words = ["the", "rings"]
    assert remove_most_common_words(text, most_common_words) == "face love fairy"

def test_correct_words():
    nlp = en_core_web_lg.load()
    words = ["mother", "father", "child", "slfj"]
    corrected_words = correct_words(words, nlp)
    assert corrected_words == ["mother", "father", "child"]

def test_remove_city_state_names():
    top_words = ["project", "tulsa", "tulsa", "oklahoma", "cars"]
    city_names = ["tulsa"]
    state_names = ["oklahoma"]
    filtered_top_words = remove_city_state_names(top_words, city_names, state_names)
    assert filtered_top_words == ["project", "cars"]