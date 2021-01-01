import pytest
from models import * 

SENTENCE_EXAMPLE = {
    "review_text": [["This", "is", "a", "sentence", "."],
                    ["<br>"],
                    ["It", "is", "part", "of", "a", "review", "."]]
    }

SENTENCE_REVIEW_TEXT = [
  ["This", "is", "a", "sentence", "."], ["It", "is", "part", "of", "a",
  "review", "."]]

def test_get_text_from_example():
  assert get_text_from_example(
      SENTENCE_EXAMPLE, "review_text") == SENTENCE_REVIEW_TEXT

def test_token_indexizer():
  correct_result = {0: list(range(5)), 1: list(range(5, 12))} 
  assert token_indexizer(SENTENCE_REVIEW_TEXT, "sentence") == correct_result


def test_similarities():
  assert TfIdfModel.sim({0:1, 1:1}, {0:2, 1:2}) == pytest.approx(1.0)
  assert TfIdfModel.sim({0:1, 1:1}, {3:2, 7:2}) == pytest.approx(0.0)
  assert SentenceBERTModel.sim(
      np.array([1,2,3,4,5]), np.array([2,4,6,8,10])
      ) == pytest.approx(1.0)
  assert SentenceBERTModel.sim(
      np.array([1,2]), np.array([2,0])
      ) == pytest.approx(0.447213595)
