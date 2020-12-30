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
  assert TfidfModel.sim({0:1, 1:1}, {0:2, 1:2}) == 1.0
