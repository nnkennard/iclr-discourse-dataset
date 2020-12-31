import collections
import json
import math 
import sys


WINDOW = 2
Q = 2124749677  # Is this too big for Python int


def myhash(tokens):
  tok_str = "".join(tokens)
  hash_acc = 0
  for i, ch in enumerate(reversed(tok_str)):
    hash_acc += math.pow(2, i) * ord(ch)

  return hash_acc % Q


def get_hashes(tokens):
  return {i:myhash(tokens[i:i+WINDOW]) for i in range(len(tokens) - WINDOW)}


def expand_match(match, tokens_1, tokens_2):
  i = WINDOW
  j_1, j_2 = match
  while (tokens_1[j_1:j_1 + i]
      == tokens_2[j_2: j_2 + i]):
      i += 1
      if j_1 + i == len(tokens_1) or j_2 + i == len(tokens_2):
        break
  return i-1

def karp_rabin(tokens_1, tokens_2):
  hashes_1 = get_hashes(tokens_1)
  hashes_2 = get_hashes(tokens_2)
  matches = []
  for k1, v1 in hashes_1.items():
    for k2, v2 in hashes_2.items():
      if v1 == v2:
        if tokens_1[k1:k1+WINDOW] == tokens_2[k2:k2+WINDOW]:
          matches.append((k1, k2))
  final_matches = []
  if matches:
    expanded_matches = []
    for match in matches:
      i, j = match
      expanded_match = expand_match(match, tokens_1, tokens_2)
      tokens = tokens_1[i:i+expanded_match]
      assert tokens == tokens_2[j:j+expanded_match]
      expanded_matches.append(MiniMatch(i, j, expanded_match, tokens))
    sorted_expanded = sorted(expanded_matches, key=lambda x:x.review_start)
    final_matches.append(sorted_expanded[0])
    for match in sorted_expanded[1:]:
      if match.review_start < final_matches[-1].review_start + len(
          final_matches[-1].tokens):
        continue
      final_matches.append(match)
  return final_matches

MiniMatch = collections.namedtuple("MiniMatch",
    "review_start rebuttal_start len tokens".split())


