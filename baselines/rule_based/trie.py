import tqdm 

class Trie(object):
  def __init__(self, input_strings):
    self.root = TrieNode("^", False, None, "^")
    self.all_nodes = []
    for input_string in input_strings:
      self.insert_string(input_string)

  def insert_string(self, input_string):
    current_node = self.root
    i = 0
    for char in input_string:
      if char in current_node.children:
        current_node.count += 1
        current_node = current_node.children[char]
      else:
        new_node = TrieNode(char, False, current_node, current_node.prefix +
            char)
        self.all_nodes.append(new_node)
        current_node.children[char] = new_node
        current_node = new_node
    current_node.make_final()

  def __str__(self):
    str_rep = ""
    for node in reversed(sorted(self.all_nodes, key = lambda x:x.count)):
      if len(node.children) == 1:
        continue
      if node.count == 1:
        break
      str_rep += node.prefix + "\t" + str(node.count) + "\n"
    return str_rep


class TrieNode(object):
  def __init__(self, char, is_final, parent, prefix):
    self.char = char
    self.is_final = is_final
    self.parent = parent
    self.count = 1
    self.children = {}
    self.prefix = prefix

  def make_final(self):
    self.is_final = True



