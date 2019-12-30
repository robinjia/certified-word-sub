"""Find connected components of word vector neighbors."""
import collections
import json
import sys

def main():
  with open('data/counterfitted_neighbors.json') as f:
    neighbors = json.load(f)
  print('Read %d words' % len(neighbors))
  for w in list(neighbors):
    if not neighbors[w]:
      del neighbors[w]
  words = list(neighbors)
  # Directed edges -> Undirected edges
  edges = collections.defaultdict(set)
  for w in words:
    for v in neighbors[w]:
      edges[w].add(v)
      edges[v].add(w)
  # DFS
  visited = set()
  comps = []
  for start_word in words:
    if start_word in visited: continue
    cur_comp = set()
    stack = [start_word]
    while stack:
      w = stack.pop()
      if w in visited: continue
      visited.add(w)
      cur_comp.add(w)
      for v in edges[w]:
        if v not in visited:
          stack.append(v)
    comps.append(cur_comp)
  print('Found %d components' % len(comps))
  print('Sum of component sizes is %d' % sum(len(c) for c in comps))
  comps.sort(key=lambda x: len(x), reverse=True)
  print('Largest components: %s' % ', '.join(str(len(c)) for c in comps[:10]))
  for w in ('good', 'bad', 'awesome', 'excellent', 'terrible', 'una'):
    print('%s: visited==%s, in biggest component==%s' % (w, w in visited, w in comps[0]))

if __name__ == '__main__':
  main()

