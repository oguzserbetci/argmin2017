from glob import glob
import numpy as np
import spacy
from collections import defaultdict
import xml.etree.ElementTree as ET


class ArgumentTree(object):
    def __init__(self, value, text=None):
        self.children = []
        self.parent = None
        self.value = value
        self.text = text

    def __repr__(self):
        if self.children:
            return '[{}: {}]'.format(self.value, ', '.join([str(c) for c in self.children]))
        else:
            return '{}({})'.format(self.value, self.text)

    @property
    def p_value(self):
        if self.parent:
            return self.parent.value
        else:
            return self.value

    def __len__(self):
        if self.children:
            return 1 + sum([len(l) for l in self.children])
        else:
            return 0

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root

    @property
    def is_root(self):
        '''root if it does not have a parent.'''
        return not self.parent

    def search(self, value):
        if self.value == value:
            return self
        else:
            return next((c for c in (child.search(value) for child in self.children) if c is not None), None)

    def add(self, child):
        self.children.append(child)
        child.parent = self

def read_ac(xml):
    arguments = {x.get('id'):x.text for x in xml.findall('edu')}
    edges = [x for x in xml.findall('edge')]

    trees = []
    # parent (major claim) -> child (claims)
    for edge in edges:
        fr = edge.get('src')
        text = arguments.get(fr)
        to = edge.get('trg')
        while to.startswith('c'):
            to = next(edge for edge in edges if edge.get('id') == to).get('trg')

        child = next((node for node in (tree.search(fr) for tree in trees) if node is not None), ArgumentTree(fr, text))
        parent = next((node for node in (tree.search(to) for tree in trees) if node is not None), ArgumentTree(to))

        parent.add(child)
        trees.append(parent)
    return trees[0].root


nlp = spacy.load('en_core_web_lg')


class MTCorpus(object):
    def __init__(self):
        files = sorted(glob('./arg-microtexts/corpus/en/*.xml'))
        self.links = []
        self.types = []
        self.documents = []
        self.tokens = []
        self.lemmas = defaultdict(int)

        for file in files:
            with open(file,'r',encoding='UTF-8') as file:
                xml = ET.parse(file)
                ac_tree = read_ac(xml)
                y = [int(ac_tree.search('a{}'.format(i)).p_value[1:]) for i in range(1, len(ac_tree) + 1)]
                y = [i - min(y) for i in y]
                self.links.append(y)
                types = [int(ac_tree.search('a{}'.format(i)).is_root) for i in range(1, len(ac_tree) + 1)]
                self.types.append(types)
                '''Tokenizes a string.'''
                acs = [ac_tree.search('e{}'.format(i)).text for i in range(1, len(ac_tree) + 1)]
                document = [self.represent(ac) for ac in acs]
                self.documents.append(document)

        self.lemma_indices = dict(zip(self.lemmas.keys(),range(len(self.lemmas))))
        self.X_large = self._x_large()
        self.X_small = self._x_small()
        self.X_decoder = self._decoder()

    def represent(self, sent):
        # Tokenize sentence
        tokens = nlp(sent)
        # Build lemma vocabulary
        for token in tokens:
            self.lemmas[token.lemma_] += 1
        # Return tokens
        return tokens

    def _x_small(self):
        representations = []

        for document in self.documents:
            representations.append([])
            for i, ac in enumerate(document):
                vectors = []
                for token in ac:
                    vectors.append(token.vector)
                vectors = np.array(vectors)

                # min and max across token embeddings
                mean = np.mean(vectors, axis=0)
                pos = [int(i == 0)]
                r = np.concatenate([pos, mean], axis=0)
                representations[-1].append(r)

        return np.array(representations)

    def _x_large(self):
        representations = []

        for document in self.documents:
            representations.append([])
            for i, ac in enumerate(document):
                vectors = []
                bow = np.zeros(len(self.lemmas))
                for token in ac:
                    bow[self.lemma_indices[token.lemma_]] += 1
                    vectors.append(token.vector)
                vectors = np.array(vectors)

                # min and max across token embeddings
                maximum = np.max(vectors, axis=0)
                minimum = np.min(vectors, axis=0)
                mean = np.mean(vectors, axis=0)
                pos = [int(i == 0)]
                r = np.concatenate([pos, mean, maximum, minimum, bow], axis=0)
                representations[-1].append(r)

        return np.array(representations)

    def _decoder(self):
        representations = []
        for x, links in zip(self.X_large, self.links):
            representation = []
            for link in links:
                representation.append(x[link])
            representations.append(representation)
