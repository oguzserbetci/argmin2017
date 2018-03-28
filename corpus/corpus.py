from glob import glob
import numpy as np
import spacy
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer


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


nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])


class MTCorpus(object):
    def __init__(self):
        files = sorted(glob('./arg-microtexts/corpus/en_extended/*.xml'))
        self.links = []
        self.types = []
        self.documents = []
        self.vectors = []

        for file in files:
            with open(file,'r',encoding='UTF-8') as file:
                xml = ET.parse(file)
                # get the start index to acccount for difference between new and old
                # corpus
                base = xml.findall('edu')
                base = int(base[0].get('id')[1])
                ac_tree = read_ac(xml)

                y = []
                cont = False
                for i in range(base, len(ac_tree) + base):
                    try:
                        y.append(int(ac_tree.search('a{}'.format(i)).p_value[1:]) - base)
                    except:
                        cont = True
                        break

                types = []
                for i in range(base, len(ac_tree) + base):
                    try:
                        types.append(int(ac_tree.search('a{}'.format(i)).is_root))
                    except:
                        cont = True
                        break

                '''Tokenizes a string.'''
                acs = ['0']  # start token for decoder input
                for i in range(base, len(ac_tree) + base):
                    try:
                        acs += [ac_tree.search('e{}'.format(i)).text]
                    except:
                        cont = True
                        break

                if cont:
                    continue

                self.links.append(y)
                self.types.append(types)
                piped_acs = list(nlp.pipe(acs))
                acs = [' '.join([word.text for word in ac]) for ac in piped_acs]
                vectors = [[word.vector for word in ac] for ac in piped_acs]
                self.vectors.append(vectors)
                self.documents.append(acs)  # list of list of sentences.

        print(len(self.documents), len(self.links), len(self.types), len(self.vectors))

        base = np.min(np.min(self.links))
        self.links = [np.array(link) - base for link in self.links]

        self.clf = CountVectorizer()
        self.clf.fit(sent for doc in self.documents for sent in doc)

        self.encoder_input, self.decoder_input = self._inputs()

    def _inputs(self):
        encoder_input = []
        decoder_input = []

        for document, vectors, links in zip(self.documents, self.vectors, self.links):
            indices = np.concatenate([[0], np.array(links)+1])
            representation = []
            for i, (ac, ac_vectors, ac_indices) in enumerate(zip(document, vectors, indices)):
                bow = np.squeeze(self.clf.transform([ac]).toarray())
                # min and max across token embeddings
                maximum = np.max(ac_vectors, axis=0)
                minimum = np.min(ac_vectors, axis=0)
                mean = np.mean(ac_vectors, axis=0)
                pos = [int(i == 1)]
                r = np.concatenate([pos, maximum, minimum, bow, mean], axis=0)
                representation.append(r)
            representation = np.array(representation)

            encoder_input.append(representation[1:])
            decoder_input.append(representation[indices[:-1]])

        return encoder_input, decoder_input


def write_on_disk():
    corpus = MTCorpus()
    np.save('encoder_input_ext', corpus.encoder_input)
    np.save('decoder_input_ext', corpus.decoder_input)
    np.save('Y_links_ext', corpus.links)
    np.save('Y_types_ext', corpus.types)


if __name__ == "__main__":
    write_on_disk()
