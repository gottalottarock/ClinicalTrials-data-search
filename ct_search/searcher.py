import re
import numpy as np
from scipy import sparse
from .corpus import Corpus
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import find
from .config import STOP_WORDS


def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    return re.findall(r'[\w]+', string.lower())


def sent_to_list(sent):
    return [t for t in tokenize(sent) if t not in STOP_WORDS] if sent else sent


def sent_to_ngrams(se, tok):
    return tok.transform_sentence(sent_to_list(se))


def split_sents_to_ngrams(sents, tok, spl_pat='[()]'):
    return [t for st in re.split(pattern=spl_pat, string=sents)
            for t in sent_to_ngrams(st, tok) if t]


def tokenizer(tok):
    def tokzr(s):
        return split_sents_to_ngrams(s, tok, '[\[）\])(（|]')
    return tokzr


class Searcher():
    def __init__(self, vocab, tokenizer, df, path_to_syn_matrix):
        self.syn_mat = sparse.load_npz(path_to_syn_matrix)
        self.corpus = Corpus(df)
        self.vocab = vocab
        self.tokenizer = tokenizer

    def add_synonyms(self, embeddings):
        return embeddings.dot(self.syn_mat).maximum(embeddings).minimum(1)

    def get_query_embedding(self, query, binary_query, add_synonyms_to_query,
                            norm_query):
        cvr = CountVectorizer(vocabulary=self.vocab, tokenizer=self.tokenizer,
                              lowercase=False, binary=binary_query)
        embedding = cvr.transform([query])
        if add_synonyms_to_query:
            embedding = self.add_synonyms(embedding)
        if norm_query:
            embedding = embedding / sparse.linalg.norm(embedding)
        return embedding

    def get_docs_embedding(
        self, phases, include_without_phase, binary_docs,
        add_synonims_to_docs, columns
    ):
        docs_index, docs_embedding = self.corpus.get_docs_embedding(
            self.vocab, self.tokenizer,
            phases, include_without_phase,
            binary_docs, columns
        )
        if add_synonims_to_docs:
            docs_embedding = self.add_synonyms(docs_embedding)
        return docs_index, docs_embedding

    def get_similarity(self, docs_embedding, query_embedding,
                       norm_sim_to_doc_len):
        vsim = query_embedding.dot(
            docs_embedding.transpose()).toarray()
        if norm_sim_to_doc_len:
            docs_norm = sparse.linalg.norm(docs_embedding, axis=1)
            docs_norm[np.where(~docs_norm.astype(bool))] = np.inf
            vsim = vsim / docs_norm
        return vsim[0]

    def filter_similarity(self, docs_index, similarity, sorted=True,
                          threshold=0.0):
        iloc = np.where((similarity > threshold))[0]
        final_index = docs_index[iloc]
        final_similarity = similarity[iloc]
        if sorted:
            ind_sort = np.argsort(final_similarity)[::-1]
            iloc = iloc[ind_sort]
            final_index = final_index[ind_sort]
            final_similarity = final_similarity[ind_sort]
        return final_index, final_similarity, iloc

    def print_results_debug(self, final_docs_index, ind_sort, docs_similarity,
                            docs_embedding, query_embedding):
        inverse_vocab = {v: k for k, v in self.vocab.items()}
        # wi = 20

        def words_simularuty(i):
            found = find(docs_embedding[i].multiply(query_embedding))
            indexes = found[1][np.argsort(found[2])]
            iterw = ((("%.2f" % query_embedding[0, j]),
                      ("%.2f" % docs_embedding[i, j]),
                      ' '.join(inverse_vocab[j])) for j in indexes)
            return ''.join(['\n',
                            '\n'.join(
                                map(lambda w: ' '*20 + ' '.join(w), iterw))
                            ])
        ws = (words_simularuty(i) for i in ind_sort)
        to_print = map(' '.join,
                       zip(self.corpus.get_id_by_index(final_docs_index),
                           np.char.mod('%.2f', docs_similarity), ws))
        print('\n'.join(to_print))
        print(len(ind_sort))

    def print_results(self, final_docs_index, ind_sort, docs_similarity,
                      docs_embedding, query_embedding, debug=False,
                      similarity=False):
        if debug:
            self.print_results_debug(final_docs_index, ind_sort, docs_similarity,
                                     docs_embedding, query_embedding)
        elif similarity:
            to_print = map(' '.join,
                           zip(self.corpus.get_id_by_index(final_docs_index),
                               np.char.mod('%.2f', docs_similarity)))
            print('\n'.join(to_print))
        else:
            print('\n'.join(self.corpus.get_id_by_index(final_docs_index)))

    def search(
        self, query, binary_query, add_synonyms_to_query, norm_query, phases,
        include_without_phase, binary_docs, add_synonims_to_docs, columns,
        norm_sim_to_doc_len, print_properties
    ):

        query_embedding = self.get_query_embedding(query, binary_query,
                                                   add_synonyms_to_query,
                                                   norm_query)
        docs_index, docs_embedding = self.get_docs_embedding(
            phases, include_without_phase, binary_docs,
            add_synonims_to_docs, columns
        )
        similarity = self.get_similarity(docs_embedding, query_embedding,
                                         norm_sim_to_doc_len)
        final_docs_index, final_similarity, iloc = \
            self.filter_similarity(docs_index, similarity)
        self.print_results(final_docs_index, iloc, final_similarity,
                           docs_embedding, query_embedding, **print_properties)
