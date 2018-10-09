import re
import numpy as np
from scipy import sparse
from .corpus import Corpus
from sklearn.feature_extraction.text import CountVectorizer
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
        iloc = np.where((similarity > threshold))
        final_index = docs_index[iloc]
        final_similarity = similarity[iloc]
        if sorted:
            ind_sort = np.argsort(final_similarity)[::-1]
            final_index = final_index[ind_sort]
            final_similarity = final_similarity[ind_sort]
        return final_index, final_similarity

    def print_results(self, docs_index, docs_similarity, debug):
        if debug:
            self.print_results_debug(docs_index, docs_similarity)
        else:
            print('\n'.join(self.corpus.get_id_by_index(docs_index)))
            print(len(docs_index))

    def search(
        self, query, binary_query, add_synonyms_to_query, norm_query, phases,
        include_without_phase, binary_docs, add_synonims_to_docs, columns,
        norm_sim_to_doc_len, debug
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
        final_docs_index, final_similarity = self.filter_similarity(docs_index,
                                                                    similarity)
        self.print_results(final_docs_index, final_similarity, debug)
