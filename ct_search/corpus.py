import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


class Corpus():
    def __init__(self, df):
        self.corpus = df

    def load_corpus(self, path_to_df):
        self.corpus = pd.read_csv(path_to_df).reset_index(drop=True)
        # self.syn_mat = csr_matrix.read_pickle(path_to_syn_mat)

    def get_docs_of_phases(self, phases, include_without_phase=False):
        ''' corpus.phases:
                None of string 'phase phase'('1 2','3','3 2 1')
            param phases:
                container of required phases
        '''
        regx = '[{}]'.format(''.join(str(p) for p in phases))
        if include_without_phase:
            return self.corpus[self.corpus.phases.isna() |
                               self.corpus.phases.str.contains(regx)]
        else:
            return self.corpus[self.corpus.phases.notna() &
                               self.corpus.phases.str.contains(regx)]

    def prepare_docs(self, phases, include_without_phase=False, columns=None):
        if not columns:
            columns = ['official_title', 'brief_title', 'drug']
        df_filter = self.get_docs_of_phases(phases, include_without_phase)
        ser_data = df_filter[columns[0]]
        if len(columns) > 1:
            ser_data = ser_data.str.cat(
                df_filter[columns[1:]], sep=' | ', na_rep='')
        return ser_data.index, ser_data.tolist()

    def get_docs_embedding(
        self, vocab, tokenizer,
        phases, include_without_phase=False, binary=True,
        columns=None
    ):
        index, data = self.prepare_docs(phases, include_without_phase, columns)
        cvr = CountVectorizer(vocabulary=vocab, tokenizer=tokenizer,
                              lowercase=False, binary=binary)
        embedding = cvr.transform(data)
        return index, embedding

    def get_id_by_index(self, index):
        return self.corpus.loc[index, 'nct_id']
