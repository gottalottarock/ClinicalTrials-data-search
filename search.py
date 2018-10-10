import argparse
import pickle
import pandas as pd
from ct_search.topmine import TopmineTokenizer
from ct_search import config
from ct_search.searcher import Searcher, tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--drugs', action='store', nargs=1,
                        required=True, help='Drugs query to search', type=str)
    parser.add_argument('-p', '--phases', action='store', nargs='+',
                        default=[1, 2, 3], type=int,
                        help='Possible phases', choices=[1, 2, 3])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-s', '--similarity', action='store_true')
    parsed = parser.parse_args()
    return parsed.drugs[0], parsed.phases, parsed.debug, parsed.similarity


def import_data(paths):
    df = pd.read_csv(paths['CORPUS_PATH'], sep='\t')
    with open(paths['VOCAB_PATH'], 'rb') as f:
        vocab = pickle.load(f)
    with open(paths['COUNTER_PATH'], 'rb') as f:
        counter = pickle.load(f)
    return df, vocab, counter


def main():
    print_properties = {}
    (query, phases, print_properties['debug'],
     print_properties['similarity']) = parse_arguments()
    df, vocab, counter = import_data(config.IMPORT_PATH)

    tok = TopmineTokenizer(counter=counter, n_tokens=config.N_TOKENS,
                           threshold=config.TOPMINETH, vocab=vocab)

    searcher = Searcher(
        vocab=vocab, tokenizer=tokenizer(tok), df=df,
        path_to_syn_matrix=config.IMPORT_PATH['SYN_MATRIX_PATH'])

    searcher.search(query=query,
                    binary_query=config.BINARY_QUERY,
                    add_synonyms_to_query=config.ADD_SYNONYMS_TO_QUERY,
                    norm_query=config.NORM_QUERY,
                    phases=phases,
                    include_without_phase=config.INCLUDE_WITHOUT_PHASE,
                    binary_docs=config.BINARY_DOCS,
                    add_synonims_to_docs=config.ADD_SYNONIMS_TO_DOCS,
                    columns=config.COLUMNS,
                    norm_sim_to_doc_len=config.NORM_SIM_TO_DOC_LEN,
                    print_properties=print_properties)


if __name__ == '__main__':
    main()
