BINARY_QUERY = True           #True = do not consider word frequency in query embedding
ADD_SYNONYMS_TO_QUERY = False #add synonyms to auery embedding
NORM_QUERY = False            #normalize query embedding to the Frobenius norm(2-norm, Euclidean norm)
INCLUDE_WITHOUT_PHASE = True  #include docs with blank phase.
BINARY_DOCS = True            #index setting. True = do not consider word frequency in doc's embeddings
ADD_SYNONIMS_TO_DOCS = True   #add synonyms to doc's embeddings
COLUMNS = ['official_title', 'brief_title', 'drug'] #columns for building embeddings
NORM_SIM_TO_DOC_LEN = False   #normalize similarity to the Frobenius norm(2-norm, Euclidean norm)
TOPMINETH = 45                #Topmine treshhold, 4 used to mine ngrams from docs.
N_TOKENS = 5842               #Magic Topmine parameter
IMPORT_PATH = dict(                                     #path relative to the search.py
    VOCAB_PATH='./data/vocab',                          #Dict object with vocab: {frozenset(ngram):number}
    COUNTER_PATH='./data/counter',                      #Counter object with counts of all possible ngrams from docs
    CORPUS_PATH='./data/corpus.csv',                    #Dataframe with docs
    SYN_MATRIX_PATH='./data/syn_mat.npz'                #MATRIX FROM HELL! Precomputed words similarity!
)
STOP_WORDS = {'around', 'namely', 'against', 'front', 'only',
              'third', 'very', 'however', 'again', 'top', 'until', 'hundred',
              'between', 'towards', 'meanwhile', 'always', 'mostly', 'should',
              'regarding', 'cannot', 'such', 'where', 'a', 'own', 'put',
              'wherein', 'forty', 'whereupon', 'yourself', 'besides', 'alone',
              'can', 'there', 'herself', 'much', 'never', 'whether', 'with',
              'therein', 'his', 'into', 'has', 'rather', 'nowhere', 'although',
              'itself', 'those', 'throughout', 'its', 'in', 'less', 'then',
              'whom', 'is', 'therefore', 'this', 'yourselves', 'both', 'almost',
              'four', 'inc', 'anyhow', 'hereby', 'ten', 'ca', 'ourselves', 'the',
              'these', 'who', 'becoming', 'thereafter', 'could', 'been', 'that',
              'us', 'take', 'because', 'thence', 'even', 'together', 'whither',
              'etc', 'several', 'your', 'give', 'side', 'we', 'back', 'become',
              'being', 'empty', 'among', 'afterwards', 'be', 'across', 'already',
              'he', 'next', 'whoever', 'unless', 'last', 'used', 'full', 'what',
              'above', 'their', 'latterly', 'also', 'did', 'noone', 'else',
              'elsewhere', 'name', 'former', 'by', 'fifteen', 'beforehand',
              'serious', 'have', 'seemed', 'everywhere', 'five', 'six', 'each',
              'as', 'so', 'anything', 'someone', 'see', 'one', 'himself',
              'about', 'must', 'ever', 'during', 'nobody', 'whose', 'on', 'from',
              'two', 'was', 'onto', 'yet', 'since', 'am', 'many', 'whereafter',
              'using', 'though', 'thus', 'whole', 'all', 'our', 're', 'per',
              'once', 'it', 'might', 'over', 'became', 'seems', 'to', 'which',
              'formerly', 'just', 'ours', 'below', 'amongst', 'would', 'due',
              'now', 'behind', 'really', 'further', 'becomes', 'make', 'enough',
              'for', 'please', 'amount', 'fifty', 'if', 'along', 'myself',
              'and', 'were', 'after', 'move', 'still', 'well', 'him', 'twelve',
              'neither', 'thereby', 'wherever', 'they', 'anyone', 'me', 'few',
              'more', 'sixty', 'latter', 'through', 'keep', 'seeming',
              'anywhere', 'of', 'when', 'you', 'doing', 'bottom', 'either',
              'does', 'off', 'somehow', 'others', 'at', 'mine', 'beside',
              'down', 'indeed', 'eleven', 'everyone', 'whenever', 'often',
              'nevertheless', 'an', 'any', 'while', 'thru', 'go', 'will',
              'except', 'some', 'too', 'them', 'her', 'three', 'most',
              'nothing', 'or', 'show', 'do', 'twenty', 'nor', 'had', 'various',
              'least', 'within', 'she', 'why', 'say', 'without', 'somewhere',
              'yours', 'anyway', 'i', 'toward', 'something', 'before', 'beyond',
              'themselves', 'via', 'eight', 'whereby', 'than', 'thereupon',
              'my', 'sometime', 'get', 'quite', 'made', 'no', 'whereas', 'may',
              'part', 'herein', 'sometimes', 'out', 'another', 'whatever',
              'here', 'moreover', 'upon', 'but', 'call', 'same', 'perhaps',
              'seem', 'under', 'are', 'done', 'how', 'not', 'otherwise', 'hers',
              'none', 'hereupon', 'other', 'hence', 'up', 'whence', 'first',
              'every', 'nine', 'hereafter', 'everything'}
