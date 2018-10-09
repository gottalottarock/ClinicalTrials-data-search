import math
from heapq import heappush, heappop, heapify


class TopmineTokenizer():
    '''
    Tokenize texts, mining the most frequent n-grams.
    Ngrams are not ordered(frozenset of words)
     This is an adapted implementation of the tokenization algorithm
    detailed in: El-Kishky, Ahmed, et al.
    "Scalable topical phrase mining from text corpora."
    Proceedings of the VLDB Endowment 8.3 (2014): 305-316.APA

    Code adapted from: https://github.com/latorrefabian/topmine
    '''

    def __init__(self, counter, n_tokens,
                 threshold=1, vocab=None):
        self.counter = counter
        self.threshold = threshold
        self.n_tokens = n_tokens
        self.vocab = vocab

    def transform_document(self, sentences):
        '''sentences = list of lists (pretokenized,removed stop-words)
        '''
        return [self.transform_sentence(sentence) for sentence in sentences]

    def transform_sentence(self, sentence):
        '''sentence -- a sentence, as a list of words
           Return it as a sequence of significant phrases.
        '''
        phrases = [frozenset({x, }) for x in sentence]
        phrase_start = [x for x in range(len(phrases))]
        phrase_end = [x for x in range(len(phrases))]

        costs = [(self.cost(phrases[i], phrases[i + 1]), i, i + 1, 2)
                 for i in range(len(phrases) - 1)]
        heapify(costs)

        while True and len(costs) > 0:
            cost, i_a, i_b, length = heappop(costs)

            if cost > -self.threshold:
                break
            if phrase_start[i_a] != i_a:
                continue
            if phrase_start[i_b] != i_b:
                continue
            if length != len(phrases[i_a] | phrases[i_b]):
                continue

            phrase_start[i_b] = i_a
            phrase_end[i_a] = phrase_end[i_b]
            merged_phrase = phrases[i_a] | phrases[i_b]
            phrases[i_a] = merged_phrase
            phrases[i_b] = frozenset({})

            if i_a > 0:
                prev_phrase_start = phrase_start[i_a - 1]
                prev_phrase = phrases[prev_phrase_start]
                heappush(costs, (
                    self.cost(prev_phrase, merged_phrase),
                    prev_phrase_start, i_a,
                    len(prev_phrase | merged_phrase)))

            if phrase_end[i_b] < len(phrases) - 1:
                next_phrase_start = phrase_end[i_b] + 1
                next_phrase = phrases[next_phrase_start]
                heappush(costs, (
                    self.cost(merged_phrase, next_phrase),
                    i_a, next_phrase_start,
                    len(merged_phrase | next_phrase)))

        phrases_finish = [x for x in phrases if (
            x is not None) and (not self.vocab or (x in self.vocab))]
        return phrases_finish

    def cost(self, a, b):
        '''Calculates the cost of merging two phrases. Cost is
        defined as the negative of significance. This way we can
        use the python min-heap implementation
        '''
        if (self.vocab) and ((a | b) not in self.vocab):
            return math.inf

        a_c = self.counter[a]
        b_c = self.counter[b]
        ab = self.counter[a | b]
        if ab == 1:
            return math.inf
        if (ab == 2) and ((a_c > 2) or (b_c > 2)):
            return math.inf
        eab = a_c * b_c / self.n_tokens
        if eab > 0:
            cost_ab = (-(ab - (eab)) / math.sqrt(eab))  # like z-score
        else:
            cost_ab = math.inf

        return cost_ab
