import numpy as np

from tqdm import tqdm


class Corpus(object):
    def __init__(self):
        super().__init__()
        self.lines = []
        self.words = []
        self.vocab = set()

    def read_file(self, path, limit=None):
        with open(path) as f:
            for i, line in tqdm(enumerate(f)):
                words = line.split()
                for w in words:
                    self.words.append(w)
                    self.vocab.add(w)
                self.lines.append(words)
                if limit is not None and len(self.lines) == limit:
                    break
        return self




# def align_lines(text_a, text_b, lookback=10):
#     """
#     - Map every line from (a) to a line from (b), or to no line.
#     - Cost of mapping: difference of words (no line has 0 words).
#     - Mapping is non-overlapping.
#     """

#     for i in range(_):
#         table = np.zeros(lookback, lookback)
#         for k in range(lookback):
#             for j in range(0, k + 1):
#                 assert j <= k
#                 table[j, k] = max(table[j-1, k], cost(i, j, k))
#                     ... # best value when position (i-k) is assigned (i-j). j <= k


#     return None


class AlignLines(object):
    def __init__(self):
        super(AlignLines, self).__init__()

    def custom_edit_distance(self, text_a, text_b, history=[]):
        """
        -2 : none
        -1 : all
        """
        n_a, n_b = len(text_a), len(text_b)

        if n_a == 0:
            val = sum([len(x) for x in text_b])
            history = history + [(-2, -1)]
            return val, history
        elif n_b == 0:
            val = sum([len(x) for x in text_a])
            history = history + [(-1, -2)]
            return val, history
        else:
            diff = np.abs(len(text_a[-1]) - len(text_b[-1]))

            if diff == 0:
                val, history = self.custom_edit_distance(text_a[:-1], text_b[:-1], history)
                history = history + [(n_a-1, n_b-1)]
                return val, history
            else:
                vals, histories = zip(*[
                    self.custom_edit_distance(text_a[:-1], text_b[:-1], history),
                    self.custom_edit_distance(text_a, text_b[:-1], history),
                    # self.custom_edit_distance(text_a[:-1], text_b, history),
                ])
                vals = list(vals)
                vals[0] = vals[0] + diff + 1 # add one so that dropping is preferred.
                vals[1] = vals[1] + len(text_b[-1])
                # vals[2] = vals[2] + len(text_a[-1])
                p_hist = [None] * 3
                p_hist[0] = (n_a-1, n_b-1)
                p_hist[1] = (n_a-1, -2)
                # p_hist[1] = (-2, n_b-1)
                choice = np.argmin(vals)
                val = vals[choice]
                history = histories[choice]
                history = history + [p_hist[choice]]
                return val, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--eng', default='./data/english.txt', type=str)
    parser.add_argument('--rus', default='./data/russian.txt', type=str)
    parser.add_argument('--limit', default=13000, type=int)
    parser.add_argument('--window', default=20, type=int)
    options = parser.parse_args()

    eng = Corpus().read_file(options.eng, limit=options.limit)
    rus = Corpus().read_file(options.rus, limit=options.limit)

    print('eng : words = {}, lines = {}, vocab = {}'.format(len(eng.words), len(eng.lines), len(eng.vocab)))
    print('rus : words = {}, lines = {}, vocab = {}'.format(len(rus.words), len(rus.lines), len(rus.vocab)))

    prev_a, prev_b = 0, 0

    for iteration in range(0, options.limit - options.window, options.window):

        print('ROUND {} :: a = {}, b = {}'.format(iteration, prev_a, prev_b))

        start_a, start_b = prev_a + options.window, prev_b + options.window
        end_a = start_a + options.window
        end_b = start_b + options.window
        text_a, text_b = eng.lines[start_a:end_a], rus.lines[start_b:end_b]
        cost, alignment = AlignLines().custom_edit_distance(text_a, text_b)
        last_a, last_b = None, None

        # debug
        for j, (sa, sb) in enumerate(alignment):
            if sa == -1:
                assert j == 0, alignment
                assert sb == -2, alignment
            elif sb == -1:
                assert j == 0, alignment
                assert sa == -2, alignment

        # fixup alignment
        sa_0, sb_0 = alignment[0]

        if sa_0 == -1 or sb_0 == -1:
            alignment = alignment[1:]
        if sa_0 == -1:
            if len(alignment) > 0:
                sa_stop = alignment[0][0]
            else:
                sa_stop = len(text_a)
            assert sa_stop != -2
            new_alignment = [(j, -2) for j in range(0, sa_stop)]
        elif sb_0 == -1:
            if len(alignment) > 0:
                sb_stop = alignment[0][1]
            else:
                sb_stop = len(text_b)
            assert sb_stop != -2
            new_alignment = [(-2, j) for j in range(0, sb_stop)]

        assert len(text_a) == len(text_b)
        for j in range(len(text_a)):
            print('[{}] {} :: {}'.format('eng', j, text_a[j]))
        for j in range(len(text_b)):
            print('[{}] {} :: {}'.format('rus', j, text_b[j]))
        print('')

        for j, (sa, sb) in enumerate(alignment):
            assert sa != -1 and sb != -1
            ta = text_a[sa] if sa != -2 else '-NONE-'
            tb = text_b[sb] if sb != -2 else '-NONE-'

            if sa != -2:
                last_a = sa
            if sb != -2:
                last_b = sb

            print('[{}] {} :: {} :: {}'.format('eng', j, sa, ta))
            print('[{}] {} :: {} :: {}'.format('rus', j, sb, tb))
            print('')

        assert last_a == len(text_a) - 1

        prev_a = prev_a + last_a + 1
        if last_b is not None:
            prev_b = prev_b + last_b + 1
        else:
            raise Exception

        # last_s1, last_s2 = None, None

        # for j, (s1, s2) in enumerate(alignment):

        #     if s1 == -1:
        #         assert s2 == -2

        #         if j == len(alignment) - 1:
        #             last_print = len(text_a)
        #         else:


        #             first_print = last_s1 + 1 if last_s1 is not None else 0
        #             # print until end
        #             for k in range(first_print, len(text_a)):
        #                 print('[{}] {}: {}'.format('eng', k, text_a[k]))
        #                 print('[{}] {}: {}'.format('rus', k, 'none'))

        #     print(s1, s2)
