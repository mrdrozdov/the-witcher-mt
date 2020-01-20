import sys

from Levenshtein import *

from tqdm import tqdm

import numpy as np


class Corpus(object):
    def __init__(self):
        super().__init__()
        self.lines = []
        self.vocab = set()


def read_corpus(path, offset=None, limit=None):
    eng, rus = Corpus(), Corpus()

    with open(path) as f:
        for i, line in enumerate(f):
            if i // 2 <= offset:
                continue
            if i % 2 == 0:
                eng.lines.append(line.strip())
            else:
                rus.lines.append(line.strip())
            if limit is not None:
                if len(rus.lines) == limit:
                    break

    return eng, rus


class Game(object):
    name = None

    def __init__(self, eng, rus):
        self.eng = eng
        self.rus = rus

    def setup(self):
        pass


class FillInTheBlankGame(Game):
    name = 'fill-in-the-blank'
    blank_size = 10
    n_negative = 7

    def setup(self):
        rus_new_lines = []
        eng_new_lines = []
        for i, x in enumerate(self.rus.lines):
            if len(x) == len('RUS:'):
                continue
            x = x[len('RUS: '):]
            rus_new_lines.append(x)
            eng_new_lines.append(self.eng.lines[i][len('ENG: '):])
        self.rus.lines = rus_new_lines
        self.eng.lines = eng_new_lines

    def play(self):
        i_positive = np.random.choice(range(len(self.rus.lines)))
        positive = self.rus.lines[i_positive]
        tokens = positive.split()
        i_blank = np.random.choice(range(len(tokens)))
        blank_token = '_' * self.blank_size

        # Blank the sentence.
        positive_token = tokens[i_blank]
        tokens[i_blank] = blank_token
        viz = ' '.join(tokens)

        # Choose negatives.
        candidate_lst = []
        while len(candidate_lst) < self.n_negative:
            negative = np.random.choice(self.rus.lines)
            tokens = negative.split()
            candidate = np.random.choice(tokens)
            if candidate == positive_token:
                continue
            candidate_lst.append(candidate)

        # Shuffle.
        lst = [positive_token] + candidate_lst
        np.random.shuffle(lst)
        i_correct = lst.index(positive_token)

        # Play.
        msg = 'FILL IN THE BLANK'
        print(msg)
        print('-' * len(msg))
        print(viz)
        print('')
        for i, tok in enumerate(lst):
            print('{}. {}'.format(i, tok))
        print('')

        while True:
            choice = input("Enter your choice: ")

            try:
                choice = int(choice)
            except:
                print('Choice was not an integer from [0 - {}]. Try again.'.format(self.n_negative))
                continue

            if choice < 0 or choice > self.n_negative:
                print('Choice was not an integer from [0 - {}]. Try again.'.format(self.n_negative))
                continue

            break

        print('Original (rus): {}'.format(positive))
        print('Original (eng): {}'.format(self.eng.lines[i_positive]))
        print('')

        if choice == i_correct:
            print('That was correct!')
        else:
            print('{} is incorrect. Should have chosen: {}'.format(lst[choice], lst[i_correct]))
        print('')



class ChooseSentenceGame(Game):
    name = 'choose-sentence'
    n_factor = 5
    n_negative_hard = 6
    n_negative_easy = 1

    def setup(self):

        rus_new_lines = []
        eng_new_lines = []
        for i, x in enumerate(self.rus.lines):
            if len(x) == len('RUS:'):
                continue
            x = x[len('RUS: '):]
            rus_new_lines.append(x.strip())
            eng_new_lines.append(self.eng.lines[i][len('ENG: '):].strip())
        self.rus.lines = rus_new_lines
        self.eng.lines = eng_new_lines

        # Compute similarity.
        n = len(self.eng.lines)
        sim = np.zeros((n, n), dtype=np.float32)
        for i in tqdm(range(n)):
            for j in range(n):
                if i == j:
                    continue
                xi = self.eng.lines[i]
                xj = self.eng.lines[j]
                yi = self.rus.lines[i]
                yj = self.rus.lines[j]
                if xi == xj or yi == yj:
                    continue
                sim[i, j] = jaro_winkler(xi, xj)
        self.sim = sim

    def play(self):
        i_positive = np.random.choice(range(len(self.rus.lines)))
        positive_rus = self.rus.lines[i_positive]
        positive_eng = self.eng.lines[i_positive]
        chosen = set()
        chosen.add(i_positive)

        # Hard.
        local_chosen = set()
        n_keep = self.n_negative_hard * self.n_factor
        local_sim_arg = np.argsort(self.sim[i_positive])[-n_keep:]
        local_sim_val = self.sim[i_positive][local_sim_arg]
        local_p = np.exp(local_sim_val * n_keep) / np.exp(local_sim_val * n_keep).sum()
        while len(local_chosen) < self.n_negative_hard:
            i_negative = np.random.choice(local_sim_arg, p=local_p)
            if i_negative in local_chosen or i_negative in chosen:
                continue
            local_chosen.add(i_negative)
        chosen = chosen.union(local_chosen)

        # Easy.
        local_chosen = set()
        while len(local_chosen) < self.n_negative_easy:
            i_negative = np.random.choice(range(len(self.eng.lines)))
            if i_negative in local_chosen or i_negative in chosen:
                continue
            local_chosen.add(i_negative)
        chosen = chosen.union(local_chosen)

        # Shuffle.
        lst = list(chosen)
        np.random.shuffle(lst)
        i_correct = lst.index(i_positive)

        # Play.
        msg = 'CHOOSE SENTENCE'
        print(msg)
        print('-' * len(msg))
        print(positive_rus)
        print('')
        for i, i_sent in enumerate(lst):
            print('{}. {}'.format(i, self.eng.lines[i_sent]))
        print('')

        n_negative = self.n_negative_hard + self.n_negative_easy
        while True:
            choice = input("Enter your choice: ")

            try:
                choice = int(choice)
            except:
                print('Choice was not an integer from [0 - {}]. Try again.'.format(n_negative))
                continue

            if choice < 0 or choice > n_negative:
                print('Choice was not an integer from [0 - {}]. Try again.'.format(n_negative))
                continue

            break

        print('Original (rus): {}'.format(positive_rus))
        print('Original (eng): {}'.format(positive_eng))
        print('')

        if choice == i_correct:
            print('That was correct!')
        else:
            print('{} is incorrect. Should have chosen: {}'.format(choice, i_correct))
        print('')


class ChooseWordGame(Game):
    name = 'choose-word'

    def setup(self):
        pass

    def play(self):
        pass



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', default='aligned_eng_rus.txt', type=str)
    parser.add_argument('--limit', default=100, type=int)
    parser.add_argument('--offset', default=100, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--game', default='fill-in-the-blank')
    options = parser.parse_args()

    if options.seed is not None:
        np.random.seed(options.seed)

    eng, rus = read_corpus(options.inp, options.offset, options.limit)

    for ta, tb in zip(eng.lines, rus.lines):
        print(ta)
        print(tb)
        print('')

        wa = ta[len('ENG: '):].split()
        wb = tb[len('RUS: '):].split()

        for w in wa:
            eng.vocab.add(w)
        for w in wb:
            rus.vocab.add(w)

    print(eng.vocab)
    print(rus.vocab)

    print('vocab eng rus', len(eng.vocab), len(rus.vocab))

    games = [ChooseWordGame, ChooseSentenceGame, FillInTheBlankGame]
    for g in games:
        if g.name == options.game:
            game = g(eng, rus)

    print('\n' * 10)
    print('Beginning Game...')
    print('')

    game.setup()
    while  True:
        game.play()




if __name__ == '__main__':
    main()
