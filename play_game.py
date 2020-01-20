import sys

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

    def setup(self):
        pass

    def play(self):
        pass


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
    parser.add_argument('--game', default='fill-in-the-blank')
    options = parser.parse_args()

    np.random.seed(12)

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
