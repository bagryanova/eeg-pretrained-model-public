from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(join(PROJECT_ROOT, 'src'))

from eeg.utils import seed_everything


CNT = 0


def foo():
    global CNT
    from names_generator import generate_name
    print(generate_name() + '_' + str(CNT))
    seed_everything(228)
    CNT += 1


if __name__ == '__main__':
    foo()
    foo()
    foo()
