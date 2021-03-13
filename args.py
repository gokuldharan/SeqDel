import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=300)
    parser.add_argument('--seed', type=int, default=3456)
    parser.add_argument('--num-alternatives', type=int, default=50)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--num-simulations', type=int, default=1000)
    #parser.add_argument('--gen-plots', action='store_true', default=True)
    parser.add_argument('--plot-interval', type=int, default=100)
    parser.add_argument('--gen-delib', action='store_true')
    parser.add_argument('--num-types', type=int, default=3)
    parser.add_argument('--bargain-dev', type=float, default=0.1)
    parser.add_argument('--run-name', type=str, default='exp_2')
    #parser.add_argument('--utility', type=str, default='gaussian')
    args = parser.parse_args()
    return args
