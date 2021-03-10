import random
import numpy as np
from args import get_args
import utils
import matplotlib.pyplot as plt

class DiscretizedDecisionSpace:
    def __init__(self, n, num_types):
        self.res = 1/n
        self.dSpace = np.arange(0, 1, self.res)
        self.n_types = num_types
        self.dist_means = random.choices(self.dSpace, k=self.n_types)

    def __iter__(self):
        return iter(self.dSpace)

    def disutility(self, a, b):
        return abs(a - b)

    def genBlissPoint(self):
        bp = np.random.normal(random.choice(self.dist_means), 0.05)
        return self.dSpace[min(np.searchsorted(self.dSpace, bp), len(self.dSpace)-1)]

    #def getClosest(self, potAlternatives, a):
    #    return min(potAlternatives, key = lambda x : dS.disutility(x, a))




class Agent:
    def __init__(self, decisionSpace, disutilityScale = 1):
        self.blissPoint = decisionSpace.genBlissPoint()
        self.scale = disutilityScale


class SequentialDeliberation:
    def __init__(self, numAgents, decisionSpace, bargainFn, args, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        self.decisionSpace = decisionSpace
        self.horizon = args.horizon + 1
        self.agents = [Agent(decisionSpace) for i in range(numAgents)]
        self.bargainFn = bargainFn
        self.a = [None] *  self.horizon#keeping a list since it might be interesting to see how "a^t" progresses
        self.a[0] = self.agents[0].blissPoint
        self.altToSC = {}
        self.optSC = np.inf
        self.optAlt = None

        for alternative in self.decisionSpace:          #caching this assuming these don't get too large
            sc = 0
            for agent in self.agents:
                sc += self.decisionSpace.disutility(agent.blissPoint, alternative)
            self.altToSC[alternative] = sc
            if sc < self.optSC:
                self.optSC = sc
                self.optAlt = alternative

        self.distortions = [None] * self.horizon
        self.distortions[0] = self.getDistortion(self.a[0])
        #self.altToDist = {alt:altToSC(alt)/self.optSC for alt in self.decisionSpace}


    def getDistortion(self, a):
        return self.altToSC[a] / self.optSC

    def getWorstDistortion(self):
        return max([self.getDistortion(a) for a in self.decisionSpace])

    def getBlissPoints(self):
        return [agent.blissPoint for agent in self.agents]

    def step(self, i):
        u, v = random.choices(self.agents, k=2) #sample w/ replacement
        self.a[i]  = self.bargainFn(u, v, self.a[i-1], self.decisionSpace)
        self.distortions[i] = self.getDistortion( self.a[i])

    def deliberate(self):
        for i in range(1,self.horizon):
            print("Distortion at step " + str(i-1) + ": " + str(self.distortions[i-1]))
            self.step(i)
        print(self.a)
        print("Final Distortion :" + str(self.distortions[i-1]))
        return self.a, self.distortions



def nashBargain(u, v, a, dS):
    disU = u.scale * dS.disutility(u.blissPoint, a)
    disV = v.scale * dS.disutility(v.blissPoint, a)
    potAlternatives = []
    maxNash = -np.inf
    for s in dS:
        if u.scale * dS.disutility(u.blissPoint, s) <= disU and v.scale * dS.disutility(v.blissPoint, s) <= disV: #individual rationality
            nash_product = (disU - u.scale * dS.disutility(u.blissPoint, s)) * (disV - v.scale * dS.disutility(v.blissPoint, s))
            if nash_product > maxNash:
                maxNash = nash_product
                potAlternatives = [s]
            elif nash_product == maxNash:
                potAlternatives.append(s)
    if potAlternatives == []:
        return a
    return min(potAlternatives, key = lambda x : dS.disutility(x, a))



def main():
    args = get_args()
    save_dir = utils.get_save_dir('save/', args.run_name)
    log_path = os.path.join(save_dir, f'{name}.txt')
    for i in range(args.num_simulations):
        sd = SequentialDeliberation(args.num_agents, DiscretizedDecisionSpace(args.num_alternatives, args.num_types), nashBargain, args, seed=i)
        a, d = sd.deliberate()
        if args.gen_plots:
            bp = sd.getBlissPoints()


if __name__ == '__main__':
    main()