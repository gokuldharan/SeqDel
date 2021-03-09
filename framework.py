import random
import numpy as np


class DiscretizedDecisionSpace:
    def __init__(self, resolution, start=0, end=1):
        self.res = resolution
        self.dSpace = np.arange(start, end, self.res)

    def __iter__(self):
        return iter(self.dSpace)


class Agent:
    def __init__(self, decisionSpace, utilityFn = random.random, disutilityScale = 1):
        self.decisionSpace = decisionSpace
        self.utilityFn = utilityFn
        self.utility = {s:utilityFn() for s in self.decisionSpace}
        self.blissPoint = max(self.utility.keys(), key = lambda s: self.utility[s])
        self.scale = disutilityScale

    def disutility(self, a):
        return self.scale * (self.utility[self.blissPoint] - self.utility[a]) # potentially needs to change for other decision spaces

    #Potentially implement utility shifting after bargaining?


class SequentialDeliberation:
    def __init__(self, numAgents, decisionSpace, bargainFn, seed=0):
        random.seed(seed)
        self.decisionSpace = decisionSpace
        self.agents = [Agent(decisionSpace, utilityFn = random.random) for i in range(numAgents)]
        self.bargainFn = bargainFn
        self.a = [self.agents[0].blissPoint] #keeping a list since it might be interesting to see how "a^t" progresses
        self.altToSC = {}
        self.optSC = np.inf
        for alternative in self.decisionSpace:          #caching this assuming these don't get too large
            sc = 0
            for agent in self.agents:
                sc += agent.disutility(alternative)
            self.altToSC[alternative] = sc
            self.optSC = min(self.optSC, sc)


    def getDistortion(self, a):
        return self.altToSC[a] / self.optSC

    def step(self):
        u, v = random.choices(self.agents, k=2) #sample w/ replacement
        o = self.bargainFn(u, v, self.a[-1])
        self.a.append(o)

    def deliberate(self, horizon):
        for i in range(horizon):
            print("Distortion at step " + str(i) + ": " + str(self.getDistortion(self.a[-1])))
            self.step()
        print("Final Distortion :" + str(self.getDistortion(self.a[-1])))
        return self.a



def nashBargain(u, v, a):
    disU = u.disutility(a)
    disV = v.disutility(a)
    potAlternatives = []
    maxNash = -np.inf
    for s in u.decisionSpace:
        if u.disutility(s) <= disU and v.disutility(s) <= disV: #individual rationality
            nash_product = (disU - u.disutility(s)) * (disV - v.disutility(s))
            if nash_product > maxNash:
                maxNash = nash_product
                potAlternatives = [s]
            elif nash_product == maxNash:
                potAlternatives.append(s)
    if potAlternatives == []:
        return a
    return min(potAlternatives, key = lambda x: abs(x - a))          #change this tie-breaker for other decision spaces



sd = SequentialDeliberation(1000, DiscretizedDecisionSpace(0.001), nashBargain)
a = sd.deliberate(10)
print(a)