import random


class DecisionSpace:
    def __

class Agent:
    def __init__(decisionSpace):
        self.decisionSpace = decisionSpace

    def getBlissPoint(self):

    def disutility(self, a):






class SequentialDeliberation:
    def __init__(numAgents, decisionSpace, bargainFn, seed=0):
        self.agents = [Agent() for i in range(numAgents)]
        self.decisionSpace = decisionSpace
        self.bargainFn = bargainFn
        self.a = [agents[0].getBlissPoint()] #keeping a list since it might be itneresting to see how "a^t" progresses
        random.seed(seed)

    def step(self):
        u, v = random.choices(self.agents, 2) #sample w/ replacement
        o = self.bargainFn(u, v, self.a[-1])
        self.a.append(o)

    def deliberate(horizon):
        for i in range(horizon):
            self.step()
        return self.a


def nashBargain(u, v, a):
