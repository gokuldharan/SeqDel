import random
import numpy as np
from args import get_args
import utils
import matplotlib.pyplot as plt
import os

class DiscretizedDecisionSpace:
    def __init__(self, n, num_types):
        self.res = 1/n
        self.dSpace = np.arange(0, 1, self.res)
        self.n_types = num_types
        self.dist_means = random.choices(self.dSpace, k=self.n_types)
        self.randomnessBound = 0.1

    def __iter__(self):
        return iter(self.dSpace)

    def disutility(self, a, b):
        return abs(a - b)

    def getClosest(self, a):
        return self.dSpace[min(np.searchsorted(self.dSpace, a), len(self.dSpace)-1)]

    def genBlissPoint(self):
        bp = np.random.normal(random.choice(self.dist_means), 0.05)
        return self.getClosest(bp)

    def shiftAlt(self, src, dst, weight):
        dstVec = (dst - src)
        if dstVec == 0:
            return src
        move = (1-self.randomnessBound*np.random.rand()) * weight * dstVec / abs(dstVec)
        if abs(move) > abs(dstVec):
            return dst##GEOMETRIC?

        return self.getClosest(src + move)

    def shiftAltTowardsTwo(self, src, dst1, dst2, weight):
        dstVec = ((dst1 - src) + (dst2 - src)) / 2
        if dstVec == 0:
            return src
        move = (1-self.randomnessBound*np.random.rand()) * weight * dstVec/abs(dstVec)
        if abs(move) > abs(dstVec):
            return self.getClosest(src + dstVec)

        return self.getClosest(src + move)


    #def getClosest(self, potAlternatives, a):
    #    return min(potAlternatives, key = lambda x : dS.disutility(x, a))


class Agent:
    def __init__(self, decisionSpace, bargainingPowerDev = 0, scale=1):
        self.dS = decisionSpace
        self.blissPoint = decisionSpace.genBlissPoint()
        self.bargainPower = np.random.normal(1, bargainingPowerDev)
        self.scale = 1







class SequentialDeliberation:
    def __init__(self, numAgents, decisionSpace, bargainFn, args, seed=0, bargain_dev=0):
        np.random.seed(seed)
        random.seed(seed)
        self.decisionSpace = decisionSpace
        self.horizon = args.horizon + 1
        self.agents = [Agent(decisionSpace, bargain_dev) for i in range(numAgents)]
        self.bargainFn = bargainFn
        self.a = [None] *  self.horizon#keeping a list since it might be interesting to see how "a^t" progresses
        self.a[0] = self.agents[0].blissPoint
        self.altToSC = {}
        self.optSC = np.inf
        self.optAlt = None
        self.computeOpt()

        self.distortions = [None] * self.horizon
        self.distortions[0] = self.getDistortion(self.a[0])
        #self.altToDist = {alt:altToSC(alt)/self.optSC for alt in self.decisionSpace}

    def computeOpt(self):
        for alternative in self.decisionSpace:          #caching this assuming these don't get too large
            sc = 0
            for agent in self.agents:
                sc += self.decisionSpace.disutility(agent.blissPoint, alternative)
            self.altToSC[alternative] = sc
            if sc < self.optSC:
                self.optSC = sc
                self.optAlt = alternative

    def getDistortion(self, a):
        return self.altToSC[a] / self.optSC

    def getWorstDistortion(self):
        return max([self.getDistortion(a) for a in self.decisionSpace])

    def getBlissPoints(self):
        return [agent.blissPoint for agent in self.agents]

    def step(self, i):
        u, v = random.choices(self.agents, k=2) #sample w/ replacement
        self.a[i+1]  = self.bargainFn(u, v, self.a[i], self.decisionSpace, self)
        self.distortions[i+1] = self.getDistortion( self.a[i+1])

    def deliberate(self):
        #print("Distortion at step " + str(0) + ": " + str(self.distortions[0]))
        for i in range(self.horizon-1):
            self.step(i)
            #print("Distortion at step " + str(i+1) + ": " + str(self.distortions[i+1]))

        #print(self.a)
        #print("Final Distortion :" + str(self.distortions[-1]))
        return self.a, self.distortions



def nashBargain(u, v, a, dS, sd):
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

def imperfectNashBargain(u, v, a, dS, sd):
    a = nashBargain(u,v,a,dS, sd)
    if u.blissPoint == v.blissPoint:
        return a
    if u.bargainPower > v.bargainPower:
        a_shifted =  dS.shiftAlt(a, u.blissPoint, u.bargainPower - v.bargainPower)
    else:
        a_shifted =  dS.shiftAlt(a, v.blissPoint, v.bargainPower - u.bargainPower)
    return a_shifted


def benignNash(u, v, a, dS, sd):
    a1 = nashBargain(u,v,a,dS, sd)
    #print("Initial Blisses: " + str(u.blissPoint) + "   " + str(v.blissPoint) + "    a = " + str(a))
    #print("weights: " + str(1/u.bargainPower) + "   " + str(1/v.bargainPower) )
    u.blissPoint = dS.shiftAltTowardsTwo(u.blissPoint, v.blissPoint, a, 1/u.bargainPower)
    v.blissPoint = dS.shiftAltTowardsTwo(v.blissPoint, u.blissPoint, a, 1/v.bargainPower)
    #print("Final Blisses: " + str(u.blissPoint)+ "   "  + str(v.blissPoint))
    sd.computeOpt() #SLOW SLOW SLOW
    return a1

def genIntervalPlot(sd, save_dir, suffix = ""):
    fig, ax = plt.subplots()
    bp = sd.getBlissPoints()
    ax.hist(bp, bins=20, color="lightblue")

    ax.set(xlabel='Bliss Point', ylabel='Frequency')
    ax.set_xlim(sd.decisionSpace.dSpace[0],sd.decisionSpace.dSpace[-1])
    opt = ax.axvline(x=sd.optAlt, color="blue", ls='-')
    act = ax.axvline(x=sd.a[-1], color="orange", ls='--')

    ax.legend((act, opt), ("Final Social Choice", "Generalized Median"))
    #ax.grid()

    fig.savefig(save_dir + "/blisses_" + suffix +".png")
    plt.close()


def genDoubleHistPlot(bp1, bp2, i_opt, sd, save_dir, suffix = ""):
    fig, ax = plt.subplots()

    ax.hist(bp1, bins=20, color="lightblue", alpha=0.5)
    ax.hist(bp2, bins=20, color="lightcoral", alpha=0.5)

    ax.set(xlabel='Bliss Point', ylabel='Frequency')
    ax.set_xlim(sd.decisionSpace.dSpace[0],sd.decisionSpace.dSpace[-1])
    opt = ax.axvline(x=i_opt, color="blue", ls='-')
    post = ax.axvline(x=sd.optAlt, color="red", ls='-')
    act = ax.axvline(x=sd.a[-1], color="orange", ls='--')

    ax.legend((act, opt, post), ("Final Social Choice", "Initial Generalized Median", "Final Generalized Median"))
    #ax.grid()

    fig.savefig(save_dir + "/blisses_" + suffix +".png")
    plt.close()


def genSummaryPlots(distortions, save_dir, suffix = ""):
    fig, ax = plt.subplots()
    ax.hist(distortions[:,-1], bins=40, color="lightblue")

    ax.set(xlabel='Distortion', ylabel='Frequency')
    ax.set_xlim(1,2)
    opt1 = ax.axvline(x=1.208, color="red", ls='-')
    opt2 = ax.axvline(x=1.125, color="red", ls='-')
    act = ax.axvline(x=np.mean(distortions[:,-1]), color="orange", ls='-')

    ax.legend((act, opt1), ("Mean", "Theoretical Bounds"))
    #ax.grid()

    fig.savefig(save_dir + "/distortions" + suffix + ".png")
    plt.close()

def genDistOvTime(d, d_imp, d_bgn, save_dir, suffix = ""):
    fig, ax = plt.subplots()

    sc_d = ax.errorbar(list(range(d.shape[1])), np.mean(d, axis=0), yerr=[-np.quantile(d, 0.25, axis=0)+np.mean(d, axis=0), np.quantile(d, 0.75, axis=0)-np.mean(d, axis=0)], fmt=".k",  capsize=1)
    sc_dimp = ax.errorbar(list(range(d_imp.shape[1])), np.mean(d_imp, axis=0), yerr=[-np.quantile(d_imp, 0.25, axis=0)+np.mean(d, axis=0), np.quantile(d_imp, 0.75, axis=0)-np.mean(d, axis=0)],fmt=".b", capsize=1)
    sc_dbgn = ax.errorbar(list(range(d_bgn.shape[1])), np.mean(d_bgn, axis=0), yerr=[-np.quantile(d_bgn, 0.25, axis=0)+np.mean(d, axis=0), np.quantile(d_bgn, 0.75, axis=0)-np.mean(d, axis=0)], fmt=".g",  capsize=1)

    ax.set(xlabel='Deliberation Step', ylabel='Distortion')
    opt1 = ax.axhline(y=1.208, color="red", ls='-')
    opt2 = ax.axhline(y=1.125, color="red", ls='-')

    ax.legend((sc_d, sc_dimp, sc_dbgn, opt1), ("Nash", "Selfish Nash", "Unselfish Nash", "Theoretical Bounds"))
    #ax.grid()

    fig.savefig(save_dir + "/distortionsOverTime" + suffix + ".png")
    #plt.show()
    plt.close()


def main():
    args = get_args()
    if args.run_name != '':
        save_dir = utils.get_save_dir('save/', args.run_name)
        log_path = os.path.join(save_dir, f'log.txt')
        with open(log_path, 'w') as fp:
            for a in args.__dict__:
                fp.write(a + ": " + str(args.__dict__[a]) + "\n")

    distortions = np.zeros((args.num_simulations, args.horizon+1))
    distortions_bgn = np.zeros((args.num_simulations, args.horizon+1))
    distortions_imp = np.zeros((args.num_simulations, args.horizon+1))

    print("First stage")
    for i in range(args.num_simulations):
        sd = SequentialDeliberation(args.num_agents, DiscretizedDecisionSpace(args.num_alternatives, args.num_types), nashBargain, args, seed=args.seed+i)
        a, distortions[i,:] = sd.deliberate()

        if i % args.plot_interval == 0:
            genIntervalPlot(sd, save_dir, str(i))

    genIntervalPlot(sd, save_dir, str(args.num_simulations))
    genSummaryPlots(distortions, save_dir)

    with open(log_path, 'a+') as fp:
            fp.write("Distortion = " + str(np.mean(distortions[:,-1])) + "+/-" + str(np.std(distortions[:,-1])) + "\n")

    print("Second stage")
    for i in range(args.num_simulations):
        sd = SequentialDeliberation(args.num_agents, DiscretizedDecisionSpace(args.num_alternatives, args.num_types), imperfectNashBargain, args, seed=args.seed+i,bargain_dev=args.bargain_dev)
        a, distortions_imp[i,:] = sd.deliberate()

        if i % args.plot_interval == 0:
            genIntervalPlot(sd, save_dir, str(i) + "_imp")

    genIntervalPlot(sd, save_dir, "_imp" + str(args.num_simulations))
    genSummaryPlots(distortions_imp, save_dir, "_imp")

    with open(log_path, 'a+') as fp:
            fp.write("Imp Distortion = " + str(np.mean(distortions_imp[:,-1])) + "+/-" + str(np.std(distortions_imp[:,-1])) + "\n")

    print("Third stage")
    for i in range(args.num_simulations):
        sd = SequentialDeliberation(args.num_agents, DiscretizedDecisionSpace(args.num_alternatives, args.num_types), benignNash, args, seed=args.seed+i,bargain_dev=args.bargain_dev)
        if i % args.plot_interval == 0 or i == args.num_simulations-1:
            bp1 = sd.getBlissPoints()
            i_opt = sd.optAlt

        a, distortions_bgn[i,:] = sd.deliberate()

        if i % args.plot_interval == 0:
            bp2 = sd.getBlissPoints()
            genDoubleHistPlot(bp1,bp2,i_opt, sd, save_dir, str(i) + "_bgn")

    bp2 = sd.getBlissPoints()
    i_opt = sd.optAlt
    genDoubleHistPlot(bp1,bp2,i_opt, sd, save_dir, str(i) + "_bgn")
    genSummaryPlots(distortions_bgn, save_dir, "_bgn")

    with open(log_path, 'a+') as fp:
            fp.write("Bgn Distortion = " + str(np.mean(distortions_bgn[:,-1])) + "+/-" + str(np.std(distortions_bgn[:,-1])) + "\n")

    genDistOvTime(distortions, distortions_imp, distortions_bgn, save_dir)



if __name__ == '__main__':
    main()