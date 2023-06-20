#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/18 11:48
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : GA.py
# @Statement : Genetic Algorithm (GA)
# @Reference : Holland J H. Genetic algorithms[J]. Scientific American, 1992, 267(1): 66-73.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(x):
    # pressure vessel design problem
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3 ** 2 - 4 * np.pi * x3 ** 3 / 3 + 1296000
    g4 = x4 - 240
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3
    else:
        return np.inf


def tournament_selection(pop, objs, pc, k=2):
    # binary tournament selection
    (npop, dim) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, dim))
    for i in range(nm):
        selection = np.random.choice(npop, k)
        ind = selection[np.argmin(objs[selection])]
        mating_pool[i] = pop[ind]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < pm / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def main(npop, iter, lb, ub, pc=1, eta_c=20, pm=0.5, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: upper bound
    :param ub: lower bound
    :param pc: crossover probability
    :param eta_c: spread factor distribution index
    :param pm: mutation probability
    :param eta_m: perturbance factor distribution index
    :return:
    """
    # Step 1. Initialization
    dim = len(lb)  # dimension
    pop = np.random.rand(npop, dim) * (ub - lb)  # population
    objs = np.array([cal_obj(pop[i]) for i in range(npop)])  # objectives
    gbest = min(objs)  # the global best
    gbest_sol = pop[np.argmin(objs)]  # the global best solution
    iter_best = []  # the global best of each iteration
    con_iter = 0  # the convergence iteration

    # Step 2. The main loop
    for t in range(iter):

        # Step 2.1. Crossover + mutation
        mating_pool = tournament_selection(pop, objs, pc)
        offspring = crossover(mating_pool, lb, ub, eta_c)
        offspring = mutation(offspring, lb, ub, pm, eta_m)
        new_objs = np.array([cal_obj(offspring[i]) for i in range(len(offspring))])
        pop = np.concatenate((pop, offspring), axis=0)
        objs = np.concatenate((objs, new_objs), axis=0)
        rank = np.argsort(objs)[:npop]
        pop = pop[rank]
        objs = objs[rank]

        # Step 2.2. Update gbest
        if min(objs) < gbest:
            gbest = min(objs)
            gbest_sol = pop[np.argmin(objs)]
            con_iter = t + 1
        iter_best.append(gbest)

    # Step 3. Sort the results
    x = np.arange(iter)
    plt.figure()
    plt.plot(x, iter_best, linewidth=2, color='blue')
    plt.xlabel('iteration number')
    plt.ylabel('gbest')
    plt.title('convergence curve')
    plt.savefig('convergence curve.png')
    plt.show()
    return {'gbest': gbest, 'best solution': gbest_sol, 'convergence iteration': con_iter}


if __name__ == '__main__':
    t_npop = 300
    t_iter = 1500
    t_lb = np.array([0, 0, 10, 10])
    t_ub = np.array([99, 99, 200, 200])
    print(main(t_npop, t_iter, t_lb, t_ub))
