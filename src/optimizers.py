'''
This is the implentations of Simulated Annealing and NSGA-II used in the paper.

Created on Nov, 2020

@author: Nuno Lourenço <nlourenco@lx.it.pt>

NSGA - Adapted from https://github.com/ChengHust/NSGA-II
Updated to handle constraint optimization and fast non-dominated sorting

SA - Addapted from Matthew T. Perry's code from https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py 
'''

import numpy as np

import math
from itertools import repeat
from collections import Sequence


class Problem(object):
    """
    The problem related parameters and variation operators of 
    cross over and mutation for GA, and move for SA.

    Parameters are handled in variation operators as real values. 
    Extending classes should round integers in the cost function if needed. 

    cost_fun() is defined for multi-objective multi-constraint optimization.
    Its up to the single objective optimizers to implement objective weigthing.
     
    objectives = ["name", ... ]
    parameters = ["name", ... ]
    min = [min, ... ]
    max = [max, ... ]
    
    """
    def __init__(self, d, min, max):
   
        self.d = d
        self.upper = max
        self.lower = min
        
    def __str__(self):
        return "Target: {}".format(self.target) 

 
    def cost_fun(self, x):
        """
        calculate the objective and constraints vectors
        :param x: the decision vectors
        :return: the objective,  constraints, and additional data vectors
        """
        n = x.shape[0]
          
        obj = np.zeros((n, 1))
        
        cstr = np.zeros(n)

        # data associated with the solutions but not used in the optimization
        # it can be usefull to debug the cost function.
        # In analog IC optimization we will use this 
        # data to store the simulation outputs

        data = np.zeros((n, 1)) 

        return obj, cstr, data


    def individual(self, pop_vars):
        """
        turn decision vectors into individuals
        :param pop_vars: decision vectors
        :return: (pop_vars, pop_obj, pop_cstr)
        """
        pop_obj, pop_cstr, pop_data = self.cost_fun(pop_vars)
        return (pop_vars, pop_obj, pop_cstr, pop_data)

    def initialize(self, N):
        """
        initialize the population
        :param N: number of elements in the population
        :return: the initial population
        """
        pop_dec = (np.random.random((N, self.d)) * (self.upper - self.lower)) + self.lower
        return pop_dec

    def variation(self, pop_dec, mutation=0.1, crossover=0.6):
        """
        Generate offspring individuals
        :param boundary: lower and upper boundary of pop_dec once d != self.d
        :param pop_dec: decision vectors
        :return: 
        """
        dis_c = 10
        dis_m = 20
        pop_dec = pop_dec[:(len(pop_dec) // 2) * 2][:]
        (n, d) = np.shape(pop_dec)
        parent_1_dec = pop_dec[:n // 2, :]
        parent_2_dec = pop_dec[n // 2:, :]
        beta = np.zeros((n // 2, d))
        mu = np.random.random((n // 2, d))
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
        beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
        beta = beta * ((-1)** np.random.randint(2, size=(n // 2, d)))
        beta[np.random.random((n // 2, d)) < 0.5] = 1
        beta[np.tile(np.random.random((n // 2, 1)) > crossover, (1, d))] = 1
        offspring_dec = np.vstack(((parent_1_dec + parent_2_dec) / 2 + beta * (parent_1_dec - parent_2_dec) / 2,
                                   (parent_1_dec + parent_2_dec) / 2 - beta * (parent_1_dec - parent_2_dec) / 2))
        site = np.random.random((n, d)) < mutation
        mu = np.random.random((n, d))
        temp = site & (mu <= 0.5)
        
        lower, upper = np.tile(self.lower, (n, 1)), np.tile(self.upper, (n, 1))
        
        norm = (offspring_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                               (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                         1. / (dis_m + 1)) - 1.)
        temp = site & (mu > 0.5)
        norm = (upper[temp] - offspring_dec[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                               (1. - np.power(
                                   2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                                   1. / (dis_m + 1.)))
        offspring_dec = np.maximum(np.minimum(offspring_dec, upper), lower)
        return offspring_dec


    def move(self, parameter_values):
        """
        Inputs:
        - value - new device sizes
            self.values update is self.value += self.value + action*(self.ranges[:,1] - self.ranges[:, 0])    
        -
        Outouts: observation, reward, done, {}
        - observations array of concat [ values, measures]
        - reward +1 improved, -1 worsen, -1000 no sim, 1000 meet specs
        """
        action = np.random.normal(scale=0.1, size=len(parameter_values))
        parameter_values = parameter_values + action*(self.ranges[:,1] - self.ranges[:, 0])
        parameter_values = np.round(parameter_values / self.ranges[:,2])*self.ranges[:,2]
        parameter_values = np.fmin(np.fmax(parameter_values,self.ranges[:, 0]),self.ranges[:,1])
        
        return parameter_values


def half_tournemant(rank, cdist):
    n = len(rank)
    index_a = np.arange(n)
    np.random.shuffle(index_a)
    eq_rank = rank[index_a[:n//2]] == rank[index_a[n//2:]]
    larger_cdist = cdist[index_a[:n//2]] > cdist[index_a[n//2:]]

    decision_a = np.logical_or(np.logical_and(eq_rank, larger_cdist),  rank[index_a[:n//2]] < rank[index_a[n//2:]])

    return index_a[np.r_[decision_a, ~decision_a]]
    

def tournament(rank, cdist):
    '''
    tournament selection
    :param K: number of solutions to be compared
    :param N: number of solutions to be selected
    :param fit: fitness vectors
    :return: index of selected solutions
    '''
    n = len(rank)
    mate = np.zeros(n, dtype=np.int16)
    mate[::2] = half_tournemant(rank, cdist)
    mate[1::2] = half_tournemant(rank, cdist)

    return mate

def objective_dominance(pop_obj, i, j):
    """
    Computes objective-wise dominance between elements i and j of the population.
    :param pop_obj: the value of the populations' objectives
    :param i, j: the elems being compared

    :returns: dominator: the index of the dominator, None if i and j are non-dominated
              dominated: the index of the dominated, None if i and j are non-dominated
    """
    _,M = pop_obj.shape
    
    i_dominates_j = False
    j_dominates_i = False
                                
    for obj_idx in range(M):
        if pop_obj[i,obj_idx] < pop_obj[j,obj_idx] :
            i_dominates_j = True
        elif pop_obj[i,obj_idx] > pop_obj[j,obj_idx] :
            j_dominates_i = True
   
    if i_dominates_j and (not j_dominates_i):
        return i, j
    if (not i_dominates_j) and j_dominates_i: 
        return j, i  

    return None, None


def fnd_sort(pop_obj, pop_cstr):
    """
    Computes and sets the ranks of the population elements  using the fast non-dominated sorting method.
    :param pop_obj: population objectives (NxM)
    :param pop_cstr: population constraint violation (Nx1) 
      
    :returns: ranks: an array with the ranks
              max_rank: max rank
    """ 
    N,M = pop_obj.shape
   
    # structures for holding the domination info required for fast nd sorting 
    dominate = [[] for x in range(N)]   
    dominated_by_counter = np.zeros(N, dtype=int)   

    for i in range(N):
        for j in range(i+1,N):    
            #constrained pareto dominance
            if pop_cstr[i] == pop_cstr[j]:  
                #objective pareto dominance         
                dominator, dominated  = objective_dominance(pop_obj, i, j)       
                if dominator is not None:
                    dominate[dominator].append(dominated)
                    dominated_by_counter[dominated]+=1
            elif pop_cstr[i] < pop_cstr[j]:
                # j dominates i
                dominate[j].append(i)
                dominated_by_counter[i]+=1
            else:
                # i dominates j 
                dominate[i].append(j) 
                dominated_by_counter[j]+=1
    #assign the ranks and return           
    return assign_rank(dominate, dominated_by_counter)

def assign_rank(dominate, dominated_by_counter):
    """
    sets the ranks of the population elements  using the fast non-dominated sorting method.
    :param dominate: list of dominated population elements [[]*N]
    :param dominated_by_counter: counter of elements dominating  (Nx1) 
      
    :returns: ranks: an array with the ranks
              max_rank: max rank
    """
    N = len(dominate)
    
    ranks = np.inf * np.ones(N)
    current_rank = 1

    #    if non dominated  is part of front 1*/
    current_front = [i for i in range(N) if dominated_by_counter[i] == 0]

    while np.sum(ranks < np.inf) < N/2:
        ranks[current_front] = current_rank
        next_front = []

        for index_a in current_front:
            #  reduce the numbers of domination to the ones in its set of dominance         
            for index_b in dominate[index_a]:
                dominated_by_counter[index_b]-=1
                #  if( they become non dominated - then they are part of next front)
                if dominated_by_counter[index_b] == 0:
                    next_front.append(index_b)

        current_front = next_front
        current_rank+=1
    
    
               
    return ranks, current_rank-1


def crowding_distance(pop_obj, rank):
    """
    The crowding distance of the Pareto front "front_id"
    :param pop_obj: objective vectors
    :param rank: front numbers
    :return: crowding distance
    """
    n, M = np.shape(pop_obj)
    crowd_dis = np.zeros(n)
    fronts = np.unique(rank)
    fronts = fronts[fronts != np.inf]
    for f in range(len(fronts)):
        front = np.array([k for k in range(len(rank)) if rank[k] == fronts[f]])
        fmax = pop_obj[front, :].max(0)
        fmin = pop_obj[front, :].min(0)
        for i in range(M):
            sorted_index = np.argsort(pop_obj[front, i])
            crowd_dis[front[sorted_index[0]]] = np.inf
            crowd_dis[front[sorted_index[-1]]] = np.inf
            for j in range(1, len(front) - 1):
                crowd_dis[front[sorted_index[j]]] +=  \
                    (
                        pop_obj[(front[sorted_index[j + 1]], i)] - 
                        pop_obj[(front[sorted_index[j - 1]], i)]
                    ) / ((fmax[i] - fmin[i]) if fmax[i] != fmin[i] else 1.0)
    return crowd_dis



def environment_selection(pop_dec, pop_obj, pop_cstr, pop_data, n):
    '''
    Environmental selection in NSGA-II

    :param population: current population
    :param n: number of selected individuals
    :return: next generation population (
        decison vars, objectives, constraints, data,
        rank, and cdist)
    '''
    
    # fast non-dominated sorting and crowding distance
    # arguably they could be refractored out of this function
    fronts, max_front = fnd_sort(pop_obj, pop_cstr)
    crowd_dis = crowding_distance(pop_obj, fronts)

    #Select elements from all fronts except the last. Note that fnd_sort only 
    #sorts half the population. extra elements are only from the in the last from
    index = [i for i in range(len(fronts)) if fronts[i] < max_front]
    last  = [i for i in range(len(fronts)) if fronts[i]== max_front]
 
    delta_n = np.argsort(-crowd_dis[last])[: (n - len(index))]
    
    index.extend([last[i] for i in delta_n])
    
    return pop_dec[index,:], pop_obj[index,:], pop_cstr[index], [pop_data[i] for i in index], fronts[index], crowd_dis[index],index



class NSGA2:

       

    def minimize(self, problem, pop_size=100,  evaluations=100 * 500, mutation=0.2, crossover=0.8, initial_pop=None):
        """
        NSGA-II algorithm

        """
        if initial_pop is None:
            self.pop, self.pop_obj, self.pop_cstr, self.pop_data = problem.individual(problem.initialize(pop_size))
        else:
            self.pop, self.pop_obj, self.pop_cstr, self.pop_data = problem.individual(initial_pop)

        front_no, max_front = fnd_sort(self.pop_obj, self.pop_cstr)
        crowd_dis = crowding_distance(self.pop_obj, front_no)
        evals = evaluations
        yield self.pop, self.pop_obj,  self.pop_cstr, self.pop_data, evals, front_no
        while evals > 0:
            mating_pool = tournament(front_no, crowd_dis)       
            
            self.offspring_dec, self.offspring_obj, self.offspring_cstr, self.offspring_data = problem.individual(
                problem.variation(self.pop[mating_pool, :], mutation = mutation, crossover= crossover ))

            self.pop = np.vstack((self.pop, self.offspring_dec))
            self.pop_obj = np.vstack((self.pop_obj, self.offspring_obj))
            self.pop_cstr = np.concatenate((self.pop_cstr, self.offspring_cstr))
            self.pop_data = self.pop_data + self.offspring_data
            
            self.pop, self.pop_obj,self.pop_cstr, self.pop_data, front_no, crowd_dis,_ = environment_selection(self.pop, self.pop_obj, self.pop_cstr,self.pop_data, pop_size)
            
            evals = evals - pop_size
            yield self.pop, self.pop_obj,  self.pop_cstr, self.pop_data, evals, front_no

       
            #remove duplicates
            vals, index = np.unique(self.pop.round(decimals=9), axis=0, return_index =True)
            
            if len(index) < self.pop.shape[0] :              
                select = np.in1d(range(self.pop.shape[0]), index)
                self.pop[~select, :], self.pop_obj[~select, :], self.pop_cstr[~select], data = problem.individual(problem.initialize(self.pop.shape[0] - len(index)))
                for i, v in zip(select, data):
                    self.pop_data[i] = v
            
            

        return self.pop, self.pop_obj,  self.pop_cstr, self.pop_data




def default_mo_2_so(objs, cstr) :
    return sum(objs)/len(objs) + cstr


def simulated_annealing(problem, steps = 10000, t_max = 1500.0, t_min = 2.5, initial_state=None, convert_multi_obj = default_mo_2_so):
    '''
    Minimizes the energy of a system by simulated annealing.
     Parameters
      state : an initial arrangement of the system
        Returns
        (state, energy, objectives, constraints, data): the best state and energy found.
    '''
    if t_min <= 0.0: raise ValueError('Exponential cooling requires a minimum temperature greater than zero.')

    # Note initial state
    if initial_state is None :
        best_state, best_obj, best_cstr, best_data = problem.initialize(1)
    else:
        best_state, best_obj, best_cstr, best_data = problem.individual(initial_state)

    state = best_state
    prev_state, prev_obj, prev_cstr, prev_data =  best_state, best_obj, best_cstr, best_data

    best_value = prev_value = convert_multi_obj(best_obj, best_cstr)

    step = 0
     # Precompute factor for exponential cooling from Tmax to Tmin
    cooling_factor = -math.log(t_max / t_min)
    # Attempt moves to new states
    while step < steps:
      step += 1
      T = t_max * math.exp(cooling_factor * step / steps)
      
      state, obj, cstr, data = problem.individual(problem.move(state))
      value = convert_multi_obj(obj, cstr)
      
      
      dV = 100*(value - prev_value)
      if dV > 0.0 and math.exp(-dV / T) < np.random.random():
        # Restore previous state
        state, obj, cstr, data, value = prev_state, prev_obj, prev_cstr, prev_data, prev_value
      else:
        # Accept new state and compare to best state
        prev_state, prev_obj, prev_cstr, prev_data, prev_value =  state, obj, cstr, data, value
        if value < best_value:
          best_state, best_obj, best_cstr, best_data, best_value, = state, obj, cstr, data, value
    # Return best state and energy
    return best_state, best_value




  
if __name__ == '__main__':
  seed = 17
  np.random.seed(seed)
 
  
  sat_conditions = {}
  sat_conditions["vov_mpm0"] = 0.05
  sat_conditions["vov_mpm1"] = 0.05
  sat_conditions["vov_mpm2"] = 0.05
  sat_conditions["vov_mpm3"] = 0.05
  sat_conditions["vov_mnm4"] = 0.05
  sat_conditions["vov_mnm5"] = 0.05
  sat_conditions["vov_mnm6"] = 0.05
  sat_conditions["vov_mnm7"] = 0.05
  sat_conditions["vov_mnm8"] = 0.05
  sat_conditions["vov_mnm9"] = 0.05
  sat_conditions["vov_mnm10"] = 0.05
  sat_conditions["vov_mnm11"] = 0.05

  sat_conditions["delta_mpm0"] = 0.1
  sat_conditions["delta_mpm1"] = 0.1
  sat_conditions["delta_mpm2"] = 0.1
  sat_conditions["delta_mpm3"] = 0.1
  sat_conditions["delta_mnm4"] = 0.1
  sat_conditions["delta_mnm5"] = 0.1
  sat_conditions["delta_mnm6"] = 0.1
  sat_conditions["delta_mnm7"] = 0.1
  sat_conditions["delta_mnm8"] = 0.1
  sat_conditions["delta_mnm9"] = 0.1
  sat_conditions["delta_mnm10"] = 0.1
  sat_conditions["delta_mnm11"] = 0.1
  
  gt={'gdc': 50,'gbw': 35e6,'pm' : 45.0, 'fom': 900}
  gt.update(sat_conditions)
  
  circuit = VCAmplifierCircuitOptProblem(
    ng.Specifications(objective=[('idd', 1)], lt={'idd': 35e-5,'pm' : 90.0},gt=gt), discrete_actions = False)
  sa = SA()

  print(circuit)

  for iter, stats in sa.minimize(circuit): 
    print("\r iter {}: {}".format(iter, stats))

  print(sa.best_state)
  print(circuit.simulate(sa.best_state))
  print(circuit.target.verify(circuit.simulate(sa.best_state)))