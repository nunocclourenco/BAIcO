'''
Created on Feb 1, 2019

@author: aida
'''
import math
import numpy as np
import optimizers as opt
import json
import ngspice



class Circuit(opt.Problem):
    def __init__(self, folder, setup_file="circuit_setup.json", corners="corners.inc"):
        '''
        Defines the design variables, ranges and  all needeed parameters to simualte circuit 
        '''
        with open(folder + setup_file, 'r') as file:
            setup = json.load(file)

        self.folder = folder
        self.target = OptimizationTarget(setup["objectives"], setup["constraints"])
        self.testbenches = setup["testbenches"]
        
        self.objectives = self.target.objectives
        #parameters names wihc are the desing variables
        self.parameters = []
        #parameters ranges (mion, max, grid)
        self.ranges = []

        for range_def in setup["ranges"]:
            for param in range_def["params"]:
                self.parameters.append(param)
                self.ranges.append(list((range_def["min"],range_def["max"], range_def["grid"] )))

        self.ranges = np.array( self.ranges)
 
        self.ngspice = ngspice.Ngspice(self.folder, self.testbenches, self.parameters,corners=corners, num_threads=7)
 
        super().__init__(len(self.parameters), self.range_min(), self.range_max())


    def initialize(self, N):
        '''
        Creates N random designs.
        '''
        values = super().initialize(N)
        values = np.round(values / self.ranges[:,2])*self.ranges[:,2] 
        return values

    def __str__(self):
        return "Running Folder: {}\nParameters: {}\nRanges: {}".format(self.folder, self.parameters, self.ranges)

    def get_parameters(self):
        return self.parameters
    
    def sort_param_values(self, sizing):
        return [sizing[p] for p in self.parameters]

    def range_min(self):
        ''' 
        Return range.min as a column vector 
        '''
        return self.ranges[:,0]
  
    def range_max(self):
        ''' 
        Return range.max as a column vector 
        '''
        return self.ranges[:,1]
  
    def simulate(self, values):
        '''
        Simulate n circuits circuit using the values as parameters 
        in all test benches and colects measures.
        inputs:
        values a numpy array with the x solutions to simulate
        
        Returns: the measures as a dict (indexed by corner) of dicts (indexed by meas)for each point in a list
        '''
        try:
            parameter_values =np.expand_dims(np.array([ values[k] for  k in self.parameters ]), axis=0)
        except Exception:
            parameter_values = values


        assert parameter_values.shape[1] == len(self.parameters)
     
        
        sim_results = self.ngspice.simulate( parameter_values)

        return sim_results

    def variation(self, pop_dec, mutation=0.1, crossover=0.6):
        parameter_values = super().variation(pop_dec, mutation, crossover)
        #move to grid
        parameter_values = np.round(parameter_values / self.ranges[:,2])*self.ranges[:,2]
        parameter_values = np.fmin(np.fmax(parameter_values,self.ranges[:, 0]),self.ranges[:,1])
        return parameter_values
   

    def cost_fun(self, x):
        """
        Simulates n design points, and compute the cost funciton
        calculate the objective and constraints vectors
        :param x: the decision vectors
        :return: the objective,  constraints, and additional data vectors
          data is a list of tuples containing the meas form the simulator 
          and the cost evaluation report from the target 
        """
        n = x.shape[0]
      
        obj = np.zeros((n, len(self.target.objectives)))
        cstr = np.zeros((n, 1))
        data = [None]*n
        measures = self.simulate(x)

        for i in range(n):
            obj[i,:],cstr[i], log = self.target.evaluate(measures[i])

            # data associated with the solutions but not used in the optimization
            # it can be usefull to debug the cost function.
            # In analog IC optimization we will use this 
            # data to store the simulation outputs
            data[i] = (measures[i], log)
        return obj, cstr, data


__objective_directions = {"MIN":1, "MAX":-1}

def parse_objectives(objs_def):
    objectives = []

    for obj_str in objs_def:
        token = obj_str.upper().split()
        if len(token) == 2 and token[0] in __objective_directions:
            objectives.append((token[1], __objective_directions[token[0]]))
        else: 
            raise ValueError("Found {} in objective definition, expectes 'min meas' or 'max meas'".format(obj_str))
    return objectives if objectives else None


def parse_constraints(constraint_list):
    '''
    Parses an iterable  of constraint strings in the format: "meas > value" or "meas < value"
    Multiples measure identifiers are also valid, e.g.,  "meas1 meas2 > value".

    params: constraint_list: an itrable of string that define the optimization target constraints
    returns: the constraints dictionaries.
    '''
    constraints = {"<":{}, ">":{}}
  
    for cstr_str in constraint_list:
        token = cstr_str.upper().split()
    
        value = float(token[-1])
        if token[-2] in constraints:
            for meas in token[:-2]:  
                constraints[token[-2]][meas] = value
        else: 
            raise ValueError("Found {} in constraint  definition, expects 'meas > value' or 'meas < value'".format(cstr_str))
    return constraints["<"] , constraints[">"]



class OptimizationTarget:
    '''
    Specifications for circuit sizing.
    '''
    def __init__(self, objectives, constraints):
      '''
      Constructor
      Args:
      - lt dictionary with less than measures and their target value
      - gt dictionary with large then measures and their target value
      '''
  
      
      self.lt, self.gt = parse_constraints(constraints)
      self.objectives = parse_objectives(objectives)

    def __str__(self):
      return "obj: {}, lt: {} gt: {}".format(self.objective, self.lt, self.gt)

    def update(self, lt, gt):
      """
      """
      self.lt.update(lt)
      self.gt.update(gt)


    def evaluate_corner(self, measures):
      log = {}
      gsum = 0
      obj = []
      
      for meas, limit in self.lt.items():
        if meas not in measures or measures[meas] == None:
          log[meas+"_lt"] = (limit, None)
          gsum += -1000
        elif measures[meas] > limit:
          log[meas+"_lt"] = (limit, measures[meas], (limit - measures[meas])/abs(limit)  )
          gsum += (limit - measures[meas])/abs(limit) 
          
      for meas, limit in self.gt.items():
        if meas not in measures or measures[meas] == None:
          log[meas+"_gt"] = (limit, None)
          gsum += -1000
        elif measures[meas] < limit:
          log[meas+"_gt"] = (limit, measures[meas], (-limit + measures[meas])/abs(limit))
          gsum += (-limit + measures[meas])/abs(limit) 

      if self.objectives is not None:
        obj = [ (measures[obj[0]]*obj[1]) if obj[0] in measures and measures[obj[0]] is not None else math.inf for obj in self.objectives]
        
      return obj, gsum, log
      
    def evaluate(self, measures):  
      log = {}
      gsum = 0
      obj = [-math.inf] * len(self.objectives)
      for corner, meas in measures.items():
        obj_cnr, gsum_cnr, log_cnr = self.evaluate_corner(meas)
        log[corner] = log_cnr
        gsum += gsum_cnr
        obj = [ max(o1, o2) for o1, o2 in zip(obj, obj_cnr) ]

      return obj, gsum, log




      
    def evaluate_single(self, measures):
      obj, gsum, log = self.evaluate(measures)
      
      return sum(obj) + gsum / (len(self.gt) + len(self.lt)), gsum==0, log
    

    
    def asarray(self):
      return list(self.lt.values()) + list(self.gt.values())   







if __name__ == "__main__":
    c = Circuit("./circuit_examples/ptm130_folded_cascode/")
    with open("./circuit_examples/ptm130_folded_cascode/sizing_example.json", 'r') as file:
        sizing = json.load(file)


    meas = c.simulate(np.array([c.sort_param_values(sizing)]))

    print(meas)

    obj,cstr, log = c.cost_fun(np.array([c.sort_param_values(sizing)])) 

    print(obj, cstr)
    print(log[0][1])
    for k,v in log[0][1]['TT'].items():
        print(k, v)





