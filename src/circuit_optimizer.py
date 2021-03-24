class Specifications:
  '''
  Specifications for circuit sizing.
  '''
  def __init__(self, lt={}, gt={}, objective=None):
    '''
    Constructor
    Args:
    - lt dictionary with less than measures and their target value
    - gt dictionary with large then measures and their target value
    '''
    self.lt = lt
    # {'idd': 35e-6,'pm' : 90.0}
    self.gt = gt
    # {'gdc': 50,'gbw': 35e6}
    self.objective = objective

  def __str__(self):
    return "obj: {}, lt: {} gt: {}".format(self.objective, self.lt, self.gt)

  def update(self, lt, gt):
    """
    """
    self.lt.update(lt)
    self.gt.update(gt)

  def verifyMOO(self, measures):
    log = {}
    gsum = 0
    obj = []
    for meas, limit in self.lt.items():
      if meas in measures and measures[meas] != None:
        if(measures[meas] > limit):
          gsum += (limit - measures[meas])/abs(limit) 
          log[meas+"_lt"] = (limit, measures[meas])
      else:
        gsum += -1000
        log[meas+"_lt"] = (limit, measures[meas])
        
    for meas, limit in self.gt.items():
      if meas in measures and measures[meas] != None:
        if(measures[meas] < limit):
          gsum += (-limit + measures[meas])/abs(limit) 
          log[meas+"_gt"] = (limit, measures[meas])
      else:
        gsum += -1000
        log[meas+"_gt"] = (limit, measures[meas])
    
    if self.objective is not None:
      obj = [ (measures[obj[0]] if measures[obj[0]] is not None else -1000)*obj[1] for obj in self.objective]
      
    return obj, gsum, log
    
    
    
  def verify(self, measures):
    obj, gsum, log = self.verifyMOO(measures)
    
    return sum(obj) + gsum / (len(self.gt) + len(self.lt)), gsum==0, log
  

  
  def asarray(self):
    return list(self.lt.values()) + list(self.gt.values()) 

if __name__ == '__main__':
  
  parameters = ('_w8','_w6','_w4','_w10','_w1','_w0',
          '_l8','_l6','_l4','_l10','_l1','_l0',
          "_nf8","_nf6", "_nf4", "_nf10", "_nf1", "_nf0" )
          
  values     = ( 1.0000e-06, 7.1800e-05, 1.5700e-05, 2.2000e-06, 1.6000e-06, 9.0000e-06,
           9.4000e-07, 8.8000e-07, 6.7000e-07, 8.9000e-07, 8.9000e-07, 8.4000e-07,
           5.0000e+00, 1.0000e+00, 7.0000e+00, 1.0000e+00, 3.0000e+00, 3.0000e+00)
  
  folder = "/home/aida/workspace/DeepLearning/ssvcamp-ngspice/umc_013/"
  
  measures = simulate(cwd = folder, netlist="3a_VCOTA_OLtb_AC_OP.cir", param = parameters, val = values)
  
  
  print(measures)	
    



if __name__ == '__main__':
  seed = 17
  np.random.seed(seed)
  random.seed(seed)
  
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
  
  circuit = VCAmpMOOProb(
    ng.Specifications(objective=[('idd', 1), ('gbw', -1) ], lt={'idd': 35e-5,'pm' : 90.0},gt=gt))
  a = Nsga2(circuit, eva=100*2 )

  pop_dec, pop_obj, pop_cstr = a.run()

  a.draw(pop_dec, pop_obj, pop_cstr )
  plt.show()

  print(circuit)