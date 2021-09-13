import sys
import subprocess
import glob
import threading
import os
import shutil
import numpy as np
import logging
import queue
from collections import namedtuple

import ngspice_meas as ngmeas


# global definitions for filenames
MEAS_OUT = '_meas.txt'
VARS     = 'design_var.inc'
GLOBAL_PARAMS = 'global_params.inc'

Simulation = namedtuple("Simulation", ["corner_id", "netlist_file", "meas_file", "sim_results", "values" ])


# Runs each netlist in interactive mode and saves the standard output to a file
def run_simulation(cwd, netlist_file, meas_file, timeout=2):
    '''
      runs ngspice  in interactive mode and saves the
      standard output to a file to collect measures.
      Gives a timeout [in minutes].
      param: netlist: the testbench to be simulated
      param: corner: current working directory (defautl=".")
      param: timeout: the timeour in minutes (defautl=2)
      '''
    try:
        #with open(os.path.join(cwd, meas_file), 'w') as f:
            subprocess.run(['ngspice', "-b", "-o", meas_file, netlist_file ], cwd=cwd, stdout= subprocess.DEVNULL, timeout=timeout*60)
    except (subprocess.TimeoutExpired,BlockingIOError) as e:
        logging.warning(e)


def parse_sim_data(sim_results, line, vector_id, vector_idx, vector_size, parse_vector):
    '''
    parses the values from the outupt file

    updates vector size and parse_vector
    '''
    content = line.split()
    if len(content) >= 3 and content[1] == '=':
        sim_results[content[0].upper()] = float(content[2])
    elif content[0] == 'Index':
        parse_vector = True
    elif parse_vector:
        try:
            sim_results[vector_id][vector_idx] = [ float(x) for x in content[1:]]
            vector_idx = vector_idx + 1
            if vector_idx == vector_size: parse_vector = False
        except Exception as e:
            logging.warning(e + "was raised when parsing " + line)

    
    return vector_idx, vector_size, parse_vector



def parse_sim_results(cwd, meas_file):
    sim_results = {}
    vector_id = None
    parse_vector = False
    vector_idx = 0
    vector_size = None
    try:
        with open(os.path.join(cwd, meas_file), 'r') as file:
            for line in file:
                line = line.strip() 
                if (not line) or (line[0] == '*' ) or (line[0] == '-'): continue

                try: 
                    if "AC output" in line: 
                        vector_id = "AC"
                        vector_size =  int(line.split()[-1])
                        vector_idx = 0
                        sim_results[vector_id] = [None]*vector_size
                    elif "NOISE output" in line:
                        vector_id = "NOISE"
                        vector_size =  int(line.split()[-2])
                        vector_row_size =  int(line.split()[-1])
                        vector_idx = 0
                        sim_results[vector_id] = np.full((vector_size, vector_row_size), np.NaN)
                    else:
                        if "=" in line:
                            content = line.split()
                            if len(content) >= 3 and content[1] == '=': sim_results[content[0].upper()] = float(content[2])
                        elif 'Index' in line:
                            parse_vector = True
                        elif parse_vector:
                            content = line.split()
                            try:
                                sim_results[vector_id][vector_idx, :] = content[1:]
                                vector_idx = vector_idx + 1
                                if vector_idx == vector_size: parse_vector = False
                            except Exception as e:
                                logging.warning(str(e) + "was raised when parsing " + line)
                except ValueError as e:
                    logging.warning(str(e) + " when parsing " + line )

    except FileNotFoundError:
        logging.warning(os.path.join(cwd, meas_file) + " file not found!")

    return sim_results



def worker(q, cwd, param):
    '''
    worker thread function that just runs the simulatiosn and collect the sim_results
    '''
    prev_values = None
    while True:
        simulation = q.get()
        if simulation is None: break 
        #updated design_var if needed
        if simulation.values is not prev_values:
            write_design_variables(cwd, param, simulation.values)
            prev_values = simulation.values
        #remove meas file

        run_simulation(cwd, simulation.netlist_file, simulation.meas_file)
        curr_meas = parse_sim_results(cwd, simulation.meas_file)
        with simulation.sim_results as meas:
            if simulation.corner_id in meas: meas[simulation.corner_id].update(curr_meas)
            else: meas[simulation.corner_id] = curr_meas
        q.task_done()
        

def load_corners(corners):
    corners_dic = {}
    corner_name = "UNDEF"
    try:
        with open(corners, 'r') as file:
            for line in file:
                line = line.strip() 
                if (not line) or (line[0] == '*'): continue
                if '.ALTER' in line.upper():
                    corner_name = line.split()[1]
                    corners_dic[corner_name] = list()
                elif line[0] == '+':
                    corners_dic[corner_name][-1] += "\n"+line
                else:
                    corners_dic[corner_name].append(line)
    except FileNotFoundError:
        pass

    return corners_dic



def build_corner_netlist(cwd, corners_data, netlist):
     for n in netlist: create_corner_netlist(n, cwd, corners_data)

def create_corner_netlist(netlist,cwd, corners_data):
    with open(os.path.join(cwd,netlist), "r") as f:
        netlist_data = f.read()
        
    for corner, values in corners_data.items():
        corner_netlist = os.path.join(cwd, corner + "_" + netlist)

        with open(corner_netlist, 'w') as f:
            f.write(netlist_data.replace(".include 'nominal.corner'", "\n".join(values)))
       


def remove_tmp_files(netlist, cwd):
    for f in glob.glob(cwd+'*'+MEAS_OUT): os.remove(f)
    for f in glob.glob(cwd+'*'+VARS): os.remove(f)
    for f in glob.glob(cwd+'*'+GLOBAL_PARAMS): os.remove(f)
    
    for n in netlist:
        for f in glob.glob(cwd+'*_'+n): os.remove(f)
    


def write_design_variables(cwd, param, val):
    with open(os.path.join(cwd,VARS), 'w') as outFile:
      outFile.write('*  Design Var\n .param\n')
      outFile.write('\n'.join(('+{}={}'.format(p,v) for p, v in zip(param, val))))

        
  
class ThreadSafeDict(dict):
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()

class Ngspice():
    '''
    '''

    def __init__(self, cwd, netlist, param,  corners='corners.inc', num_threads=6):
        '''
    The netlist must be built including the nominal libraries and parameter 
    definitions in a file named: 	nominal.corner 

    Additional corners can be specified in a coners file using .alter syntax.
    Exmaple corners.inc:
    .ALTER SS
        .lib '../ptm130/ptm-130.lib' SS
        .param vdd = 1.5

    .ALTER FF
        .lib '../ptm130/ptm-130.lib' FF
        .param vdd = 1.5
    EOF

        params:
         cwd: current working directory string
         netlist: list of netlists <strings>
         corners: name os corner file
         num_treads: number of threads
        '''
        self.cwd = cwd
        self.netlist = netlist 
        self.corners_data = load_corners(os.path.join(self.cwd, "cir", corners)) if corners is not None else {}
        self.num_threads=min(num_threads, os.cpu_count())
        self.param = param
        
        #list to hold the simulations to be executed per point
        # netlist * corners
        self.simulation_list = []
        for n in self.netlist:
            #place nominal 'TT' in queue
            self.simulation_list.append(Simulation('TT',n, n+MEAS_OUT, None, None))
            #if corners defined, build their simulation definitions
            for corner in self.corners_data.keys():
                self.simulation_list.append(Simulation(corner,corner + "_" + n, corner + "_" + n + MEAS_OUT, None, None))

        # create files for worker threads
        thread_source_path = os.path.join(self.cwd, "t")
        if os.path.exists(thread_source_path): shutil.rmtree(path=thread_source_path)
        shutil.copytree(src=os.path.join(self.cwd, "cir"), dst=thread_source_path)

        #create corner netlist in source dir for worker threads
        build_corner_netlist(thread_source_path,  self.corners_data, self.netlist)

        #copy folders fo reach worker thread and launch thread    
        self.q = queue.Queue()

        for tid in range(self.num_threads):
            thread_path = os.path.join(self.cwd, "t"+str(tid))
            if os.path.exists(thread_path):  shutil.rmtree(thread_path)
            shutil.copytree(src=thread_source_path, dst=thread_path)
            t = threading.Thread(target=worker, args=(self.q, thread_path, self.param))
            t.daemon = True
            t.start()

    def __del__(self):
        '''
        Tells worker threads to stop, exits imediatly
        '''
        for _ in range(self.num_threads):
            self.q.put(None) #poison pill
        # not waiting on self.q.join() , workser dont update 
        # task count on exit either
        
        

    def simulate (self, values):
        '''
        Runs ngspice.
        param: values: ndarray of shape (n, len(self.param)
        '''
        n = values.shape[0]

        #List of dictionariy that will hold sim_results of all corners
        sim_results = [ThreadSafeDict() for _ in range(n)]

        # fills jobs queue
        for i in range(n):
            for sim in self.simulation_list: 
                self.q.put(Simulation(
                    sim.corner_id, sim.netlist_file, sim.meas_file, 
                    sim_results[i], values[i,:]))

        self.q.join()

        # consider to create/update worker threads for this also
        for i in range(n):
            ngmeas.compute_additional_measures(sim_results[i])

        return sim_results


def simulate(cwd, netlist, param, val, corners='corners.inc', num_threads=6):
    '''
    Runs ngspice.

    The netlist must be built including the nominal libraries and parameter 
    definitions in a file named: 	nominal.corner 

    Additional corners can be specified in a coners file using .alter syntax.
    Exmaple corners.inc:
    .ALTER SS
        .lib '../ptm130/ptm-130.lib' SS
        .param vdd = 1.5

    .ALTER FF
        .lib '../ptm130/ptm-130.lib' FF
        .param vdd = 1.5
    EOF
    '''
    return Ngspice(cwd,netlist,param, corners,num_threads).simulate(val)



if __name__ == '__main__':
    
    for _ in range(100):
        run_simulation("./circuit_examples/ptm130_folded_cascode/cir", "tb_ac.cir", "tb_ac.cir_meas.txt.test", timeout=2)
        curr_meas = parse_sim_results("./circuit_examples/ptm130_folded_cascode/cir", "tb_ac.cir_meas.txt.test")
