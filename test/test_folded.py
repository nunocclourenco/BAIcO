import json
import mtprof
import numpy as np

import circuits as cir
import ngspice
import optimizers as opt


def test_folded_corners_sim():

    process = ["SS", "FF", "SNFP", "FNSP"]
    voltage_tempt = ["VDD_MAX_TEMP_MAX", "VDD_MAX_TEMP_MIN", "VDD_MIN_TEMP_MAX", "VDD_MIN_TEMP_MIN"]

    corner_set = set(["TT"]  + [ p+"_"+vt for p in process for vt in voltage_tempt])
    
    with open("./circuit_examples/ptm130_folded_cascode/sizing_example.json", 'r') as file:
        sizing = json.load(file)

    parameters = [ k for  k,v in sizing.items() ]
    values     = [ v for  k,v in sizing.items() ]


    folder = "./circuit_examples/ptm130_folded_cascode/"

    meas = ngspice.simulate(cwd = folder, netlist=["tb_ac.cir"], param = parameters, val = np.array([values]))


    errors = []

    # replace assertions by conditions
    if not len(meas) == 1:
        errors.append("Failed output size, found {} expected 1".format(len(meas)))
    else:
        if not len(meas[0]) == 17:
            errors.append("Failed number of coners , found {} expected 17".format(len(meas[0])))
    
        if meas[0].keys()  != corner_set : 
            errors.append("Failed to find all corners , found {} expected one of {}".format(sorted(meas.keys()), sorted(corner_set)))
    
    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def itest_folded_corners_opt():    
    seed = 42
    np.random.seed(seed)
    
    nsga2 = opt.NSGA2()

    for pop, pop_obj,  pop_cstr, pop_data, evals, front_no in nsga2.minimize(
        cir.Circuit("./circuit_examples/ptm130_folded_cascode/"), pop_size=32, evaluations=32*10, mutation=0.3):
        print(evals)
        print(pop_obj[pop_cstr.argmax()] , pop_cstr[pop_cstr.argmax()])
        print(pop_data[pop_cstr.argmax()][1]['TT'])
    


if __name__ == '__main__':
    itest_folded_corners_opt()
