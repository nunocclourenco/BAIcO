'''
    This method computes measures from the data. 
    While some could be computed using ngspice meas statments. We opted to save 
    the data and implement the measurments in Python as it more flexible and 
    better documented,easing the learning curve.
    These fucntions are located in a separate module for ease of custumization.
'''

def compute_additional_measures(sim_results):
    '''
    This is the entry point for the measument post processing.
    param: sim_results: a dictionary with the simulation outputs for all corners. 
    sim_results ={"TT": {"VOV_M1":100e-3, ... , "AC": [(freq, vdb, vp),...], "NOISE":[(freq, inoise_spectrum, onoise_spectrum)]}, ...}
    '''
    
    for corner, corner_sim in sim_results.items() : 
        compute_ac_meas(corner_sim)
        compute_noise_meas(corner_sim)
        compute_foms(corner_sim)
        


def compute_noise_meas(sim_results):
    '''
    Computes NOISE measures. SDINOISE and SDONOISE
    param: sim_results: a dictionary with the simulation outputs for one corner.
    
        sim_results = "... , "AC": [(freq, vdb, vp),...], "NOISE":[(freq, inoise_spectrum, onoise_spectrum)]}
    '''

    if "NOISE" in sim_results:
        if "GBW" in sim_results:
            sd_inoise = None
            sd_onoise = None    
            for row  in sim_results["NOISE"]:
                if row[0] < sim_results["GBW"]:
                    sd_inoise = row[1]
                    sd_onoise = row[2]
            if sd_inoise is not None: 
                sim_results["SDINOISE"] = sd_inoise
                sim_results["SDONOISE"] = sd_onoise
        #remove raw data
        del sim_results["NOISE"]

def compute_ac_meas(sim_results):
    '''
    Computes AC measures: DC gain (GAIN), -3dB cutoff frequency (FC), gain bandwith product (GBW), phase margin (PM)
    param: sim_results: a dictionary with the simulation outputs for one corner. 
    

    sim_results = "... , "AC": [(freq, vdb, vp),...], ...]}
    '''
    if "AC" not in sim_results: return

    gain = sim_results["AC"][0][1]
    inverted_out = True if abs(sim_results["AC"][0][2]) > 178 else False
    fc = None
    gbw = None
    pm = None

    for freq, vdb, vp in sim_results["AC"]:
        if vdb > gain - 3 : fc = freq
        if vdb > 0 : 
            gbw = freq
            pm = vp


    sim_results["GAIN"] = gain
    if fc is not None: 
        sim_results["FC"] = fc
    if gbw is not None:
        sim_results["GBW"] = gbw
        sim_results["PM"] = pm if inverted_out else pm + 180

    #remove raw data
    sim_results["AC"] = []


def compute_foms(sim_results):
    '''
    Computes FOMS is the respective measures are available
    param: sim_results: a dictionary with the simulation outputs for one corner.
    
        sim_results = "... , "AC": [(freq, vdb, vp),...], "NOISE":[(freq, inoise_spectrum, onoise_spectrum)]}
    '''

    if "GBW" in sim_results and "CL" in sim_results and "IDD" in sim_results:
        sim_results["FOM"] =  1000*(sim_results['GBW']*sim_results['CL'])/(sim_results['IDD'])






def _extended_meas(self, measures):
    """
    inputs raw measures from simulator and outputs only relevant measures
    """
    meas_out = measures
    meas_out['gdc'] = float(measures['gdc'])
    meas_out['gbw'] = float(measures['gbw']) if 'gbw' in measures else None
    meas_out['pm'] = float(measures['pm']) if 'pm' in measures else None
    meas_out['inoise_total'] = float(measures['inoise_total']) if 'inoise_total' in measures else None
    meas_out['onoise_total'] = float(measures['onoise_total']) if 'onoise_total' in measures else None

    meas_out['idd'] = ((-float(measures['vdc_i'])) - 0.0001) 

    meas_out["fom"] =  (((meas_out['gbw']/1000000)*6)/(meas_out['idd']*1000)) if meas_out['gbw'] != None else None

    meas_out["vov_mpm0"] = float(measures["m_xinova_mpm0_vgs"]) - float(measures["m_xinova_mpm0_vth"])
    meas_out["vov_mpm1"] = float(measures["m_xinova_mpm1_vgs"]) - float(measures["m_xinova_mpm1_vth"])
    meas_out["vov_mpm2"] = float(measures["m_xinova_mpm2_vgs"]) - float(measures["m_xinova_mpm2_vth"])
    meas_out["vov_mpm3"] = float(measures["m_xinova_mpm3_vgs"]) - float(measures["m_xinova_mpm3_vth"])
    meas_out["vov_mnm4"] = float(measures["m_xinova_mnm4_vgs"]) - float(measures["m_xinova_mnm4_vth"])
    meas_out["vov_mnm5"] = float(measures["m_xinova_mnm5_vgs"]) - float(measures["m_xinova_mnm5_vth"])
    meas_out["vov_mnm6"] = float(measures["m_xinova_mnm6_vgs"]) - float(measures["m_xinova_mnm6_vth"])
    meas_out["vov_mnm7"] = float(measures["m_xinova_mnm7_vgs"]) - float(measures["m_xinova_mnm7_vth"])
    meas_out["vov_mnm8"] = float(measures["m_xinova_mnm8_vgs"]) - float(measures["m_xinova_mnm8_vth"])
    meas_out["vov_mnm9"] = float(measures["m_xinova_mnm9_vgs"]) - float(measures["m_xinova_mnm9_vth"])
    meas_out["vov_mnm10"] = float(measures["m_xinova_mnm10_vgs"]) - float(measures["m_xinova_mnm10_vth"])
    meas_out["vov_mnm11"] = float(measures["m_xinova_mnm11_vgs"]) - float(measures["m_xinova_mnm11_vth"])

    meas_out["delta_mpm0"] = float(measures["m_xinova_mpm0_vds"]) - float(measures["m_xinova_mpm0_vdsat"])
    meas_out["delta_mpm1"] = float(measures["m_xinova_mpm1_vds"]) - float(measures["m_xinova_mpm1_vdsat"])
    meas_out["delta_mpm2"] = float(measures["m_xinova_mpm2_vds"]) - float(measures["m_xinova_mpm2_vdsat"])
    meas_out["delta_mpm3"] = float(measures["m_xinova_mpm3_vds"]) - float(measures["m_xinova_mpm3_vdsat"])
    meas_out["delta_mnm4"] = float(measures["m_xinova_mnm4_vds"]) - float(measures["m_xinova_mnm4_vdsat"])
    meas_out["delta_mnm5"] = float(measures["m_xinova_mnm5_vds"]) - float(measures["m_xinova_mnm5_vdsat"])
    meas_out["delta_mnm6"] = float(measures["m_xinova_mnm6_vds"]) - float(measures["m_xinova_mnm6_vdsat"])
    meas_out["delta_mnm7"] = float(measures["m_xinova_mnm7_vds"]) - float(measures["m_xinova_mnm7_vdsat"])
    meas_out["delta_mnm8"] = float(measures["m_xinova_mnm8_vds"]) - float(measures["m_xinova_mnm8_vdsat"])
    meas_out["delta_mnm9"] = float(measures["m_xinova_mnm9_vds"]) - float(measures["m_xinova_mnm9_vdsat"])
    meas_out["delta_mnm10"] = float(measures["m_xinova_mnm10_vds"]) - float(measures["m_xinova_mnm10_vdsat"])
    meas_out["delta_mnm11"] = float(measures["m_xinova_mnm11_vds"]) - float(measures["m_xinova_mnm11_vdsat"])

    return meas_out
 



