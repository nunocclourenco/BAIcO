# BAIcO - An Open Source Benchmark for Analog IC Optimization

Optimization is the de facto approach for Analog IC Sizing automation. However, the lack of an open benchmark due to intellectual property barriers prevents the comparison between approaches. BAIcO, is an open framework for circuit sizing optimization and benchmarking based only on open source tools and open device models.  The framework implemented in python, and the circuit models are from the http://ptm.asu.edu/, generated for TT, SS, FF, SF, and FS corners, and the simulator is ngspice (http://ngspice.sourceforge.net/) 

## Device models
BAIcO considers devices models for 130nm and 65nm from (http://ptm.asu.edu/) with the corner options for FF, SS, SF, and FS.

## Circuit examples
- PTM 130nm
    - Folded Cascode Amplifier
    - Symmetric Amplifier
    - Folded Inverter Based OTA for biomedical applications [1]
    - Folded Voltage-Combiners Biased Amplifier [2]
    - Single-Stage OTA Biased by Voltage-Combiners With Enhanced Performance Using Current Starving [3]
    - Single-Stage OTA Biased by Voltage-Combiners [4] 
    - Two Stage Miller Amplifier
    - Two Stage Folded Cascode Miller Amplifier
- PTM 65nm
    - Folded Voltage-Combiners Biased Amplifier [2]
    - Ultra low Power OTA for biomedical applications [5]

### References

[1] R. Póvoa, A. Canelas, R. Martins, N. Horta, N. Lourenço, J. Goes,
A new family of CMOS inverter-based OTAs for biomedical and healthcare applications,
Integration,
Volume 71,
2020,
https://doi.org/10.1016/j.vlsi.2019.12.004.

[2] R. Póvoa, N. Lourenço, R. Martins, A. Canelas, N. Horta and J. Goes, "A Folded Voltage-Combiners Biased Amplifier for Low Voltage and High Energy-Efficiency Applications," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 67, no. 2, pp. 230-234, Feb. 2020, https://doi.org/10.1109/TCSII.2019.2913083.

[3] R. Póvoa, N. Lourenço, R. Martins, A. Canelas, N. Horta and J. Goes, "Single-Stage OTA Biased by Voltage-Combiners With Enhanced Performance Using Current Starving," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 65, no. 11, pp. 1599-1603, Nov. 2018, https://doi.org/10.1109/TCSII.2017.2777533.

[4] R. Póvoa, N. Lourenço, R. Martins, A. Canelas, N. C. G. Horta and J. Goes, "Single-Stage Amplifier Biased by Voltage Combiners With Gain and Energy-Efficiency Enhancement," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 65, no. 3, pp. 266-270, March 2018, https://doi.org/10.1109/TCSII.2017.2686586.

[5] R. Póvoa, R. Arya, A. Canelas, F. Passos, R. Martins, N. Lourenço, N. Horta,
Sub-µW Tow-Thomas based biquad filter with improved gain for biomedical applications,
Microelectronics Journal,
Volume 95,
2020,
https://doi.org/10.1016/j.mejo.2019.104675.


###### @2021 Instituto de Telecomunicações

