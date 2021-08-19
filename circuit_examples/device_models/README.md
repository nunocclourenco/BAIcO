# Generic_Lib and Mapping to PTM65, PTM130 and SKY130A


The device models folder consists of the model themselves for the supported technologies and the mapping files from the generic library from [1] to SKY130, PTM130, and PTM65.  This design strategy allows for a technology-independent netlist definition. 

The following table describes the device mapping.


<table border="1">
   <thead>
      <tr>
         <th>Generic Device</th>
         <th>Description</th>
         <th>Parameters</th>
         <th>Ports</th>
         <th>SKY130A</th>
         <th>PTM130</th>
         <th>PTM65</th>
      </tr>
   </thead>
   <tbody>
      <tr><td>pmos    </td><td>Standard Voltage PMOS transistor        </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__pfet_01v8    </td><td>TBD</td><td>TBD</td></tr>
      <tr><td>pmos_lvt</td><td>Standard Voltage LVT PMOS transistor    </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__pfet_01v8_lvt</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>pmos3v3</td><td>3.3V PMOS transistor             </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__pfet_g5v0d10v5</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>pmos5v0</td><td>5V PMOS transistor               </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__pfet_g5v0d10v5</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>nmos    </td><td>Standard Voltage NMOS transistor        </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__nfet_01v8    </td><td>TBD</td><td>TBD</td></tr>
      <tr><td>nmos_lvt</td><td>Standard Voltage LVT NMOS transistor    </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__nfet_01v8_lvt</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>nmos3v3</td><td>3.3V NMOS transistor             </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__nfet_g5v0d10v5</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>nmos5v0</td><td>5V NMOS transistor               </td><td>W, L, FINGERS</td><td>d, g, s, b</td><td>sky130_fd_pr__nfet_g5v0d10v5</td><td>TBD</td><td>TBD</td></tr>   
      <tr><td>mimcap </td><td>MimCap                           </td><td>W, L</td><td>c0, c1</td><td>sky130_fd_pr__cap_mim_m3_2</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>rpoly  </td><td>Poly resistor                     </td><td>W, L</td><td>r0, r1, b</td><td>sky130_fd_pr__res_high_po</td><td>TBD</td><td>TBD</td></tr>
      <tr><td>rpoly_high  </td><td>HR Poly resistor                     </td><td>W, L</td><td>r0, r1, b</td><td>sky130_fd_pr__res_xhigh_po</td><td>TBD</td><td>TBD</td></tr>
   </tbody>
</table>


# Parameter ranges
## SKY130
<table border="1">
   <thead>
      <tr>
         <th>Device</th>
         <th>W[uM] (min., max.)</th>
         <th>L[uM] (min., max.)</th>
         <th>Fingers (min., max.)</th>
      </tr>
   </thead>
   <tbody>
      <tr><td>pmos    </td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>pmos_lvt</td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>pmos3v3</td><td><i>(min, max)</i></td><td><i>(0.5, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>pmos5v0</td><td><i>(min, max)</i></td><td><i>(0.9, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>nmos    </td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>nmos_lvt</td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>nmos3v3</td><td><i>(min, max)</i></td><td><i>(1, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>nmos5v0</td><td><i>(min, max)</i></td><td><i>(1, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>mimcap </td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>rpoly  </td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
      <tr><td>rpoly_high  </td><td><i>(min, max)</i></td><td><i>(0.15, max)</i></td><td><i>(min, max)</i></td></tr>
   </tbody>
</table>

*Values to be confirmed in PDK [documents](https://skywater-pdk.readthedocs.io/en/latest/)* 

## Corners

**TBD**


### References

[1] J. Cachaço, N. Machado, N. Lourenço, J.G. Guilherme, N. Horta, Automatic Technology Migration of Analog IC Designs using Generic Cell Libraries, Design, Automation, and Test in Europe - DATE, Lausanne, Switzerland, Vol. N/A, pp. 1 - 4, March, 2017,