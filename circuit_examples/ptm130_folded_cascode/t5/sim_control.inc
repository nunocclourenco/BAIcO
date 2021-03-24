let cl = $&cl_sp
print cl
AC DEC 21 1 1G
meas AC GDC FIND vdb(vout) at=1
meas AC GBW WHEN vdb(vout)=0
meas AC PM_Negative FIND vp(vout) WHEN vdb(vout)=0
meas AC GPS FIND vdb(voutt) at=1
let PM = PM_Negative + 180
let PSRR = GDC-GPS
print PM PSRR
************************************************
* Run noise
NOISE V(vout, 0) v1 dec 21 0.01 10G
print inoise_total onoise_total
let count = length(noise1.frequency)
echo "NOISE output $&count 3"
print noise1.all
OP
let idd = mag(i(v0)) - $&ib_sp
let outswing = 1.65 - @m.xi10.mnm6[vdsat] - @m.xi10.mnm8[vdsat]
print idd outswing 
let vov_mpm0 = @m.xi10.mpm0[vgs] - @m.xi10.mpm0[vth] 
let vov_mpm1 = @m.xi10.mpm1[vgs] - @m.xi10.mpm1[vth]
let vov_mpm3 = @m.xi10.mpm3[vgs] - @m.xi10.mpm3[vth]
let vov_mpm4 = @m.xi10.mpm4[vgs] - @m.xi10.mpm4[vth]
let vov_mpm5 = @m.xi10.mpm5[vgs] - @m.xi10.mpm5[vth]
let vov_mpm6 = @m.xi10.mpm6[vgs] - @m.xi10.mpm6[vth]
let vov_mnm1 = @m.xi10.mnm1[vgs] - @m.xi10.mnm1[vth] 
let vov_mnm2 = @m.xi10.mnm2[vgs] - @m.xi10.mnm2[vth] 
let vov_mnm3 = @m.xi10.mnm3[vgs] - @m.xi10.mnm3[vth] 
let vov_mnm4 = @m.xi10.mnm4[vgs] - @m.xi10.mnm4[vth] 
let vov_mnm5 = @m.xi10.mnm5[vgs] - @m.xi10.mnm5[vth] 
let vov_mnm6 = @m.xi10.mnm6[vgs] - @m.xi10.mnm6[vth] 
let vov_mnm7 = @m.xi10.mnm7[vgs] - @m.xi10.mnm7[vth] 
let vov_mnm8 = @m.xi10.mnm8[vgs] - @m.xi10.mnm8[vth] 
print vov_mpm0 vov_mpm1 vov_mpm3 vov_mpm4 vov_mpm5 vov_mpm6 vov_mnm1 vov_mnm2 vov_mnm3 vov_mnm4 vov_mnm5 vov_mnm6 vov_mnm7 vov_mnm8
let delta_mpm0 = @m.xi10.mpm0[vds] - @m.xi10.mpm0[vdsat]
let delta_mpm1 = @m.xi10.mpm1[vds] - @m.xi10.mpm1[vdsat]
let delta_mpm3 = @m.xi10.mpm3[vds] - @m.xi10.mpm3[vdsat]
let delta_mpm4 = @m.xi10.mpm4[vds] - @m.xi10.mpm4[vdsat]
let delta_mpm5 = @m.xi10.mpm5[vds] - @m.xi10.mpm5[vdsat]
let delta_mpm6 = @m.xi10.mpm6[vds] - @m.xi10.mpm6[vdsat]
let delta_mnm1 = @m.xi10.mnm1[vds] - @m.xi10.mnm1[vdsat]
let delta_mnm2 = @m.xi10.mnm2[vds] - @m.xi10.mnm2[vdsat]
let delta_mnm3 = @m.xi10.mnm3[vds] - @m.xi10.mnm3[vdsat]
let delta_mnm4 = @m.xi10.mnm4[vds] - @m.xi10.mnm4[vdsat]
let delta_mnm5 = @m.xi10.mnm5[vds] - @m.xi10.mnm5[vdsat]
let delta_mnm6 = @m.xi10.mnm6[vds] - @m.xi10.mnm6[vdsat]
let delta_mnm7 = @m.xi10.mnm7[vds] - @m.xi10.mnm7[vdsat]
let delta_mnm8 = @m.xi10.mnm8[vds] - @m.xi10.mnm8[vdsat]
print delta_mpm0 delta_mpm1 delta_mpm3 delta_mpm4 delta_mpm5 delta_mpm6 delta_mnm1 delta_mnm2 delta_mnm3 delta_mnm4 delta_mnm5 delta_mnm6 delta_mnm7 delta_mnm8
