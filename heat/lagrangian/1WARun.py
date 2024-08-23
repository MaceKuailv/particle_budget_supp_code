import numpy as np
from open4dense import give_me_orig_ecco
# import oceanspy as ospy
import seaduck as sd
import xarray as xr
from seaduck.get_masks import which_not_stuck
import os 
# import sys
# int_arg = int(sys.argv[-1])
# print('got sys parameter:',int_arg)
int_arg = 0

filedb_lst = []
for i in range(1,13):
    for j in range(1,4):
        filedb_lst.append(f'/sciserver/filedb{i:02}-0{j}')

seed = 2011

save_path = filedb_lst[9+int_arg]+'/ocean/wenrui_temp/particle_file/WA/nc_new/'
path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'

ds = give_me_orig_ecco()
ds['utrans'] = (ds['u_gm']+ds['UVELMASS'])*ds.dyG*ds.drF
ds['vtrans'] = (ds['v_gm']+ds['VVELMASS'])*ds.dxG*ds.drF
ds['wtrans'] = (ds['w_gm']+ds['WVELMASS'])*ds.rA
tseas1 = xr.open_zarr(path+'tseas1.zarr')
tseas2 = xr.open_zarr(path+'tseas2.zarr')
tseas3 = xr.open_zarr(path+'tseas3.zarr')
tseas = xr.concat([tseas1,tseas2,tseas3],dim = 'dayofyear')
ta = (ds.THETA.groupby('time.dayofyear') - tseas).transpose('time','Z','face','Y','X').THETA

tub = sd.OceData(ds)

time = '2011-03-01'
t = sd.utils.convert_time(time)
end_time = t-365*86400*1.1
stops = np.array([end_time])

ylim = (-35,-22.5)
xlim = (100,120)

xbool = np.logical_and(ds.XC>xlim[0],ds.XC<xlim[1])
ybool = np.logical_and(ds.YC>ylim[0],ds.YC<ylim[1])
zbool = ds.Z>-50
pos_bool = np.logical_and(np.logical_and(xbool,ybool),zbool)
warm_bool = ta.sel(time = time)[0]>1.5
those = np.logical_and(warm_bool,pos_bool)

p = sd.Particle(
    bool_array=those, num=10000, random_seed=seed,
    t = t,
    data = tub, free_surface = 'noflux',
    save_raw = True,
    uname = 'utrans',vname = 'vtrans',wname = 'wtrans',
    transport  = True
)
p=p.subset(sd.get_masks.which_not_stuck(p))
# Np = p.N
# bins = Np//20+1
# slc = slice(int(sys.argv[-1])*bins,(int_arg+1)*bins)
# p = p.subset(slc)
p.empty_lists()

print('finished pre-calculating',p.N)

p.to_list_of_time(normal_stops = stops,dump_filename = save_path+f'Seed{seed}_',store_kwarg = {'preserve_checks':True})
print('success', p.N)