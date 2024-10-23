import seaduck as sd
import xarray as xr
import matplotlib.pyplot as plt
import numpy  as np
# import tys
import os
import dask
import warnings
warnings.filterwarnings("ignore")
dask.config.set(scheduler='threads')
os.listdir('/sciserver/filedb01-02')
os.listdir('/sciserver/filedb02-02')
os.listdir('/sciserver/filedb03-02')

path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'
path_to_output = '/sciserver/filedb04-03/ocean/wenrui_temp/'

# arg = tys.argv[-1]

mean = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/mean*',engine = 'zarr')
snap = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/snap*',engine = 'zarr')
grid = xr.open_zarr('~/ECCO_transport')
bc = xr.open_zarr('/sciserver/filedb09-01/ocean/GM_vel.zarr') #bolus correct
ds = xr.merge([mean, snap])
ds = ds.reset_coords().assign_coords(grid.coords).astype(float)


ut_mean = xr.open_zarr(path+'/utrans_m.zarr')
# ws_mean = xr.open_zarr(path+'/walls_m.zarr')
wt_mean = xr.open_zarr(path+'/wallt_m.zarr')

# ws = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_salt.zarr')
wt = xr.open_zarr('/sciserver/filedb09-01/ocean/wall_theta.zarr')

ds['utrans'] = ((ds['UVELMASS']+bc['u_gm'])*ds.drF*ds.dyG).transpose('time','Z','face','Y','Xp1')
ds['vtrans'] = ((ds['VVELMASS']+bc['v_gm'])*ds.drF*ds.dxG).transpose('time','Z','face','Yp1','X')
ds['wtrans'] = ((ds['WVELMASS']+bc['w_gm'])*ds.rA).transpose('time','Zl','face','Y','X')

from seaduck.eulerian_budget import *
xgcmgrd = create_ecco_grid(ds)
tub = sd.OceData(ds)

# Constants
R = 0.62
zeta1 = 0.6
zeta2 = 20.0
rhoconst = 1029
c_p = 3994

Depth = ds.Depth
dxG = ds.dxG
dyG = ds.dyG
drF = ds.drF
rA = ds.rA
hFacC = ds.HFacC.load()
vol = (rA*drF*hFacC).transpose('face','Z','Y','X')

xfluxname = 'ADVx'
yfluxname = 'ADVy'
zfluxname = 'ADVz'

# import zarr
    
# # zarr.blosc.list_compressors()

# compressor = zarr.Blosc(cname='zlib')
# opts = {}
# for arg in ['divus','divupsp','divupsm','divumsp','divumsm']:
#     opts[arg] = {'compressor': compressor}

for it in range(9497):
    sl = ds.isel(time = slice(it,it+1))
    wsl = wt.isel(time = slice(it,it+1))
    strtime = str(ds['time'][it].values)[:10]
    if it % 30 ==0:
        print(strtime)
    tub._ds = sl
    for arg in ['divut','divuptp','divuptm','divumtp','divumtm']:
        use_most_common_chunking = True
        
        if arg == 'divut':
            sl['ADVx'] = sl['utrans']*wsl['tx']
            sl['ADVy'] = sl['vtrans']*wsl['ty']
            sl['ADVz'] = sl['wtrans']*wsl['tz']
        if arg == 'divuptp':
            sl['ADVx'] = (sl['utrans']-ut_mean['utrans_m'])*(wsl['tx'] - wt_mean['txm'])
            sl['ADVy'] = (sl['vtrans']-ut_mean['vtrans_m'])*(wsl['ty'] - wt_mean['tym'])
            sl['ADVz'] = (sl['wtrans']-ut_mean['wtrans_m'])*(wsl['tz'] - wt_mean['tzm'])
        if arg == 'divumtp':
            sl['ADVx'] = (ut_mean['utrans_m'])*(wsl['tx'] - wt_mean['txm'])
            sl['ADVy'] = (ut_mean['vtrans_m'])*(wsl['ty'] - wt_mean['tym'])
            sl['ADVz'] = (ut_mean['wtrans_m'])*(wsl['tz'] - wt_mean['tzm'])
        if arg == 'divuptm':
            sl['ADVx'] = (sl['utrans']-ut_mean['utrans_m'])*(wt_mean['txm'])
            sl['ADVy'] = (sl['vtrans']-ut_mean['vtrans_m'])*(wt_mean['tym'])
            sl['ADVz'] = (sl['wtrans']-ut_mean['wtrans_m'])*(wt_mean['tzm'])
        if arg == 'divumtm':
            sl['ADVx'] = (ut_mean['utrans_m'])*(wt_mean['txm'])
            sl['ADVy'] = (ut_mean['vtrans_m'])*(wt_mean['tym'])
            sl['ADVz'] = (ut_mean['wtrans_m'])*(wt_mean['tzm'])
            # use_most_common_chunking = False
        sl[arg] = total_div(tub, xgcmgrd, xfluxname, yfluxname,zfluxname)
    output_list = ['divut','divuptp','divuptm','divumtp','divumtm']
    out = sl[output_list]

    for varname in out.data_vars:
        if varname !='divumtm':
            out[varname] = out[varname].transpose('time','Z','face','Y','X').chunk((1,50,13,90,90))
        else:
            out[varname] = out[varname].transpose('Z','face','Y','X').chunk((50,13,90,90))
    
    out.to_netcdf(path_to_output+'divut/'+strtime+'.nc',mode = 'w')