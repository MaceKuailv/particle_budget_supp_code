import xarray as xr
import numpy as np
import os

import dask
dask.config.set(scheduler='threads')
path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'
os.listdir(path)

datasets_2 = sorted([path+'wall_t/'+i for i in os.listdir(path+'wall_t/') if '.nc' in i])
print('before opening')
ugm = xr.open_mfdataset(datasets_2)
print('finished opening')

out = ugm.reset_coords()[['tx','ty','tz']]

import zarr

# zarr.blosc.list_compressors()

compressor = zarr.Blosc(cname='zlib')
opts = {}
for varname in out.data_vars:
    out[varname] = out[varname].chunk((1,50,13,90,90))
    opts[varname] = {'compressor': compressor}

from dask.diagnostics import ProgressBar
output_path = '/sciserver/filedb09-01/ocean/wall_theta.zarr'
with ProgressBar():
    out.to_zarr(output_path, encoding = opts,mode = 'w')