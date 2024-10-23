import xarray as xr
import numpy as np
import os

import dask
a = 't'
dask.config.set(scheduler='threads')
path = f'/sciserver/filedb04-03/ocean/wenrui_temp/divu{a}/'
os.listdir(path)


datasets_2 = sorted([path+i for i in os.listdir(path) if '.nc' in i])
print('before opening')
ugm = xr.open_mfdataset(datasets_2)
ugm[f'divum{a}m'] = ugm[f'divum{a}m'].isel(time = 0)
print('finished opening')

out = ugm.reset_coords()[[f'divu{a}',f'divum{a}m',f'divup{a}m',f'divum{a}p',f'divup{a}p']]

import zarr

# zarr.blosc.list_compressors()

compressor = zarr.Blosc(cname='zlib')
opts = {}
for varname in out.data_vars:
    out[varname] = out[varname].chunk((1,50,13,90,90))
    opts[varname] = {'compressor': compressor}

from dask.diagnostics import ProgressBar
output_path = f'/sciserver/filedb02-02/ocean/wenrui_temp/heat/divu{a}.zarr'
with ProgressBar():
    out.to_zarr(output_path, encoding = opts,mode = 'w')