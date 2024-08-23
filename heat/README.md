# Reproduce the heat example

The daily mean ECCO dataset is available via direct download (e.g. wget), or from `sciserver` and `oceanspy`. 

In a sciserver `Oceanography` environment, one can access ECCO by

```python
import oceanspy as ospy
ecco = ospy.open_oceandataset.from_catalog('daily_ecco')._ds
```

Change the variable names and path accordingly, and run the `eulerian` and `lagrangian` folder in file name order. 