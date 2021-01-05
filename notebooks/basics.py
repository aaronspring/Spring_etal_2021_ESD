import xarray as xr
import numpy as np
import dask
import cdo
cdo = cdo.Cdo(tempdir="/work/mh0727/m300524/tmp/.")
import os
import pymistral
import seaborn as sns
from tqdm.notebook import tqdm

from pymistral.cdo_post import load_var_cdo

output_df = pymistral.cdo_post.output_df

output_df2 = output_df.replace("tracer", "tracer_mm").replace(
    "echam6_tracer", "echam6_tracer_mm"
)

var_area = output_df.units.str.contains("m-2")
var_area = list(var_area[var_area == True].index)


# 
direct_run = 'ATMTSIDICALKlandr' #not renamed
indirect_run = 'ATMTSI'
c = "vga0220a_Rerun"
run_dict = {'perfect':c, 'indirect':indirect_run, 'direct':direct_run}


# zarr locations
outpath = "/work/bm1124/m300524/src/mpiesm-1.2.01-release/experiments/"
exppath = outpath[:-1]
workpath = f"/work/bm1124/m300524/experiments/"
postpath = "/work/mh0727/m300524/191008_PM_assim/post"
nudgingdatapath = f"{workpath}{c}/nudging_data"

ocean_zarr = f"{postpath}/ocean.zarr"
atm_zarr = f"{postpath}/atm.zarr"

pocean_zarr = f"{postpath}/pred_ocean.zarr"
patm_zarr = f"{postpath}/pred_atm.zarr"



# metadata
def latexify_units(da):
    if 'units' in da.attrs:
        da.attrs['units'] = da.attrs['units'].replace('-1','$^{-1}$')
    return da

showname_dict=dict()
showname_dict['spco2'] = 'pCO$_{2,ocean}$' 
showname_dict['fgco2'] = 'air-sea CO$_2$ flux'
showname_dict['co2_flx_ocean'] = 'air-sea CO$_2$ flux'
showname_dict['lai'] = 'LAI'
showname_dict['ws'] = 'soil wetness'
showname_dict['box_Cpools_total'] = 'total land carbon' 
showname_dict['cVeg'] = 'vegetation carbon pool' 
showname_dict['co2_flx_land'] = 'air-land CO$_2$ flux'
showname_dict['CO2'] = '$X$CO$_{2,atm}$'

def add_showname(da):
    da.attrs['showname']=showname_dict[da.name]
    return da

metrics = ["pearson_r", "rmse",'bias','mae']
metric_longname = dict()
metric_longname["pearson_r"] = "Anomaly Correlation Coefficient"
metric_longname["rmse"] = "Root mean square error"
metric_longname["bias"] = "Bias"
metric_longname["mae"] = "Mean Absolute Error"
metric_shortname = dict()
metric_shortname["pearson_r"] = "ACC"
metric_shortname["rmse"] = "RMSE"
metric_shortname["bias"] = "Bias"
metric_shortname["mae"] = "MAE"


r_ppmw2ppmv = 28.8 / 44.0095
CO2_to_C = 44.0095 / 12.0111
def convert_C(ds):
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
        was_da = True
    else:
        was_da = False
    if isinstance(ds, xr.Dataset):
        if "CO2" in ds.data_vars:
            ds["CO2"] = ds["CO2"] * 1e6 * r_ppmw2ppmv
            ds["CO2"].attrs["long_name"] = "mixing ratio of CO$_2$ in air"
            ds["CO2"].attrs["units"] = "ppm"
            print("converted CO2 from kg kg-1 to ppm")
        for v in ["spco2", "dpco2", "po2"]:
            if v in ds.data_vars:
                if ds[v].attrs["units"] == "Pa":
                    ds[v] *= 10
                    ds[v].attrs["units"] = "ppm"
                    #dsz[v].attrs["long_name"] = "surface ocean pCO$_2$"
                    #dsz[v].attrs["name"] = "surface ocean pCO$_2$"
                    print(f"converted {v} from Pa to ppm")
        for v in ["dissicos", "talkos"]:
            if v in ds.data_vars:
                if ds[v].attrs["units"] != "mmol m-3":
                    ds[v] *= 1000
                    ds[v].attrs["units"] = "mmol m-3"
        for data_var in ds.data_vars:
            if "co2_fl" in data_var:
                ds[data_var] = ds[data_var] / CO2_to_C
                ds[data_var].attrs["long_name"] = (
                    ds[data_var].attrs["long_name"].replace("CO2", "C")
                )
                ds[data_var].attrs["units"] = (
                    ds[data_var].attrs["units"].replace("kg", "kgC")
                )
                print(f"converted {data_var} from CO2 units to C units.")
        if 'intpp' in ds.data_vars and 'mol' in ds['intpp'].attrs['units']:
            ds['intpp'] *= 12/1000
            ds['intpp'].attrs['units'] = ds['intpp'].attrs['units'].replace('mol','kg')
        if 'lai' in ds.data_vars:
            ds['lai'].attrs['units']=''
        if 'cVeg' in ds.data_vars:
            ds['cVeg'].attrs['units']='kgC m-2'
        if 'box_Cpools_total' in ds.data_vars:
            if 'mol(CO2)' in ds['box_Cpools_total'].attrs['units']:
                ds['box_Cpools_total'] = ds['box_Cpools_total']*12/1000
                ds['box_Cpools_total'].attrs['units'] = ds['box_Cpools_total'].attrs['units'].replace('mol(CO2)','kgC').replace('(grid box)','').strip(' ')
                print(f'converted box_Cpools_total from mol(CO2) to kgC')

    if was_da:
        VAR = list(ds.data_vars)[0]
        ds = ds[VAR]
    return ds

def kgC_s_to_PgC_month(ds):
    units = ds.attrs["units"]
    if "kg" in units and "s-1" in units:
        ds = ds * 1 * 10 ** -12 * (60 * 60 * 24 * 30)
        ds.attrs["units"] = "PgC mon-1"
        print(f"converted {ds.attrs['long_name']}")
    elif 'kgC' in units or 'kg' in units:
        ds = ds * 1e-12
        ds.attrs["units"] = ds.attrs["units"].replace('kg','Pg').strip(' ')
    return ds

long_name_rename_dict = {'intpp': 'integrated net primary productivity',
                         'fgco2': 'surface CO$_2$ flux',
                         'spco2': 'surface partial pressure of CO$_2$',
                         'dpco2': 'surface CO$_2$ partial pressure sea-air difference',
                         'box_Cpools_total': 'total terrestrial carbon pools'
                        }



##### variables postprocessed
atmvarlist = [
    "wind10",
    "u10",
    "v10",
    "temp2",
    "CO2",
    "co2_flux",
    "co2_flx_ocean",
    "co2_flx_land",
    "precip",
    "co2_flx_npp",
    "co2_flx_resp",
    "tsw",
    "slp",
    "ws",
    'box_Cpools_total'
]

oceanvarlist = [
    "zmld",
    "fgco2",
    "intpp",
    "spco2",
    "dpco2",
    #"dissicos",
    #"talkos",
    #"no3os",
    #"po4os",
    #"chlos",
    #"dfeos",
    #"o2os",
    "tos",
    "sos",
    #"siconc",
]

vars_to_cmorize = [
    "cVeg",
    "cLand",
    "netAtmosLandCO2Flux",
    "nep",
    "nbp",
    'npp',
    "gpp",
    "rh",
    "cSoil",
    "fco2nat",
    "cLitter",
]

def _maybe_auto_chunk(ds, dims):
    """Auto-chunk on dimension `dims`.

    Args:
        ds (xr.object): input data.
        dims (list of str or str): Dimensions to auto-chunk in.

    Returns:
        xr.object: auto-chunked along `dims`

    """
    if dask.is_dask_collection(ds) and dims is not []:
        if isinstance(dims, str):
            dims = [dims]
        chunks = [d for d in dims if d in ds.dims]
        chunks = {key: "auto" for key in chunks}
        ds = ds.chunk(chunks)
    return ds

def to_zarr(varlist, name, process_var):
    zarrpath = f"{postpath}/{name}.zarr"
    t = tqdm(varlist, desc="Var:")
    for i, v in enumerate(t):
        t.set_description(f"var: {v}")
        if os.path.exists(zarrpath):
            zarr_before = xr.open_zarr(zarrpath)
            if v in zarr_before.data_vars:
                print(f"variable {v} already present")
                continue
        try:
            # assert v in output_df2.index
            ds = process_var(v=v).to_dataset(name=v)
            ds = clean(ds)
        except Exception as e:
            print(v, "failed", e)
        if "run" in ds.dims:
            ds = ds.chunk({"run": 1, "time": -1})
            ds = _maybe_auto_chunk(ds, ["lon", "lat"])
        elif "init_type" in ds.dims:
            if 'x' not in list(ds.dims):
                ds = ds.chunk(
                    {
                        "init_type": 1,
                        "lead": -1,
                        "init": -1,
                        "member": -1,
                        "lon": -1,
                        "lat": -1,
                    }
                )
            else:
                ds = ds.chunk(
                    {
                        "init_type": 1,
                        "lead": -1,
                        "init": -1,
                        "member": -1,
                        "x": "auto",
                        "y": -1,
                    }
                )

        if os.path.exists(zarrpath):
            if "run" in ds.dims:
                assert ds.run.size == zarr_before.run.size
            elif "init_type" in ds.dims:
                for d in ["init_type", "init", "member", "lead"]:
                    assert ds[d].size == zarr_before[d].size
            print(f"append to {v} to {zarrpath}")
            ds.to_zarr(zarrpath, mode="a", consolidated=False)
        else:
            print(f"write {v} to {zarrpath}")
            ds.to_zarr(zarrpath, mode="w", consolidated=False)
            
def add_varlist_of_runs_to_zarr(zarr, varlist, process_var=None):
    ds = xr.Dataset()
    t = tqdm(varlist, desc="Var:")
    for v in t:
        t.set_description(f"var: {v}")
        ds[v] = process_var(v=v)
    if "run" in ds.dims:
            ds = ds.chunk({"run": 1, "time": -1})
            ds = _maybe_auto_chunk(ds, ["lon", "lat"])
    elif "init_type" in ds.dims:
        if output_df.loc[v]["model"] in ["echam6", "jsbach"]:
            ds = ds.chunk(
                {
                    "init_type": 1,
                    "lead": -1,
                    "init": -1,
                    "member": -1,
                    "lon": -1,
                    "lat": -1,
                }
            )
        else:
            ds = ds.chunk(
                {
                    "init_type": 1,
                    "lead": -1,
                    "init": -1,
                    "member": -1,
                    "x": "auto",
                    "y": -1,
                }
            )
    ds = clean(ds)
    ds.to_zarr(zarr, consolidated=False, mode="a", append_dim="run")

###### area files #####
ocean_area = xr.open_dataset("/work/mh0727/m300524/masks/GR15L40_fx.nc")[
    "area"
].squeeze()
del ocean_area["time"]
del ocean_area["depth"]
# ocean_area.plot(yincrease=False, robust=True)

land_area = cdo.gridarea(
    input="/work/bm1124/m300524/experiments/sample_files/echam6_BOT_mm.nc",
    returnXDataset=True,
)["cell_area"]
# land_area.plot()

def global_agg(da, how=None, area=ocean_area):
    spatial_dims = [d for d in da.dims if d not in ["run", "time", 'lead','init_type','init','member']]
    if not how:
        if 'units' in da.attrs:
            if "m-2" in da.attrs["units"] or "m-3" in da.attrs["units"]:
                how = "sum"
            else:
                how = "mean"
        else:
            how = "mean"

    if how == "mean":
        da = (da * area).sum(spatial_dims) / area.sum(spatial_dims)
    elif how == "sum":
        da = (da * area).sum(spatial_dims)
        if 'units' in da.attrs:
            da.attrs["units"] = da.attrs["units"].replace("m-2", "").replace("m-3", "m-1")
    return da

### misc
def yearmean(ds, dim="time"):
    if dim=='lead':
        ds['lead']=xr.cftime_range(start='2000',freq='MS',periods=ds.lead.size)
    ds = ds.groupby(f"{dim}.year").mean(dim).rename({"year": dim})
    if dim=='lead':
        ds['lead']=np.arange(1,1+ds.lead.size)
    if 'units' in ds.attrs:
        if 'month' in ds.attrs['units']:
            ds.attrs['units'] = ds.attrs['units'].replace('month','year')
            ds=ds*12
    return ds

def clean(dsz):
    for co in ["depth", "lev", 'height' ,'member', 'plev']:
        if co in dsz.coords and co not in dsz.dims:
            del dsz[co]
    return dsz

def reset_attrs(ds_new, ds_old):
    for v in ds_new.data_vars:
        ds_new[v].attrs = ds_old[v].attrs
    return ds_new


### plotting
def add_figurelabels(ax,labels='abcdefghijklmnopqrstuvwxyz',shift=0,labelsize=None,frameon=False, loc=2, pad=.25, borderpad=.1,**kwargs):
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    for i,axes in enumerate(ax.flatten()):
        axes.add_artist(AnchoredText(f'{labels[i+shift]})', prop=dict(size=labelsize),
                                     frameon=frameon, loc=loc, pad=pad, borderpad=borderpad))
    return ax

def annotate_figurelabels(
    ax,
    labels="abcdefghijklmnopqrstuvwxyz",
    shift=0,
    xy=(0.05, 0.9),
    xycoords="axes fraction",
    **kwargs,
):
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

    for i, axes in enumerate(ax.flatten()):
        axes.annotate(
            s=f"{labels[i+shift]})", xy=xy, xycoords=xycoords, zorder=101, **kwargs
        )
    return ax


c_dict=dict()
c_dict['perfect']='gray'
c_dict['indirect'],c_dict['direct']=sns.color_palette('Set2',2)