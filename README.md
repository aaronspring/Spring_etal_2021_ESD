# Spring et al. 2021

A manuscript will be soon submitted to Biogeosciences.

## Aim

This repo is setup for scientists interested to reproduce our `Spring et al., 2021` paper. It contains scripts to reproduce the analysis and create the shown figures. It is inspired by `Irving (2015)` to enhance reproducibility in geosciences.

-   Irving, Damien. “A Minimum Standard for Publishing Computational Results in the Weather and Climate Sciences.” Bulletin of the American Meteorological Society 97, no. 7 (October 7, 2015): 1149–58. <https://doi.org/10/gf4wzh>.

## Climate model setup

- Mauritsen, T., Bader, J., Becker, T., Behrens, J., Bittner, M., Brokopf, R., Brovkin, V., Claussen, M., Crueger, T., Esch, M., Fast, I., Fiedler, S., Fläschner, D., Gayler, V., Giorgetta, M., Goll, D. S., Haak, H., Hagemann, S., Hedemann, C., … Roeckner, E. (2019). Developments in the MPI‐M Earth System Model version 1.2 (MPI‐ESM1.2) and Its Response to Increasing CO 2. Journal of Advances in Modeling Earth Systems, 11(4), 998–1038. doi: 10/gftpps

See `model_setup`

## Packages used mostly

-   model output aggregation: `cdo`, `pymistral`
-   analysis: `xarray`
-   visualisation: `matplotlib`, `cartopy`, `seaborn`
-   predictive skill analysis: [`climpred`](https://climpred.readthedocs.io/)

## Computation

The results in this paper were obtained using a number of different software packages. The command line tool known as Climate Data Operators (CDO) was used to aggregate output and perform routine calculations on those files (e.g., the calculation of temporal and spatial means). For more complex analysis and visualization, a Python distribution called Anaconda was used. A Python library called `xarray` was used for reading/writing netCDF files and data analysis. `matplotlib` (the default Python plotting library) and `cartopy` were used to generate the maps. The high-dimensional global aggregation plots are done with `seaborn`.

-   CDO: Climate Data Operators, 2018. <http://www.mpimet.mpg.de/cdo>.
-   Hoyer, Stephan, and Joe Hamman. “Xarray: N-D Labeled Arrays and Datasets in Python.” Journal of Open Research Software 5, no. 1 (April 5, 2017). <https://doi.org/10/gdqdmw>.
-   Hunter, J. D. “Matplotlib: A 2D Graphics Environment.” Computing in Science Engineering 9, no. 3 (May 2007): 90–95. <https://doi.org/10/drbjhg>.
-   Waskom, M., Olga Botvinnik, Paul Hobson, John B. Cole, Yaroslav Halchenko, Stephan Hoyer, Alistair Miles, Tom Augspurger, Tal Yarkoni, Tobias Megies, Luis Pedro Coelho, Daniel Wehner, cynddl, Erik Ziegler, diego0020, Yury V. Zaytsev, Travis Hoppe, Skipper Seabold, Phillip Cloud, … Dan Allan. (2014). seaborn: v0.5.0 (November 2014). Zenodo. doi: 10.5281/zenodo.12710
-   Brady, R. X., & Spring, A. (2020). climpred: verification of weather and climate forecasts. Journal of Open Source Software.

## Environment

Dependencies (Packages installed) can be found in `requirements.txt` (conda list). Installed via conda (see setup `conda_info.txt`) and pip.
