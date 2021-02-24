import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import json

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.convolution import Tophat2DKernel
from regions import CircleSkyRegion, RectangleSkyRegion

from astropy.time import Time

from gammapy.data import DataStore
from gammapy.irf import PSFKernel
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.datasets import MapDataset, Datasets, FluxPointsDataset
from gammapy.makers import (
    MapDatasetMaker,
    SafeMaskMaker,
    FoVBackgroundMaker,
)
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    Models,
    SkyModel,
    BackgroundModel,
    PowerLawSpectralModel,
    PowerLaw2SpectralModel,
    PointSpatialModel,
    ExpCutoffPowerLawSpectralModel,
    TemplateSpatialModel,
    GaussianSpatialModel,
    FoVBackgroundModel
)
#from gammapy.stats import significance, excess # utiles ?
from gammapy.estimators import (
    #LiMaMapEstimator,
    TSMapEstimator,
    ExcessMapEstimator,
    FluxPointsEstimator
)

import gammapy

src_pos = SkyCoord(359.94, -0.04, unit="deg", frame="galactic")


from pathlib import Path

# Directory for outputs

chnl = input("which channel ? (fr/hd) ")
channel = "../../../hess_results/GC_variability_0.18.2/hap-"+chnl

tried = "1cutoff"

path = Path(channel)
path.mkdir(exist_ok=True)

pathma = Path(path/"mapdatasets")
pathma.mkdir(exist_ok=True)

pathmo = Path(path/"models")
pathmo.mkdir(exist_ok=True)

# for consistency we will use the template using exp cutoff for both the central source and the DE
# but it will generally require that the cutoff of the DE be frozen and set to infinity (lambda = 0)

#model_name = pathmo/"models_template_2cutoff.yaml" 

model_name = channel+"/3Dspectra/2amps_2indexes_"+tried+"/models_joint_fitted.yaml"

res = "simu_variable/2amps_2indexes_"+tried
pathres = Path(path/res)
pathres.mkdir(exist_ok=True)
pathres = Path(path/"year_per_year_sensitivity")
pathres.mkdir(exist_ok=True)
pathres = Path(path/"inprocess")
pathres.mkdir(exist_ok=True)

## Geometry

emin, emax = [0.5, 100] * u.TeV

e_bins = 20

energy_axis = MapAxis.from_energy_bounds(
    emin.value, emax.value, e_bins, unit="TeV"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    frame="galactic",
    proj="CAR",
    axes=[energy_axis],
)

geom2d = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    frame="galactic",
    proj="CAR",
)

emintrue, emaxtrue = [0.3,200] * u.TeV
e_bins_true = 30

energy_axis_true = MapAxis.from_energy_bounds(
    emintrue.value, emaxtrue.value, e_bins_true, unit="TeV", name="energy_true"
)

## Importing models
modelGC,modelG09, modeldiff= Models.read(model_name)

# Setting which parameters are free, and their "baseline" value

modelGC.parameters["amplitude"].frozen = False
modelGC.parameters["amplitude"].value = 2.14e-12 #2.12

modelGC.parameters["index"].frozen = True
modelGC.parameters["index"].value = 1.835

modelGC.spectral_model.parameters['lambda_'].frozen = True
modelGC.spectral_model.parameters['lambda_'].value = 1/6.381

modeldiff.parameters["amplitude"].frozen = False
modeldiff.parameters["amplitude"].value = 5.83e-12

modeldiff.parameters["index"].frozen = True
modeldiff.parameters["index"].value = 2.243

modeldiff.spectral_model.parameters['lambda_'].frozen = True
modeldiff.spectral_model.parameters['lambda_'].value = 0.0

# models to initialize the fit after data has been replaced
modelGCnaive, modeldiffnaive = modelGC.copy(), modeldiff.copy()

modelGCnaive.parameters["amplitude"].value = 3e-12
modeldiffnaive.parameters["amplitude"].value = 5e-12

## The Mask
fit_region = RectangleSkyRegion(src_pos, 4*u.deg, 3*u.deg)

J1745_303_region = CircleSkyRegion(SkyCoord(358.6,  -0.6, unit="deg", frame="galactic"), 0.75 * u.deg)

mask_fit = geom.region_mask([fit_region])*geom.region_mask([J1745_303_region] , inside=False)

mask_fit = Map.from_geom(geom, data=mask_fit)

def fit_dataset(mapdataset):
    
    fit = Fit([mapdataset])
    result = fit.run()
    
    table = result.parameters.to_table()
    
    return table, result


emin = 1.0*u.TeV
emax = 10*u.TeV

#GCflux_distribution = []
#DEflux_distribution = []


# à ne définir qu'une fois, puis on rajoute des éléments 
#en prenant soin de laisser k augmenter sans remise à zéro dans les appels successifs de la cellule suivante

year = int(input("which year : "))
iterations = int(input('how many iterations : ')) #should be 50, it breaks around 47 usually, might depend on the year
adding = input("are we adding simulated runs ? (y/n) ")


if adding=='y' or adding=="yes":
    list = []
    with open(pathres/f'GC_fluxes_simu_{year}.txt', 'r') as GCFile:
        for line in GCFile:
            x = line[:-1]
            list.append(x)
            n = len(list)
            k = n
            iterations = k + iterations
else:
    k = 0


with open(pathres/f'GC_fluxes_simu_{year}.txt', 'a+') as GCfile:
    GCfile.write("")
with open(pathres/f'DE_fluxes_simu_{year}.txt', 'a+') as DEfile:
    DEfile.write("")
        
while k <  iterations :
            #ça fonctionne comme ça, tous les modèles existent indépendamment des datasets donc on peut continuer à 
    dataset = MapDataset.read(pathma/f"map_{year}.fits")

    dataset.models =  [modelGC.copy(),modelG09.copy(),modeldiff.copy()]     
    dataset.fake(k) 

            # on remet les modèles
    dataset.mask_fit = mask_fit
    bkg_model = FoVBackgroundModel(dataset_name=f"map_{year}")

    dataset.models =  [modelGCnaive.copy(), modelG09.copy(), modeldiffnaive.copy(),bkg_model]

                # on fait le fit
    table, result = fit_dataset(dataset)

    diffuse_flux = dataset.models[2].spectral_model.integral(emin, emax)
    GC_flux = dataset.models[0].spectral_model.integral(emin, emax)

            #GCflux_distribution.append(GC_flux.value)
            #DEflux_distribution.append(diffuse_flux.value)
    with open(pathres/f'GC_fluxes_simu_{year}.txt', 'a+') as GCfile:
        GCfile.write('%s\n' % GC_flux.value)
    with open(pathres/f'DE_fluxes_simu_{year}.txt', 'a+') as DEfile:
        DEfile.write('%s\n' % diffuse_flux.value)

    k = k+1
    


