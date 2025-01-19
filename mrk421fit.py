import math
import os
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import pyvo
from agnpy import BrokenPowerLaw, add_systematic_errors_gammapy_flux_points
from agnpy.fit.models import SynchrotronSelfComptonModel
from agnpy.utils.plot import sed_y_label, load_mpl_rc
from astropy.constants import m_e, c
from astropy.table import Table
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel

# set the day (see Table5 for available days)
MJD = 57757

# Load the params from Table 5, see https://www.aanda.org/articles/aa/full_html/2021/11/aa41004-21/T5.html
df = pd.read_fwf("Table5", header=None, widths=[10, 7, 7, 7, 7]).set_index(0).T
df.columns.name = None
df.set_index("MJD", inplace=True)
df.index = df.index.astype(int)
params = df.loc[MJD]
print("Params for the MJD%s:" % MJD)
for key, value in params.items():
    print(f"{key:<12}{value:>4}")

# Download the SED file
downloadDir = "download"
file = "%s/result-mrk421-MJD%s.csv" % (downloadDir, MJD)
if not os.path.exists("%s" % downloadDir):
    os.makedirs("%s" % downloadDir)
if not os.path.exists(file):
    service = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
    query = 'SELECT * from "J/A+A/655/A89/mjd%s"' % MJD

    result = service.search(query)
    # see https://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/A%2bA/655/A89/mjd57757 for the received format
    result.to_table().write(file, format="csv", overwrite=True)

# load table from file
astropyTableVizier = Table.read(file, format="csv")
astropyTableVizier["Freq"].unit = u.Hz
astropyTableVizier["BFreq"].unit = u.Hz
astropyTableVizier["bFreq"].unit = u.Hz
astropyTableVizier["nuFnu"].unit = u.mW / u.m**2
astropyTableVizier["e_nuFnu"].unit = u.mW / u.m**2
astropyTableVizier["E_nuFnu"].unit = u.mW / u.m**2

print()
print("Input VizieR table:")
astropyTableVizier.pprint_all()

# gammapy expects different column names, so convert it to the desired format, see https://gamma-astro-data-formats.readthedocs.io/en/v0.3/spectra/flux_points/index.html
astropyTable = Table()
astropyTable["e_ref"] = astropyTableVizier["Freq"].to(u.MeV, equivalencies=u.spectral())
astropyTable['e2dnde'] = astropyTableVizier["nuFnu"].to(u.MeV/(u.cm**2 * u.s))
astropyTable['e2dnde_errn'] = astropyTableVizier["e_nuFnu"].to(u.MeV/(u.cm**2 * u.s))
astropyTable['e2dnde_errp'] = astropyTableVizier["E_nuFnu"].to(u.MeV/(u.cm**2 * u.s))
astropyTable['is_ul'] = (astropyTableVizier["l_nuFnu"] == '<')
astropyTable['instrument'] = astropyTableVizier["Inst"]

print()
print("Astropy table prepared for gammapy:")
astropyTable.pprint_all()

n_e = BrokenPowerLaw.from_total_energy_density(
    params.U_e_pr * 1e-2 * u.erg * u.cm ** -3,
# n_e = BrokenPowerLaw(
#     k=10**-7.5623 * u.Unit("cm-3"), //use this version instead if you know the log_10 k value
    mass=m_e,
    p1=params.a1,
    p2=params.a2,
    gamma_min=params.y_pr_min ** 1e3,
    gamma_b=params.y_pr_br * 1e5,
    gamma_max=params.y_pr_max * 1e6
)

Rb = params.R_pr * 1e16 * u.cm
delta_D = params.Gamma_b
redshift = 0.031

ssc_model = SynchrotronSelfComptonModel(n_e, backend="gammapy")
ssc_model.parameters["delta_D"].value = delta_D
ssc_model.parameters["log10_B"].value = math.log10(params.B_pr * 1e-2)
ssc_model.parameters["z"].value = redshift
ssc_model.parameters["t_var"].value = (Rb * (1+redshift) / (c*delta_D)).to_value("s")
ssc_model.parameters["t_var"].frozen = True

print()
print("Spectral model:")
print(ssc_model.parameters.to_table())
print()

E_min_for_fit = 1e-2 * u.eV

gammapyDatasets = Datasets()
instruments = astropyTable.to_pandas()["instrument"].unique()
for instrument in instruments:
    instrumentRows = astropyTable[astropyTable["instrument"] == instrument]
    data = FluxPoints.from_table(instrumentRows, sed_type="e2dnde", reference_model=ssc_model)
    syst_error = 0.3 if instrument == "MAGIC" else 0.1 if instrument == "Fermi-LAT" or instrument == "Swift/XRT" else 0.05
    add_systematic_errors_gammapy_flux_points(data, syst_error)
    dataset = FluxPointsDataset(data=data, name=instrument)
    # set the mask with minimum energy to be used for the fit
    dataset.mask_fit = (dataset.data.energy_ref >= E_min_for_fit)
    gammapyDatasets.append(dataset)

load_mpl_rc() # custom styling for the plots, to make them bigger

ig, ax = plt.subplots(figsize=(8, 6))
ax.xaxis.units = u.eV # without explicit unit definition, it may pick up MeV if rendered in wrong order
x_axis_bounds_eV = [1e-6, 1e14]
ssc_model.plot(
    ax=ax,
    energy_bounds=x_axis_bounds_eV * u.eV,
    sed_type="e2dnde",
    label="SSC model",
    color="k",
    lw=1.6,
)
for dataset in gammapyDatasets:
    dataset.data.plot(
        ax=ax,
        label=dataset.name)

sky_model = SkyModel(spectral_model=ssc_model, name="Mrk421") #GB to chyba niepotrzebne bo FluxMaps.fromMap tworzy swój SkyModel
gammapyDatasets.models = [sky_model]

fitter = Fit()
results = fitter.run(gammapyDatasets)

print(results)

if results.success:
    print(ssc_model.parameters.to_table())

    ssc_model.plot(
        ax=ax,
        energy_bounds=x_axis_bounds_eV * u.eV,
        sed_type="e2dnde",
        label="fit model",
        color="r",
        lw=1,
        ls="--"
    )

    # plot a line marking the minimum energy considered in the fit
    ax.axvline(E_min_for_fit, ls="--", color="gray")

ax.set_ylabel(sed_y_label)
ax.set_xlabel(r"$E\,/\,{\rm eV}$")
ax.set_xlim(x_axis_bounds_eV)
ax.legend(ncol=4, fontsize=9)
plt.title("Mrk421 MJD%s" % MJD)
plt.show()
