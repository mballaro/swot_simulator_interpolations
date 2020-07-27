import os
import swot_simulator.plugins.ssh

"""
Interpolation of the SSH enatl60
==============================
"""
import os
import re
import numpy as np
# import pyinterp.backends.xarray
import pyinterp
import xarray as xr
import time
import dask.array as da
import logging

# from swot_simulator.plugins.ssh import mit_gcm
#from . import detail
#from .mit_gcm import MITGCM, _time_interp
from swot_simulator.plugins.ssh import detail
from swot_simulator.plugins.ssh.mit_gcm import MITGCM, _time_interp


LOGGER = logging.getLogger(__name__)

def _spatial_interp_2D(z_model: da.array, x_model: da.array, y_model: da.array,
                    x_sat: np.ndarray, y_sat: np.ndarray):
    mesh = pyinterp.RTree(dtype="float32")
    x, y, z = (), (), ()

    start_time = time.time()
    x_face = x_model.compute()
    y_face = y_model.compute()
    # We test if the face covers the satellite positions.
    ix0, ix1 = x_face.min().values, x_face.max().values
    iy0, iy1 = y_face.min().values, y_face.max().values

    box = pyinterp.geodetic.Box2D(pyinterp.geodetic.Point2D(ix0, iy0),
                                    pyinterp.geodetic.Point2D(ix1, iy1))
    mask = box.covered_by(x_sat, y_sat)
    if not np.any(mask == 1):
        print ("no covers")
    del box, mask

    # The undefined values are filtered
    z_face = z_model.compute()
    defined = ~np.isnan(z_face)

    # The tree is built and the interpolation is calculated
    x = x_face.values[defined]
    y = y_face.values[defined]

    coordinates = np.vstack((x, y)).T
    del x, y

    z = z_face.values[defined]
    LOGGER.info("loaded %d MB in %.2fs",
                 (coordinates.nbytes + z.nbytes) // 1024**2,
                 time.time() - start_time)
    start_time = time.time()
    mesh.packing(coordinates, z)
    LOGGER.info("mesh build in %.2fs", time.time() - start_time)

    del coordinates, z

    start_time = time.time()
    z, _ = mesh.inverse_distance_weighting(np.vstack(
        (x_sat, y_sat)).T.astype("float32"),
                                           within=True,
                                           k=11,
                                           radius=8000,
                                           num_threads=1)
    LOGGER.debug("interpolation done in %.2fs", time.time() - start_time)
    del mesh
    return z.astype("float32")


class Enatl60(MITGCM):
    """
    Interpolation of the SSH eNATL60 products in zarr format.
    """

    def __init__(self, path): #xc: xr.DataArray, yc: xr.DataArray, eta: xr.DataArray):
        self.path = path
        with xr.open_zarr(self.path) as ds:
            self.lon = ds.nav_lon
            self.lat = ds.nav_lat
            self.ssh = ds.sossheig
            self.ts = ds.time_counter.data.astype("datetime64[us]")
        self.dt = self._calculate_dt(self.ts)

    def interpolate(self, lon: np.ndarray, lat: np.ndarray,
                    dates: np.ndarray) -> np.ndarray:
        """Interpolate the SSH for the given coordinates
        copy from the mit_gcm.interpolate function, just to overwrite spatial_interp function
        """
        first_date = self._grid_date(dates[0], -1)
        last_date = self._grid_date(dates[-1], 1)

        if first_date < self.ts[0] or last_date > self.ts[-1]:
            raise IndexError(
                f"period [{first_date}, {last_date}] is out of range: "
                f"[{self.ts[0]}, {self.ts[-1]}]")

        # Mask for selecting data covering the time period provided.
        mask = (self.ts >= first_date) & (self.ts <= last_date)

        LOGGER.debug("fetch data for %s, %s", first_date, last_date)

        # 4D cube representing the data necessary for interpolation.
        frame = self.ssh[mask]

        # Spatial interpolation of the SSH on the different selected grids.
        start_time = time.time()
        layers = []
        for index in range(len(frame)):
            layers.append(
                _spatial_interp_2D(frame[index, :], self.lon, self.lat, lon, lat))

        # Time interpolation of the SSH.
        layers = np.stack(layers)
        LOGGER.debug("interpolation completed in %.2fs for period %s, %s",
                     time.time() - start_time, first_date, last_date)
        return _time_interp(self.ts[mask].astype("int64"), layers,
                            dates.astype("datetime64[us]").astype("int64"))




# Geographical area to simulate defined by the minimum and maximum corner
# point :lon_min, lat_min, lon_max, lat_max
#
# Default: None equivalent to the area covering the Earth: -180, -90, 180, 90
# area = None

# Distance, in km, between two points along track direction.
delta_al = 2.0

# Distance, in km, between two points across track direction.
delta_ac = 2.0

# Ephemeris file to read containing the satellite's orbit.
# ephemeris = os.path.join("..", "..", "data", "ephem_science_sept2015_ell.txt")
ephemeris = "/home/ad/ballarm/tools/swotsimulator_interp/swot_simulator/data/ephem_science_sept2015_ell.txt"


# Index of columns to read in the ephemeris file containing, respectively,
# longitude in degrees, latitude in degrees and the number of seconds elapsed
# since the start of the orbit.
# Default: 1, 2, 0
ephemeris_cols = [1, 2, 0]

# If true, the swath, in the final dataset, will contain a center pixel
# divided in half by the reference ground track.
central_pixel = True

# If true, the generated netCDF file will be the complete product compliant
# with SWOT's Product Description Document (PDD), otherwise only the calculated
# variables will be written to the netCDF file.
complete_product = False

# Distance, in km, between the nadir and the center of the first pixel of the
# swath
half_gap = 2.0

# Distance, in km, between the nadir and the center of the last pixel of the
# swath
half_swath = 70.0

# Limits of SWOT swath requirements. Measurements outside the span will be set
# with fill values.
requirement_bounds = [10, 60]

# The next two parameters (cycle_duration and height) can be read from the
# ephemeris file if it includes these values in comments. The ephemeris
# delivered with this software contain this type of declaration

# Duration of a cycle.
# #cycle_duration=

# Satellite altitude (m)
# #height=

# True to generate Nadir products
nadir = True

# True to generate swath products
swath = True

# Type of SWOT product to be generated. Possible products are "expert",
# "basic" and "wind_wave". Default to expert
product_type = "basic"

# The plug-in handling the SSH interpolation under the satellite swath.
# ssh_plugin = swot_simulator.plugins.ssh.AVISO("PATH to AVISO files")
ssh_plugin = Enatl60("/work/ALT/swot/aval/wisa/tmp/eNATL60-BLBT02-SSH-1h")

# Orbit shift in longitude (degrees)
# #shift_lon=

# Orbit shift in time (seconds)
# #shift_time=

# The working directory. By default, files are generated in the user's root
# directory.
# #working_directory=
working_directory='/work/ALT/swot/aval/wisa/tmp/swot_simulator_science_phase/eNATL60-BLBT02-SSH-1h/'


# Generation of measurement noise.

# The calculation of roll errors can be simulated, option "roll_phase", or
# interpolated, option "corrected_roll_phase", from the dataset specified by
# the option "roll_phase_dataset". Therefore, these two options are
# mutually exclusive. In other words, if the "roll_phase" option is present,
# the "corrected_roll_phase" option must be omitted, and vice versa.
noise = [
    'altimeter',
    'baseline_dilation',
    'karin',
    # 'corrected_roll_phase',
    'roll_phase',
    'timing',
    'wet_troposphere',
]

# repeat length
len_repeat = 20000

# File containing spectrum of instrument error
# error_spectrum = os.path.join("..", "..", "data", "error_spectrum.nc")
error_spectrum = os.path.join("/home/ad/ballarm/tools/swotsimulator_interp/swot_simulator/data/error_spectrum.nc")

# KaRIN file containing spectrum for several SWH
# karin_noise = os.path.join("..", "..", "data", "karin_noise_v2.nc")
karin_noise = os.path.join("/home/ad/ballarm/tools/swotsimulator_interp/swot_simulator/data/karin_noise_v2.nc")

# Estimated roll phase dataset
# #corrected_roll_phase_dataset =

# SWH for the region
swh = 2.0

#  Number of km of random coefficients for KaRIN noise (recommended
# nrand_karin=1000)
nrand_karin = 1000

# Number of beam used to correct wet troposphere signal (1, 2 or 'both')
nbeam = 2

# Gaussian footprint of sigma km
sigma = 6.0

# Beam position if there are 2 beams (in km from nadir):
beam_position = [-20, 20],

# Seed for RandomState. Must be convertible to 32 bit unsigned integers.
nseed = 0
