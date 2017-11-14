from math import cos
from math import pow
from math import sqrt

from prototype.utils.utils import deg2rad

# GLOBAL_VARIABLE
EARTH_RADIUS_M = 6378137.0


def latlon_offset(lat_ref, lon_ref, offset_N, offset_E):
    """Calculate new latitude and longitude coordinates with an offset in
    North and East directions

    Parameters
    ----------
    lat_ref : float
        Latitude of origin (decimal format)
    lon_ref : float
        Longitude of origin (decimal format)
    offset_N : float
        Offset in North direction (meters)
    offset_E : float
        Offset in East direction (meters)

    Returns
    -------

        (lat, lon)

    """
    lat_new = lat_ref + (offset_E / EARTH_RADIUS_M)
    lon_new = lon_ref + (offset_N / EARTH_RADIUS_M) / cos(deg2rad(lat_ref))
    return (lat_new, lon_new)


def latlon_diff(lat_ref, lon_ref, lat, lon):
    """Calculate difference in North and East from two GPS coordinates

    Parameters
    ----------
    lat_ref : float
        Latitude of origin (decimal format)
    lon_ref : float
        Longitude of origin (decimal format)
    offset_N : float
        Offset in North direction (meters)
    offset_E : float
        Offset in East direction (meters)
    lat :

    lon :


    Returns
    -------

    """
    d_lon = lon - lon_ref
    d_lat = lat - lat_ref

    dist_N = deg2rad(d_lat) * EARTH_RADIUS_M
    dist_E = deg2rad(d_lon) * EARTH_RADIUS_M * cos(deg2rad(lat))

    return (dist_N, dist_E)


def latlon_dist(lat_ref, lon_ref, lat, lon):
    """Calculate Euclidean distance between two GPS coordinates

    Parameters
    ----------
    lat_ref : float
        Latitude of origin (decimal format)
    lon_ref : float
        Longitude of origin (decimal format)
    lat : float
        Latitude of target (decimal format)
    lon : float
        Longitude of target (decimal format)

    Returns
    -------

        Euclidean distance between two GPS coordinates (float)

    """
    (dist_N, dist_E) = latlon_diff(lat_ref, lon_ref, lat, lon)
    dist = sqrt(pow(dist_N, 2) + pow(dist_E, 2))
    return dist
