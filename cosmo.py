import castor as ca
import healpy as hp
import numpy as np
import astropy.coordinates as coord
import astropy.units as u

def radec2thetaphi(ra,dec):
    """
    Converts ra and dec (in degrees) to theta and phi (Healpix conventions, in radians)

    """
    theta = np.pi/2. - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi
#


def thetaphi2radec(theta,phi):
    """
    Converts heta and phi (Healpix conventions, in radians) to ra and dec (in degrees)

    """
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)
    return ra, dec
#


def make_healpix_map(ra, dec, quantity, nside, mask=None, weight=None, fill_UNSEEN=False):
    """
    Creates healpix maps of quantity observed at ra, dec (in degrees) by taking
    the average of quantity in each pixel.

    Parameters
    ----------
    ra : array
        Right ascension.
    dec : array
        Declination.
    quantity : array
        `quantity` can be 2D, in which case several maps are created.
    nside : int
        `nside` parameter for healpix.
    mask : array
        If None, the mask is created and has value 1 in pixels that contain at least one object, 0 elsewhere.
    weight : type
        Weights of objects. Must be the same size as `quantity`.
    fill_UNSEEN : boolean
        If `fill_UNSEEN` is True, pixels outside the mask are filled with hp.UNSEEN, 0 otherwise.

    Returns
    -------
    type
        Description of returned object.

    """

    if weight is not None:
        assert quantity.shape==weight.shape, "[make_healpix_map] quantity and weight must have the same shape"
        assert np.all(weight > 0.), "[make_healpix_map] weight is not strictly positive"
        weight = np.atleast_2d(weight)
    else:
        weight = np.ones_like(quantity)

    quantity = np.atleast_2d(quantity)

    assert len(ra) == len(dec) == quantity.shape[0] == weight.shape[0], "[make_healpix_map] arrays don't have the same length"

    npix = hp.nside2npix(nside)

    if mask is not None:
        assert len(mask)==npix, "[make_healpix_map] mask array does not have the right length"

    count = np.zeros(npix, dtype=float)
    outmaps = [np.zeros(npix, dtype=float) for _ in range(quantity.shape[1])]

    # Getting pixels for each object
    ipix = hp.ang2pix(nside, (90-dec)/180*np.pi, ra/180*np.pi)

    # Counting objects and getting the mask
    np.add.at(count, ipix, 1.)
    mask = (count > 0)

    for i in range(quantity.shape[1]):
        sum_w = np.zeros(npix, dtype=float)
        np.add.at(sum_w, ipix, weight[:,i])

        np.add.at(outmaps[i], ipix, quantity[:,i]*weight[:,i])

        outmaps[i][mask] /= sum_w[mask]

        if fill_UNSEEN:
            x = hp.UNSEEN
        else :
            x = 0.0

        outmaps[i][np.logical_not(mask)] = x

    return outmaps, count, mask.astype(float)
#


def density2gal(densitymap, nbar, pixel=True, nside=None):
    """
    Generates a Poisson sampling of a given density map.

    If pixel=True, nbar is the mean number of galaxies per pixel.
    If pixel=False, nbar is the angular density (per unit steradian) and one needs nside to get the mean pixel number of galaxies.

    """
    if pixel :
        nbarpix = nbar
    else :
        nbarpix = nbar * hp.nside2pixarea(nside)

    lamb = nbarpix * np.clip(1. + densitymap, 0., np.inf)

    return np.random.poisson(lamb)
#


# def gal2density(galmap, binarymask=None, nside):
#     pixarea = hp.nside2pixarea(nside)
#     area = np.sum(binarymask) * pixarea
#     maskedgalmap = galmap * binarymask
#     Ngal = np.sum(maskedgalmap)
#     nbar = Ngal / area
#     return (galmap * binarymask / pixarea) / ngal - 1.
# #


def maskmap(hpmap, binarymask):
    """
    Mask a map, setting masked pixels to hp.UNSEEN

    """
    hpmap[np.where(binarymask == 0)] = hp.UNSEEN
#


def map1ell(alm, ell0, nside, mask = None):
    """
    Returns the ell0 multipole of a map

    """
    alm0 = hp.almxfl(alm, ca.kronecker_array(ell0+2,ell0))
    themap = hp.alm2map(alm0, nside)
    if mask is not None:
        return maskmap(themap, mask)
    else :
        return themap
#


############################################
#### Transform .ply masks to Healpix maps
from . import parallel
from . import maths
def ply2hp_aux(args):
    ply, ra, dec = args
    return ply.get_polyids(ra,dec)

def ply2hp(ply, nside, fk52gal=False):
    """
    Convert a mangle .ply mask into a Healpix map

    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,np.arange(npix))
    ra, dec = thetaphi2radec(theta, phi)

    if fk52gal :
        coord_gal = coord.SkyCoord(coord.Angle(ra*u.degree), coord.Angle(dec*u.degree), frame = 'galactic')
        coord_fk5 = coord_gal.transform_to('fk5')
        ra = coord_fk5.ra.deg
        dec = coord_fk5.dec.deg

    ral = maths.chunk(ra, 100)
    decl = maths.chunk(dec, 100)

    arglist = [(ply,ral[i],decl[i]) for i in range(100)]

    ids = np.concatenate(parallel.map(ply2hp_aux, arglist, timesleep=1.0))

    pos = np.where(ids > -1)

    hpmap = np.zeros(npix)
    hpmap[pos] = ply.weights[ids[pos]]

    return hpmap
#
#
# from ipyparallel import Client
# import mangle
#
# def mgl_get_polyids(mglobj, ra, dec):
#     """
#     Wrapper of mangle.Mangle.get_polyids(self, ra, dec) for parallelization
#
#     """
#     return mglobj.get_polyids(ra,dec)
# #
#
#
# def mgl_get_polyids_parallel(mglobj, ra, dec, nchunk):
#     """
#     Divides ra and dec in nchunk chunks and get the ids of polygones corresponding to these positions
#
#     """
#     ralist = ca.chunk(ra, nchunk)
#     declist = ca.chunk(dec, nchunk)
#
#     c = Client()   # here is where the client establishes the connection
#     lv = c.load_balanced_view()   # this object represents the engines (workers)
#
#     tasks = []
#
#     for i in range(nchunk):
#         tasks.append(lv.apply(mgl_get_polyids, mglobj, ralist[i], declist[i]))
#
#     result = [task.get() for task in tasks]
#
#     return np.concatenate(result)
#
#
#
# def ply2hp(ply, nside, fk52gal=False, nchunk=100):
#     """
#     Convert a mangle .ply mask into a Healpix map
#
#     """
#
#     npix = hp.nside2npix(nside)
#     theta, phi = hp.pix2ang(nside,np.arange(npix))
#     ra, dec = thetaphi2radec(theta, phi)
#
#     if fk52gal :
#         coord_gal = coord.SkyCoord(coord.Angle(ra*u.degree), coord.Angle(dec*u.degree), frame = 'galactic')
#         coord_fk5 = coord_gal.transform_to('fk5')
#         ra = coord_fk5.ra.deg
#         dec = coord_fk5.dec.deg
#
#     ids = mgl_get_polyids_parallel(ply, ra, dec, nchunk)
#     # iter = [(ply, ra[i], dec[i]) for i in range(npix)]
#     # ids = ca.map_q(mgl_get_polyids, iter, verbose=True, timeout=None, timesleep=1.0)
#
#     pos = np.where(ids > -1)
#
#     hpmap = np.zeros(npix)
#
#     hpmap[pos] = ply.weights[ids[pos]]
#
#     return hpmap
#

############################################
#### NFFT
# from pynfft import nfft
#
# def doNFFT (xv, fxv):
#     """
#     Computes the Fourier transform of a function with values fxv at irregularly spaced positions xv.
#
#     Returns kk and FT f_hat.
#
#     """
#     dx = xv[1:] - xv[:-1] # sampling
#     Dx = xv[-1] - xv[0] # width
#     m = len(xv) - 1
#
#     kfd = (2*np.pi) / Dx
#     knyq = np.pi / np.mean(dx)
#     kk = np.arange(0, knyq, kfd)
#     dk = kk[1] - kk[0]
#     n = len(kk)
#
#     plan = nfft.NFFT(N=2*n, M=m)
#     plan.x = xv[:-1] * dk / (2*np.pi)
#     plan.f = np.zeros(plan.M, dtype=np.complex128)
#     plan.f_hat = np.zeros(plan.N, dtype=np.complex128)
#     plan.precompute()
#
#     plan.f = fxv[:-1] * dx
#     plan.f_hat = np.zeros_like(plan.f_hat)
#     f_hat = np.copy(plan.adjoint())
#
#     del plan
#
#     return kk, f_hat[n:]
# #
#
# def power_spectrum_1D(xv, fxv):
#     k, fk = doNFFT(xv, fxv)
#
#     return k, (fk.real**2 + fk.imag**2)/(xv[-1]-xv[0])
