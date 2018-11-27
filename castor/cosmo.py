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


def make_healpix_map(ra, dec, quantity, nside, mask=None, weight=None, fill_UNSEEN=False, return_extra=False):
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
        If None, the mask is created and has value 1 in pixels that contain at
        least one object, 0 elsewhere.
    weight : type
        Weights of objects (the default is None, in which case all objects have
        weight 1). Must be the same size as `quantity`.
    fill_UNSEEN : boolean
        If `fill_UNSEEN` is True, pixels outside the mask are filled with
        hp.UNSEEN, 0 otherwise (the default is False).
    return_extra : boolean
        If True, a dictionnary is returned that contains count statistics and
        the masked `ipix` array to allow for statistics on the quantities to be
        computed.

    Returns
    -------
    List of outmaps, the count map and the mask map.

    """

    if quantity is not None:
        quantity = np.atleast_2d(quantity)

        if weight is not None:
            assert quantity.shape==weight.shape, "[make_healpix_map] quantity and weight must have the same shape"
            assert np.all(weight > 0.), "[make_healpix_map] weight is not strictly positive"
            weight = np.atleast_2d(weight)
        else:
            weight = np.ones_like(quantity)

        assert quantity.shape[1] == weight.shape[1], "[make_healpix_map] quantity/weight arrays don't have the same length"

        assert len(ra) == len(dec), "[make_healpix_map] ra/dec arrays don't have the same length"

    npix = hp.nside2npix(nside)

    if mask is not None:
        assert len(mask)==npix, "[make_healpix_map] mask array does not have the right length"

    # Value to fill outside the mask
    x = hp.UNSEEN if fill_UNSEEN else 0.0

    count = np.zeros(npix, dtype=float)
    outmaps = []

    # Getting pixels for each object
    ipix = hp.ang2pix(nside, (90.0-dec)/180.0*np.pi, ra/180.0*np.pi)

    # Counting objects in pixels
    np.add.at(count, ipix, 1.)

    # Creating the mask if it does not exist
    if mask is None:
        bool_mask = (count > 0)
    else:
        bool_mask = mask.astype(bool)

    # Masking the count in the masked area
    count[np.logical_not(bool_mask)] = x
    if mask is None:
        assert np.all(count[bool_mask] > 0), "[make_healpix_map] count[bool_mask] is not positive on the provided mask !"

    # Create the maps
    if quantity is not None:
        for i in range(quantity.shape[0]):
            sum_w = np.zeros(npix, dtype=float)
            np.add.at(sum_w, ipix, weight[i,:])

            outmap = np.zeros(npix, dtype=float)
            np.add.at(outmap, ipix, quantity[i,:]*weight[i,:])
            outmap[bool_mask] /= sum_w[bool_mask]
            outmap[np.logical_not(bool_mask)] = x

            outmaps.append(outmap)

    if mask is None:
        returned_mask = bool_mask.astype(float)
    else:
        returned_mask = mask

    if return_extra:
        extra = {}
        extra['count_tot_in_mask'] = np.sum(count[bool_mask])
        extra['count_per_pixel_in_mask'] = extra['count_tot_in_mask'] * 1. / np.sum(bool_mask.astype(int))
        extra['count_per_steradian_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=False)
        extra['count_per_sqdegree_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=True)
        extra['count_per_sqarcmin_in_mask'] = extra['count_per_sqdegree_in_mask'] / 60.**2
        extra['ipix_masked'] = np.ma.array(ipix, mask=bool_mask[ipix])

        return outmaps, count, returned_mask, extra
    else:
        return outmaps, count, returned_mask
#


def density2Ngal(densitymap, nbar, mask=None, pixel=True):
    """
    Generates a Poisson sampling of a given density map.

    Parameters
    ----------
    densitymap : array
        Healpix map of density field. Note that the density map is clipped where it is below -1.
    nbar : float
        Mean galaxy density.
    mask : array (optional)
        Binary mask of where to perform sampling.
    pixel : type
        If pixel=True (default), nbar is the mean number of galaxies per pixel.
        If pixel=False, nbar is the angular density (per unit steradian).
    Returns
    -------
    array
        Map with the number of object per pixel.

    """
    if mask is None:
        mask = np.ones(len(densitymap), dtype=bool)

    if np.any(densitymap[mask] < - 1.):
        print("[density2Ngal] The density map has pixels below -1.")

    if pixel :
        nbarpix = nbar
    else :
        nbarpix = nbar * hp.nside2pixarea(hp.npix2nside(len(densitymap)))

    onepdelta = np.clip(1. + densitymap, 0., np.inf)
    onepdelta[mask.astype(bool)] = 0.

    lamb = nbarpix * onepdelta

    return np.random.poisson(lamb)
#

def count2density(Ngal, mskfrac_map=None, mask=None):
    """
    Creates a reconstructed density map from count-in-pixel map Ngal, with completeness and mask support.

    Parameters
    ----------
    Ngal : array
        Healpix map of number count of object per pixel.
    mskfrac_map : array (optional)
        Healpix map of the fraction each pixel has been observed, also called completeness or masked map fraction (the default is None).
    mask : array
        Binary mask of the sky (the default is None).

    Returns
    -------
    array
        Density map.

    """

    npix = len(Ngal)

    if mskfrac_map is None:
        mskfrac_map = np.ones(npix, dtype=float)
    if mask is None:
        mask = np.ones(npix, dtype=bool)

    msk = mask.astype(bool)

    # Local mean density to compare Ngal with.
    avg_in_pixel = np.zeros(npix, dtype=float)
    avg_in_pixel[msk] = mskfrac_map[msk] * np.sum(Ngal[msk]) / np.sum(mskfrac_map[msk])

    # Density
    density = np.zeros(npix, dtype=float)
    density[msk] = Ngal[msk] / avg_in_pixel[msk] - 1.

    return density
#


def maskmap(hpmap, binarymask, fill_UNSEEN=False):
    """
    Mask a map in place.

    Parameters
    ----------
    hpmap : array
        Input map.
    binarymask : array
        Mask.
    fill_UNSEEN : bool
        If True, fills pixels outside the mask with hp.UNSEEN, else with 0.0 (the default is False).

    Returns
    -------
    None
    """

    x = hp.UNSEEN if fill_UNSEEN else 0.0

    hpmap[np.logical_not(binarymask.astype(bool))] = x
#

def plot_hp_skymapper(obs, mask, projection=None, filename=None, vmax=None, cmap=None, cb_label=None, nside_out=True):
    """
    Plot a healpix masked map using skymapper.

    Parameters
    ----------
    obs : array
        Healpix map.
    mask : array
        Mask map.
    projection : type
        - 'DESY3', in which case it uses precomputed best projection for the Y3 footprint,
        - a predefined skymapper projection objectself,
        - or None, if which case the projection is infered from the mask.
    filename : string
        If not None, the name of the file to save the figure (the default is None).
    vmax : type
        - a float, in which case the color scale goes from -vmax to +vmax
        - 'best', in which case skymapper uses the 10-90 percentiles
        - None, in which case the min/max values of `obs` are used.
    cmap : type
        Color map (the default is None).
    cb_label : string
        Label of the color bar (the default is None).
    nside_out : bool or int
        Whether to degrade obs to make if faster or the nside to use (the default
        is True, in which case nside=256 is used).

    Returns
    -------
    type
        Description of returned object.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    import skymapper as skm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    assert obs.shape == mask.shape, "[plot_hp_skymapper] `obs` and `mask` don't have the same shape."

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    nside_in = hp.npix2nside(len(mask))
    theta, phi = hp.pix2ang(nside_in, np.arange(len(mask))[mask.astype(bool)])
    ra, dec = thetaphi2radec(theta, phi)

    if projection == 'DESY3':
        # define the best Albers projection for the footprint
        # minimizing the variation in distortion
        # crit = skm.meanDistortion
        # proj = skm.Albers.optimize(a['ra'], a['dec'], crit=crit)
        proj = skm.Albers( 28.51234891, -43.90175288, -55.63596295, -32.39570739)
    else :
        if projection is None:
            crit = skm.meanDistortion
            proj = skm.Albers.optimize(ra, dec, crit=crit)

    # construct map: will hold figure and projection
    # the outline of the sphere can be styled with kwargs for matplotlib Polygon
    map = skm.Map(proj, ax=ax)

    # add graticules, separated by 15 deg
    # the lines can be styled with kwargs for matplotlib Line2D
    # additional arguments for formatting the graticule labels
    # sep = 15
    map.grid()
    map.focus(ra, dec)

    obs_plot = np.copy(obs)
    maskmap(obs_plot, mask, fill_UNSEEN=False)

    if nside_out:
        if type(nside_out) is int:
            obs_plot = hp.ud_grade(obs_plot, nside_out=nside_out)
        else:
            obs_plot = hp.ud_grade(obs_plot, nside_out=256)

    if type(vmax) is float:
        mappable = map.healpix(obs_plot, cmap=cmap, vmin=-vmax, vmax=vmax)
    if vmax == 'best' : # skymapper uses 10-90 percentiles
        mappable = map.healpix(obs_plot, cmap=cmap)
    if vmax is None :
        mappable = map.healpix(obs_plot, cmap=cmap, vmin=np.min(obs_plot), vmax=np.max(obs_plot))

    map.colorbar(mappable, cb_label=cb_label)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

############################################
#### Transform .ply masks to Healpix maps
from . import parallel
from . import maths

def _ply2hp_aux(args):
    ply, ra, dec = args
    return ply.get_polyids(ra,dec)

def ply2hp(ply, nside, fk52gal=False):
    """
    Fast convertion of a mangle .ply mask into a Healpix map using the weight of
    the polygon where centers of healpix pixels fall.

    Parameters
    ----------
    ply : mangle.Mangle
        mangle.Mangle object.
    nside : int
        `nside` of output healpix map.
    fk52gal : bool
        If true, transform galactic coordinates to fk5 coordinates, useful for galaxy surveys (the default is False).

    Returns
    -------
    array
        Healpix map.

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

    ids = np.concatenate(parallel.map(_ply2hp_aux, arglist, timesleep=1.0))

    pos = np.where(ids > -1)

    hpmap = np.zeros(npix)
    hpmap[pos] = ply.weights[ids[pos]]

    return hpmap

############################
# Same but using ipyparallel
############################
#
# from ipyparallel import Client
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
# #
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


###########################################
### NFFT

def doNFFT (xv, fxv):
    """
    Computes the Fourier transform of a function with values fxv at irregularly
    spaced positions xv.

    Parameters
    ----------
    xv : array
    fxv : array

    Returns
    -------
    array
        Returns the modes kk and the Fourier transform evaluated at those modes f_hat(kk).

    """
    from pynfft import nfft

    dx = xv[1:] - xv[:-1] # sampling
    Dx = xv[-1] - xv[0] # width
    m = len(xv) - 1

    kfd = (2*np.pi) / Dx
    knyq = np.pi / np.mean(dx)
    kk = np.arange(0, knyq, kfd)
    dk = kk[1] - kk[0]
    n = len(kk)

    plan = nfft.NFFT(N=2*n, M=m)
    plan.x = xv[:-1] * dk / (2*np.pi)
    plan.f = np.zeros(plan.M, dtype=np.complex128)
    plan.f_hat = np.zeros(plan.N, dtype=np.complex128)
    plan.precompute()

    plan.f = fxv[:-1] * dx
    plan.f_hat = np.zeros_like(plan.f_hat)
    f_hat = np.copy(plan.adjoint())

    del plan

    return kk, f_hat[n:]
#

def power_spectrum_1D_NFFT(xv, fxv):
    """
    Computes the power spectrum of a function with values fxv at irregularly
    spaced positions xv.

    Parameters
    ----------
    xv : array
    fxv : array

    Returns
    -------
    array
        Returns the modes kk and the power spectrum evaluated at those modes.

    """

    k, fk = doNFFT(xv, fxv)
    return k, (fk.real**2 + fk.imag**2)/(xv[-1]-xv[0])
