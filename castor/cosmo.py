import healpy as hp
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from .maths import _add_at_cst, _add_at

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
    Converts theta and phi (Healpix conventions, in radians) to ra and dec (in degrees)

    """
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)
    return ra, dec
#


def make_healpix_map(ra, dec, quantity, nside, mask=None, weight=None, ipix=None, fill_UNSEEN=False, return_w_maps=False, return_extra=False, mode='mean'):
    """
    Creates healpix maps of quantity observed at ra, dec (in degrees) by taking
    the mean or sum of quantity in each pixel.

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
    ipix : array
        `ipix` should be the array of healpix pixel indices corresponding to the
        input `ra` and `dec`. By default it is None and will be computed.

    fill_UNSEEN : boolean
        If `fill_UNSEEN` is True, pixels outside the mask are filled with
        hp.UNSEEN, 0 otherwise (the default is False).
    return_extra : boolean
        If True, a dictionnary is returned that contains count statistics and
        the masked `ipix` array to allow for statistics on the quantities to be
        computed.
    mode : string
        Whether to return the 'mean' or 'sum' of quantity in each pixel.

    Returns
    -------
    List of outmaps, the count map and the mask map.

    """

    if quantity is not None:
        quantity = np.atleast_2d(quantity)

        if weight is not None:
            w = np.atleast_2d(weight)
            # Weights can also be the same for all quantities
            # assert quantity.shape==weight.shape, "[make_healpix_map] quantity and weight must have the same shape"
            if w.shape[0] > 1:
                assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"
            else:
                w = np.tile(w[0], (quantity.shape[0],1))

            assert np.all(w > 0.), "[make_healpix_map] weight is not strictly positive"
        else:
            w = np.ones_like(quantity)

        assert quantity.shape == w.shape, "[make_healpix_map] quantity/weight arrays don't have the same length"

    npix = hp.nside2npix(nside)

    if mask is not None:
        assert len(mask)==npix, "[make_healpix_map] mask array does not have the right length"

    # Value to fill outside the mask
    x = hp.UNSEEN if fill_UNSEEN else 0.0

    # Make sure mode is correct
    assert (mode in ['sum','mean']), "[make_healpix_map] mode should be 'mean' or 'sum'"

    count = np.zeros(npix, dtype=float)
    outmaps = []
    sum_w_maps = []

    # Getting pixels for each object
    if ipix is None:
        assert len(ra) == len(dec), "[make_healpix_map] ra/dec arrays don't have the same length"
        if quantity is not None:
            assert len(ra) == quantity.shape[1]
        ipix = hp.ang2pix(nside, (90.0-dec)/180.0*np.pi, ra/180.0*np.pi)
    else:
        if quantity is not None:
            assert len(ipix) == quantity.shape[1], "[make_healpix_map] ipix has wrong size"

    # Counting objects in pixels
    # np.add.at(count, ipix, 1.)
    _add_at_cst(count, ipix, 1.)

    # Creating the mask if it does not exist
    if mask is None:
        bool_mask = (count > 0)
    else:
        bool_mask = mask.astype(bool)

    # # Masking the count in the masked area
    # count[np.logical_not(bool_mask)] = x
    # if mask is None:
    #     assert np.all(count[bool_mask] > 0), "[make_healpix_map] count[bool_mask] is not positive on the provided mask !"

    # Create the maps
    if quantity is not None:
        for i in range(quantity.shape[0]):
            sum_w = np.zeros(npix, dtype=float)
            # np.add.at(sum_w, ipix, w[i,:])
            _add_at(sum_w, ipix, w[i,:])

            outmap = np.zeros(npix, dtype=float)
            # np.add.at(outmap, ipix, quantity[i,:]*w[i,:])
            _add_at(outmap, ipix, quantity[i,:]*w[i,:])

            if mode=='mean':
                outmap[bool_mask] /= sum_w[bool_mask]
                
            outmap[np.logical_not(bool_mask)] = x

            outmaps.append(outmap)
            if return_w_maps:
                sum_w_maps.append(sum_w)

    if mask is None:
        returned_mask = bool_mask.astype(float)
    else:
        returned_mask = mask

    res = [outmaps, count, returned_mask]

    if return_w_maps:
        res += [sum_w_maps]

    if return_extra:
        extra = {}
        extra['count_tot_in_mask'] = np.sum(count[bool_mask])
        extra['count_per_pixel_in_mask'] = extra['count_tot_in_mask'] * 1. / np.sum(bool_mask.astype(int))
        extra['count_per_steradian_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=False)
        extra['count_per_sqdegree_in_mask'] = extra['count_per_pixel_in_mask'] / hp.nside2pixarea(nside, degrees=True)
        extra['count_per_sqarcmin_in_mask'] = extra['count_per_sqdegree_in_mask'] / 60.**2
        extra['ipix_masked'] = np.ma.array(ipix, mask=bool_mask[ipix])

        res += [extra]

    return res
#


def density2count(densitymap, nbar, mask=None, completeness=None, pixel=True):
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
    if completeness is None:
        completeness = mask.astype(float)

    if np.any(densitymap[mask.astype(bool)] < - 1.):
        print("[density2count] The density map has pixels below -1, will be clipped.")

    if pixel :
        nbarpix = nbar
    else :
        nbarpix = nbar * hp.nside2pixarea(hp.npix2nside(len(densitymap)))

    onepdelta = np.clip(1. + densitymap, 0., np.inf)
    onepdelta[np.logical_not(mask.astype(bool))] = 0.

    lamb = nbarpix * onepdelta * completeness

    return np.random.poisson(lamb)
#


def count2density(count, mask=None, completeness=None, density_convention=2):
    """
    Creates a reconstructed density map from count-in-pixel map count, with completeness and mask support.

    Under the assumption that count[i] = Poisson(completeness[i] * nbar * (1+density[i])), the MLE is:
        density[i]=(count[i])/(completeness[i]*nbar)-1.
    The true density per pixel nbar is unknown and the MLE is degenerate between nbar and the density, therefore an additional hypothesis is needed. The true density field is unknown too so a hypothesis has to be made on the density estimator itself.
    - case 1: mean(density[i])=0, where the mean over observed pixels.
    - case 2: mean(density[i]*completeness[i])=0, where the mean is over all pixels, even unobserved.
    Both cases are equivalent if the completeness is 1 within the mask. They amount to having the estimated density average to zero, without weights (case 1) or weighted by completeness (case 2).

    Parameters
    ----------
    count : array
        Healpix map of number count of object per pixel.
    mask : array
        Binary mask of the sky (the default is None).
    completeness : array (optional)
        Healpix map of the fraction each pixel has been observed, also called completeness or masked map fraction (the default is None).
    density_convention : int (optional)
        If 2 (default), uses case 2 where density*completeness averages to zero over full sky, if 1 uses case 1 where density averages to zero over observed sky.

    Returns
    -------
    array
        Density map.

    """

    assert density_convention in [1,2]

    npix = len(count)

    if completeness is None:
        completeness = np.ones(npix, dtype=float)
    if mask is None:
        mask = np.ones(npix, dtype=bool)

    msk = mask.astype(bool)

    assert np.all(completeness[msk] > 0.) , "[count2density] completeness has pixels = 0 in mask"
    assert np.all(completeness[msk] <= 1.) , "[count2density] completeness has pixels > 1 in mask"

    # Local mean density to compare count with.
    avg_in_pixel = np.zeros(npix, dtype=float)

    cnt = count[msk].astype(float)
    comp = completeness[msk].astype(float)
    if density_convention == 1:
        avg_in_pixel[msk] = comp * np.mean(cnt / comp)
    else:
        avg_in_pixel[msk] = comp * np.sum(cnt) / np.sum(comp)

    # Density
    density = np.zeros(npix, dtype=float)
    density[msk] = cnt / avg_in_pixel[msk] - 1.

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

def read_map_fits(filename, col, ext=1, nside_in=None, nside_out=None, ipix='HPIX', fill_UNSEEN=False):
    a = fits.open(filename)
    try:
        nside = int(a[ext].header['NSIDE'])
        if nside_in is not None:
            assert nside_in == nside
    except KeyError:
        assert nside_in is not None
        nside = nside_in
    npix = hp.nside2npix(nside)
    if fill_UNSEEN:
        b = np.ones(npix) * hp.UNSEEN
    else:
        b = np.zeros(npix)

    b[a[ext].data[ipix]] = a[ext].data[col]
    
    if nside_out is not None:
        return hp.ud_grade(a, nside_out)
    else:
        return b


def random_point_2dsphre(N, return_radec=True):
    """
    Generates N random points uniformly distributed on the sphere.

    Parameters
    ----------
    N : int
        Number of points
    return_radec : bool, optional
        If True, returns ra/dec in degrees, else theta/phi in radians, by default True

    Returns
    -------
    [type]
        [description]
    """
    u, v = np.random.rand(2,int(N))
    phi = 2. * np.pi * u
    theta = np.arccos(2*v - 1.)
    if return_radec:
        ra = np.rad2deg(phi)
        dec = np.rad2deg(np.pi/2-theta)
        return ra, dec
    else:
        return theta, phi
#


def make_mask_radec_cuts(nside, ramin, ramax, decmin, decmax):
    """
    Generates a square mask limited by RA/DEC.

    Parameters
    ----------
    nside : int
        `nside` parameter for healpix.
    ramin, ramax : float
        Should be in degrees in [0., 360.]
    decmin, decmax: [type]
        Should be in degrees in [-90., +90.]

    Returns
    -------
    array
        Returns the square mask.
    """
    # Make sure arguments are correct
    assert (0.<=ramin<=360.) & (0.<=ramax<=360.) & (ramin < ramax)
    assert (-90.<=decmin<=90.) & (-90.<=decmax<=90.) & (decmin < decmax)

    mask = np.zeros(hp.nside2npix(nside))
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    ths, phs = radec2thetaphi([ramin, ramax], [decmin, decmax])
    thmin, thmax = min(ths), max(ths)
    phmin, phmax = min(phs), max(phs)
    mask[np.where((th < thmax) & (th > thmin) & (ph > phmin) & (ph < phmax))[0]] = 1.
    return mask
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

    ids = np.concatenate(parallel.map(_ply2hp_aux, arglist)) #timesleep=1.0)

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


