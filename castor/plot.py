import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib as mpl
import numpy as np

def mat_symlog(my_matrix, vmin=None, vmax=None, logthresh = 5):
    """
    Plot my_matrix with a symmetric logarithmic scale

    from http://stackoverflow.com/questions/11138706/colorbar-for-imshow-centered-on-0-and-with-symlog-scale

    """
    img = plt.matshow( my_matrix ,
                vmin=vmin, vmax=vmax,
                norm=clr.SymLogNorm(10**-logthresh) )

    if vmin is None :
        vmin = np.min(my_matrix)
    if vmax is None :
        vmax = np.max(my_matrix)
    maxlog = int(np.ceil( np.log10(vmax) ))
    minlog = int(np.ceil( np.log10(-vmin) ))

    #generate logarithmic ticks
    tick_locations = ([-(10**x) for x in xrange(minlog,-logthresh-1,-1)]
                    +[0.0]
                    +[(10**x) for x in xrange(-logthresh,maxlog+1)] )

    cb = plt.colorbar(ticks=tick_locations, format='%.2e')

    return img,cb
#


def scatter_hist(x,y, bins, xlabel=None, ylabel=None, *args, **kwargs):
    """
    Makes a scatter plot with histograms of x/y on the upper/right side (plus 1 to 5 sigma lines)

    from http://matplotlib.org/examples/pylab_examples/scatter_hist.html
    """

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1)#, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    # axHistx.xaxis.set_major_formatter(nullfmt)
    # axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, **kwargs)


    # now determine nice limits by hand:
    # binwidth = 0.25
    # xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    # lim = (int(xymax/binwidth) + 1) * binwidth

    # axScatter.set_xlim((-lim, lim))
    # axScatter.set_ylim((-lim, lim))

    # bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins, linewidth=0)
    meanx = np.mean(x)
    stdx = np.std(x)
    axHistx.axvline(x=meanx, ls='-', c='r')
    for i in range(1,6):
            axHistx.axvline(x=meanx + i * stdx, ls='--', c='r', alpha=1./float(i))
            axHistx.axvline(x=meanx - i * stdx, ls='--', c='r', alpha=1./float(i))


    axHisty.hist(y, bins=bins, linewidth=0, orientation='horizontal')
    meany = np.mean(y)
    stdy = np.std(y)
    axHisty.axhline(y=meany, ls='-', c='r')#, orientation='horizontal')
    for i in range(1,6):
            axHisty.axhline(y=meany + i * stdy, ls='--', c='r', alpha=1./float(i))
            axHisty.axhline(y=meany - i * stdy, ls='--', c='r', alpha=1./float(i))

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # plt.show()
#


def colorz(x, y, z, plotfct=plt.plot, cmapname='jet', *args, **kwargs):
    """
    Plots series of (x,y) with color linearly scalled from third parameter z

    From http://stackoverflow.com/a/11558629/3245309

    """
    zmin = np.min(z)
    zmax = np.max(z)

    cmap = plt.get_cmap(cmapname)
    norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

    if x.ndim > 1 :
        for i in range(len(z)):
            plotfct(x[i], y[i], c=cmap((z[i]-zmin)/(zmax-zmin)), *args, **kwargs)
    else :
        for i in range(len(z)):
            plotfct(x, y[i], c=cmap((z[i]-zmin)/(zmax-zmin)), *args, **kwargs)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    plt.colorbar(sm)
#


def step_bins(xbins, yvalues, *args, **kwargs):
    """
    Step plot where y=y[i] for x[i] < x < x[i+1]

    From http://stackoverflow.com/questions/11297030/matplotlib-stepped-histogram-with-already-binned-data

    """

    x = np.ravel(zip(xbins[:-1], xbins[1:]))
    y = np.ravel(zip(yvalues, yvalues))

    plt.plot(x, y, *args, **kwargs)
#


def plot_linearlog(x, y, xtr, *args, **kwargs):
    """
    Plots x,y with linear scale for x < xtr and a log scale for x>xtr

    from http://stackoverflow.com/questions/21746491/combining-a-log-and-linear-scale-in-matplotlib

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    axLin = plt.subplot(111)
    axLin.plot(x, y, *args, **kwargs)
    axLin.set_xscale('linear')
    axLin.set_xlim((np.min(x), xtr))
    axLin.spines['right'].set_visible(False)
    # axMain.xaxis.set_ticks_position('bottom')

    divider = make_axes_locatable(axLin)
    axLog = divider.append_axes("right", size=3.0, pad=0, sharey=axLin)
    axLog.plot(x, y)
    axLog.set_xscale('log')
    axLog.set_xlim((xtr, np.max(x)))

    # Removes bottom axis line
    axLog.spines['left'].set_visible(False)
    axLog.yaxis.set_ticks_position('right')
    plt.setp(axLog.get_yticklabels(), visible=False)
#

def plot_loglinear(x, y, xtr, *args, **kwargs):
    """
    Plots x,y with log scale for x < xtr and a linear scale for x>xtr

    from http://stackoverflow.com/questions/21746491/combining-a-log-and-linear-scale-in-matplotlib

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    axLin = plt.subplot(111)
    axLin.plot(x, y, *args, **kwargs)
    axLin.set_xscale('log')
    axLin.set_xlim((np.min(x), xtr))
    axLin.spines['right'].set_visible(False)
    # axMain.xaxis.set_ticks_position('bottom')

    divider = make_axes_locatable(axLin)
    axLog = divider.append_axes("right", size=3.0, pad=0, sharey=axLin)
    axLog.plot(x, y)
    axLog.set_xscale('linear')
    axLog.set_xlim((xtr, np.max(x)))

    # Removes bottom axis line
    axLog.spines['left'].set_visible(False)
    axLog.yaxis.set_ticks_position('right')
    plt.setp(axLog.get_yticklabels(), visible=False)
#

def scatter_cov(x, y, cov=None, covcmap=plt.cm.YlOrBr_r, label=None, *args, **kwargs):
    """"
    Scatter plot with covariance contours

    from http://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#example-covariance-plot-mahalanobis-distances-py

    """
    from scipy.spatial.distance import mahalanobis

    a = plt.scatter(x, y, *args, **kwargs)

    # xm = np.max(np.fabs(x))
    # ym = np.max(np.fabs(y))
    #
    # plt.xlim((-xm,xm))
    # plt.ylim((-ym,ym))

    if 'c' in kwargs.keys():
        plt.colorbar()

    xlim = a.axes.get_xlim()
    ylim = a.axes.get_ylim()

    xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                         np.linspace(plt.ylim()[0], plt.ylim()[1], 100))

    zz = np.c_[xx.ravel(), yy.ravel()]

    z0 = [np.mean(x), np.mean(y)]

    if cov is None :
        cov = np.cov(np.vstack([x,y]), ddof=1)

    prec = np.linalg.inv(cov)

    mahal = np.array([mahalanobis(z, z0, prec) for z in zz]).reshape(xx.shape)

    cov_contour = plt.contour(xx, yy, mahal, levels=range(1,7), cmap=covcmap, linestyles='dashed')

    if label is not None :
        plt.legend(cov_contour.collections[1], label=label, loc="upper right", borderaxespad=0)

    plt.xlim(xlim)
    plt.ylim(ylim)

    return a

#

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    From https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
#

from matplotlib.colors import Normalize

class MidPointNorm(Normalize):
    """
    From https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Subclass of matplotlib.colors.Normalize that sets the mid point of the color scale at midpoint (default=0.)

    """

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False, symmetric=True):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint
        self.symmetric = symmetric

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        if self.symmetric:
            vabs = np.max(np.abs(result))
            self.vmax = vabs
            self.vmin = -vabs

        else :
            self.autoscale_None(result)

        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin <= midpoint <= vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
        # if vmin == vmax:
        #     result.fill(0) # Or should it be all masked? Or 0.5?
        # elif vmin > vmax:
        #     raise ValueError("maxvalue must be bigger than minvalue")
        # else:
        #     if midpoint < vmin:
        #         vmin = midpoint
        #     if vmax < midpoint:
        #         vmax = midpoint
            # print vmin, midpoint, vmax
            vmin = float(vmin)
            vmax = float(vmax)

            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.ma.array(resdat, mask=result.mask, copy=False)
            # print result

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if self.symmetric:
            vabs = np.max(np.abs([vmin, vmax]))
            vmax = + vabs
            vmin = - vabs

        print("call inverse :", vmin, vmax, midpoint)
        if mpl.cbook.iterable(value):
            val = np.ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


class SymStdNorm(Normalize):
    """
    From https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Subclass of matplotlib.colors.Normalize that sets the mid point of the color scale at midpoint (default=0.)

    """

    def __init__(self, Nsigma=3, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.Nsigma = Nsigma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        mean = np.mean(result)
        std  = np.std(result)
        self.vmax = mean + self.Nsigma * std
        self.vmin = mean - self.Nsigma * std

        if self.vmin == self.vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?

        else:
            result -= mean
            result /= (2 * self.Nsigma * std)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if self.symmetric:
            vabs = np.max(np.abs([vmin, vmax]))
            vmax = + vabs
            vmin = - vabs

        print("call inverse :", vmin, vmax, midpoint)
        if mpl.cbook.iterable(value):
            val = np.ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

class Radar(object):
    """
    From https://stackoverflow.com/questions/24659005/radar-chart-with-multiple-scales-on-multiple-axes

    Usage: Radar(figure, axes_labels, ticks_labels)

    Example:
    fig = pl.figure(figsize=(6, 6))

    titles = list("ABCDE")

    labels = [
        list("abcde"), list("12345"), list("uvwxy"),
        ["one", "two", "three", "four", "five"],
        list("jklmn")
    ]

    radar = Radar(fig, titles, labels)
    radar.plot([1, 3, 2, 5, 4],  "-", lw=2, color="b", alpha=0.4, label="first")
    radar.plot([2.3, 2, 3, 3, 2],"-", lw=2, color="r", alpha=0.4, label="second")
    radar.plot([3, 4, 3, 4, 2], "-", lw=2, color="g", alpha=0.4, label="third")
    radar.ax.legend()

    """
    def __init__(self, fig, titles, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.n = len(titles)
        self.angles = np.arange(90, 90+360, 360.0/self.n)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                         for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(np.arange(0.2,1.1,0.2), angle=angle, labels=label)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 1)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
