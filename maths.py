import numpy as np
import scipy

def bindata(x,y,nbins, covmat=None, yerr=None, scale='linear'):
    """
    Bin data x,y in nbins bins with covariance matrix covmat or error on y
    
    """
    if 2*nbins > len(x):
        raise ValueError('nbins must be at least twice as small as x and y')
    
    if scale == 'linear' :
        n, xedge = np.histogram(x, bins=nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
    
    if scale == 'log' :
        if x[0] == 0.:
            x = x[1:]
            y = y[1:]
            if covmat is not None :
                covmat = covmat[1:,1:]
            if yerr is not None :
                yerr = yerr[1:]
        goodenough = False
        while not goodenough :
            n, logxedge = np.histogram(np.log(x), bins=nbins)
            if 0 in n :
                nbins -= 1
            else :
                goodenough = True
        xedge = np.exp(logxedge)
        sy, _ = np.histogram(np.log(x), bins=nbins, weights=y)
        
    ymean = sy / n
    
    if yerr is not None :
        covmat = np.diag(yerr**2)
        
    if covmat is not None :
        yerr = np.zeros(len(n))
        idx = np.searchsorted(x,xedge)
        for i in range(len(n)):
            cov = covmat[idx[i]:idx[i+1], idx[i]:idx[i+1]]
            yerr[i] = np.sqrt(np.sum(cov) / n[i]**2.0) #(np.sum(np.diagonal(cov)) / n[i])**2.0
    #
    else:
        if scale == 'linear' :
            sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
            
        if scale == 'log' :
            sy2, _ = np.histogram(np.log(x), bins=nbins, weights=y*y)
            
        yerr = np.sqrt(sy2/n - ymean*ymean)
    #
    xmean = (xedge[1:] + xedge[:-1])/2.0
    xerr = (xedge[1:] - xedge[:-1])/2.0
    
    return xmean, xerr, ymean, yerr
#

def binmatrix(mat, m, sum_mean='sum'):
    """
    Simply bins a matrix in chunks of size m taking the average
    
    """
    nrow = mat.shape[0]/m
    ncol = mat.shape[1]/m
    
    a = np.zeros((nrow,ncol))
    
    for i in range(nrow):
        for j in range(ncol):
            if sum_mean == 'sum':
                a[i,j] = np.sum(mat[i*m:(i+1)*m, j*m:(j+1)*m])
            else :
                if sum_mean == 'mean' :
                    a[i,j] = np.mean(mat[i*m:(i+1)*m, j*m:(j+1)*m])
            
    return a
#

def chunk(seq, m):
    """
    Divide seq in m chuncks
    
    """
    n = len(seq)
    p = n / m
    r = n % m
    out = []
    last = 0
    
    for i in range(m):
        if r > 0:
            length = p + 1
        else:
            length = p
        out.append(seq[last:last + length])
        last +=length
        r -= 1
        
    return out
#

def matrix_normbydiag(mat):
    """
    Normalizes a matrix by its diagonal (so it's all 1's)
    
    """
    invdiag = np.diag(1./np.sqrt(np.diagonal(mat)))
    
    return np.dot(invdiag, np.dot(mat, invdiag))
#


def matrix_frowfcol(mat, frow, fcol, output=False):
    """
    Multiplies mat[i,j] by (function of i) * (function of j)
    
    """
    m, n = mat.shape
    if output:
        out = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                out[i,j] = mat[i,j] * frow[i]*fcol[j]
        return out
    else:
        for i in range(m):
            for j in range(n):
                mat[i,j] *= frow[i]*fcol[j]
#


def calc_chi2(x, cov, xmean=None):
    """
    Computes chi2 = (x-xmean)T . cov^-1 . (x-xmean)
    
    """
    if xmean is not None :
        y = x - xmean
    else :
        y = x
    
    icov = np.linalg.inv(cov)
    
    return np.dot(y.T, np.dot(icov, y))
#


def kronecker_array(size, i0):
    """
    Returns an array with size-1 0 and one 1 at position i0
    
    """
    delta = np.zeros(size)
    delta[i0] = 1.
    return delta
#


def rectangle_function(x,a,b, norm=True):
    """
    Normalized rectangle window function (integral=1) for a<x<b
    
    """
    f = lambda x : (a<x) & (x<b)
    fx = f(x).astype(float)
    if norm :
        fx /= float(b-a)
    return fx
#

def weightedvar(x, w=None):
    """
    Computes an unbiased weighted sample variance https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    
    
    
    """
    if w is None :
        return np.var(x, ddof=1)
    
    sw = np.sum(w)
    sw2 = np.sum(w**2)
    
    if sw == 0.0 :
        raise ZeroDivisionError
    else :
        xm = np.average(x, weights=w)
        sigma = np.average((x-xm)**2, weights=w)
        
        return sigma / (1. - sw2 / sw**2)
#


def weightedcovar(x, y, w=None):
    """
    Computes an unbiased weighted sample covvariance https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
    
    equivalent to np.cov(x,y=y,aweights=w, ddof=1)
    
    """
    if w is None :
        w = np.ones(len(x))
    
    sw = np.sum(w)
    sw2 = np.sum(w**2)
    
    if sw == 0.0 :
        raise ZeroDivisionError
    else :
        X = np.vstack([x,y]).T
        Xm = np.average(X, axis=0, weights=w)
        X -= Xm
        cov = np.dot(X.T,np.multiply(w,X))
        
        return sw * cov / (sw**2 - sw2)
#


def superfactorial(n):
    """
    Never use that !    
    
    """
    if n > 2 :
        return n**(superfactorial(n-1))
    else :
        return 1
#


def dist2spline(data, xmin=None, xmax=None, **kwargs):
    """
    Estimate a smooth 1D distribution from data with weights in the interval [xmin, xmax] and returns the knots and coefficients of a fitting spline
    
    """
    from pyqt_fit.kde import KDE1D
    from scipy.interpolate import UnivariateSpline
    
    if xmin is None :
        xmin = np.min(data)
    if xmax is None :
        xmax = np.max(data)
    
    kde = KDE1D(data, lower=xmin, upper=xmax, **kwargs)
        
    xx = np.linspace(xmin, xmax, 200)
    kdexx = kde(xx)
    
    s = UnivariateSpline(xx, kdexx, s=kde.bandwidth)
    
    xs = s.get_knots()
    ys = s(xs)
    return xs, ys, s
#


def nearPD(A, nit=10):
    """
    Computes the nearest positive definite matrix to A using Higham's algorithm (2002)
    
    (from http://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix, )
    
    """
    import numpy as np,numpy.linalg
    
    def _getAplus(A):
        eigval, eigvec = np.linalg.eig(A)
        Q = np.matrix(eigvec)
        xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
        return Q*xdiag*Q.T

    def _getPs(A, W=None):
        W05 = np.matrix(W**.5)
        return  W05.I * _getAplus(W05 * A * W05) * W05.I

    def _getPu(A, W=None):
        Aret = np.array(A.copy())
        Aret[W > 0] = np.array(W)[W > 0]
        return np.matrix(Aret)

    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk
#


def diag_cov_for_chi2(x, cov):
    """
    Goal : given a covariance matrix and a vector x, make it ready to be used in scipy.optimize.curve_fit, so that the chi2 = x.T * cov * x is given by a sum over diagonal terms.
    
    Explicitely,
    
    >> cov = V * D * V^-1 where V is orthogonal (so V^-1 = V.T) and D is diagonal
    >> chi2 = (x.T * V) * D * (V^-1 * x) = y.T * D * y where y = V^-1 * x = v.T * x
    
    Returns y and D
    
    """
    if not np.all(cov.T == cov) :
        print "covariance matrix not symmetric !"
        raise ValueError
    else :
        w, v = np.linalg.eigh(cov)
        
        diag = np.diag(w)
        y = np.dot(v.T, x)
    
        return y, diag
#


def chi2tosigma(chi2, ndof):
    """
    Computes the probability to exceed of chi2 for ndof degrees of freedom and,
    assuming a gaussian distribution, computes the deviation in sigmas, ie the rejection of null
    
    """
    pte0 = 1.- scipy.stats.chi2.cdf(chi2, ndof)
    fsigmaToPTE = lambda sigma: scipy.special.erfc(sigma/np.sqrt(2.)) - pte0
    sigma0 = scipy.optimize.brentq(fsigmaToPTE , 0., 50.)
    
    return pte0, sigma0
#


def logspace(xmin, xmax, n):
    """
    Returns a log-spaced array between xmin and xmax
    
    """
    return np.logspace(np.log10(xmin), np.log10(xmax), n)
#






