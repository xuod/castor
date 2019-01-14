import multiprocessing
import time

def map_f(args):
    f, i, v = args
    v.value += 1
    return f(i)

def map(func, iter, verbose=True, timesleep=15.0, timeout=None):
    """
    Maps the function `func` over the iterator `iter` in a multi-threaded way
    using the multiprocessing package.


    Parameters
    ----------
    func : function
        func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled .
    iter : iterator
    verbose : bool
        Whether to print messages while the computation is on-going (the default is True).
    timesleep : float
        Number of seconds between messages (the default is 15.0).
    timeout : float
        Number of seconds before cancelling the whole thing (the default is None).

    Returns
    -------
    type
        Result of the computation of func(iter).
    """

    pool = multiprocessing.Pool()
    m = multiprocessing.Manager()
    v = m.Value(int, 0)

    inputs = ((func,i,v) for i in iter) #use a generator, so that nothing is computed before it's needed :)
    
    res = pool.map_async(map_f, inputs)

    try :
        n = len(iter)
    except TypeError : # if iter is a generator
        n = None

    if verbose :
        while (True):
            if (res.ready()): break
            # remaining = res._number_left
            # print "Waiting for", remaining, "task chunks to complete..."
            print("# castor.parallel.map : tasks accomplished out of {0} : {1}".format(n, v.get()))
            time.sleep(timesleep)

    pool.close()
    m.shutdown()

    return res.get(timeout)
#
