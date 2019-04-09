import multiprocessing
import time
from tqdm.auto import tqdm, trange

# def _map_f(args):
#     f, i, v = args
#     res = f(i)
#     v.value += 1
#     return res
#
# def map(func, iter, verbose=True, timesleep=15.0, timeout=None):
#     """
#     Maps the function `func` over the iterator `iter` in a multi-threaded way
#     using the multiprocessing package.
#
#
#     Parameters
#     ----------
#     func : function
#         func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled .
#     iter : iterator
#     verbose : bool
#         Whether to print messages while the computation is on-going (the default is True).
#     timesleep : float
#         Number of seconds between messages (the default is 15.0).
#     timeout : float
#         Number of seconds before cancelling the whole thing (the default is None).
#
#     Returns
#     -------
#     type
#         Result of the computation of func(iter).
#     """
#
#     pool = multiprocessing.Pool()
#     m = multiprocessing.Manager()
#     v = m.Value(int, 0)
#
#     inputs = ((func,i,v) for i in iter) #use a generator, so that nothing is computed before it's needed :)
#
#     res = pool.map_async(_map_f, inputs)
#
#     try :
#         n = len(iter)
#     except TypeError : # if iter is a generator
#         n = None
#
#     v_old = 0
#
#     if verbose :
#         # while (True):
#         #     if (res.ready()): break
#         #     print("# castor.parallel.map : tasks accomplished out of {0} : {1}\r".format(n, v.get()))
#         #     time.sleep(timesleep)
#         with tqdm(total=n, desc='# castor.parallel.map') as pbar:
#             while (True):
#                 time.sleep(timesleep)
#                 v_new = v.get()
#                 pbar.update(v_new-v_old)
#                 v_old = v_new
#                 if (res.ready()):
#                     v_new = v.get()
#                     pbar.update(v_new-v_old)
#                     break
#
#     pool.close()
#     m.shutdown()
#
#     return v_old, v_new, res.get(timeout)
#

def _map_f(args):
    f, i = args
    return f(i)

def map(func, iter, ordered=True):
    """
    Maps the function `func` over the iterator `iter` in a multi-threaded way
    using the multiprocessing package with a tqdm progress bar.


    Parameters
    ----------
    func : function
        func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled .
    iter : iterator
    ordered : bool
        Whether to use imap (default) or imap_unordered

    Returns
    -------
    type
        Result of the computation of func(iter).
    """

    pool = multiprocessing.Pool()

    inputs = ((func,i) for i in iter) #use a generator, so that nothing is computed before it's needed :)

    try :
        n = len(iter)
    except TypeError : # if iter is a generator
        n = None

    res_list = []

    if ordered:
        pool_map = pool.imap
    else:
        pool_map = pool.imap_unordered

    with tqdm(total=n, desc='# castor.parallel.map') as pbar:
        for res in pool_map(_map_f, inputs):
            try :
                pbar.update()
                res_list.append(res)
            except KeyboardInterrupt:
                pool.terminate()

    pool.close()
    pool.join()

    return res_list
#

def apply_ntimes(func, n, args, verbose=True, timeout=None):
    """
    Applies `n` times the function `func` on `args` (useful if, eg, `func` is partly random).

    Parameters
    ----------
    func : function
        func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled .
    n : int
    args : any
    timeout : int or float
        If given, the computation is cancelled if it hasn't returned a result before `timeout` seconds.

    Returns
    -------
    type
        Result of the computation of func(iter).
    """
    pool = multiprocessing.Pool()

    multiple_results = [pool.apply_async(func, args) for _ in range(n)]

    pool.close()

    return [res.get(timeout) for res in tqdm(multiple_results, desc='# castor.parallel.apply_ntimes')]
