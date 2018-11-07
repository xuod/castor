# castor -- a python library for stuff
# @author: Cyrille Doux <cdoux@sas.upenn.edu>
#
# castor/__init__.py -- wraps castor in a bow.
#

import parallel, misc, plot, maths, cosmo

import numpy

def call_item_by_item(func):
    """
    Decorator for a function such that an array passed to
    it will be executed item by item.  Return value is the
    same type as the input value (list,ndarray,matrix,etc).

    also up-casts integers.

    # From https://github.com/jakevdp/Thesis/blob/master/shear_KL/shear_KL_source/cosmology/cosmo_tools.py
    """
    def new_func(self,val,*args,**kwargs):
        if type(val) in (int,long):
            val = float(val)
        v_array = numpy.asarray(val)
        v_raveled = v_array.ravel()
        retval = numpy.array([func(self,v,*args,**kwargs) for v in v_raveled],
                             dtype = v_array.dtype)
        retval.resize(v_array.shape)
        if type(val)==numpy.ndarray:
            return retval
        else:
            return type(val)(retval)
    return new_func

def call_as_array(func):
    """
    Decorator for a function such that an array passed to
    it will be executed in one step.  Return value is the
    same type as the input value (float,list,ndarray,matrix,etc).

    also up-casts integers.

    From # https://github.com/jakevdp/Thesis/blob/master/shear_KL/shear_KL_source/cosmology/cosmo_tools.py
    """
    def new_func(self,val,*args,**kwargs):
        if type(val) in (int,long):
            val = float(val)
        v_array = numpy.asarray(val)
        v_raveled = v_array.ravel()
        retval = func(self,v_raveled,*args,**kwargs)
        numpy.asarray(retval).resize(v_array.shape)
        if type(val)==numpy.ndarray:
            return retval
        else:
            return type(val)(retval)
    return new_func
