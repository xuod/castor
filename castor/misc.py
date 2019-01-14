from astropy.io import fits

def fitsdata(filename, hdu_number=1):
    """
    Open a fits file and get the table of HDUImage hdu_nuber

    """
    file = fits.open(filename)

    return file[hdu_number].data
#

def funcdecl2gobjdoc(s):
    """
    Returns the name of the function and its arguments formatted for GObject documentation

    Example
    -------
    >>> funcdecl2gobjdoc("void nc_xcor_limber_prepare (NcXcorLimber* xcl, NcXcorLimber* xcl2, NcHICosmo* cosmo)")
    "/**\
     * nc_xcor_limber_prepare:\
     * @xcl: a NcXcorLimber\
     * @cosmo: a NcHICosmo\
     *\
     * FIXME\
     *\
     * Returns: FIXME\
     */"

    """
    res = "/**\n * "
    i = s.find(" ")
    returntype = s[:i]
    if returntype[-1] == '*':
        returntype = returntype[:-2]
    # print returntype
    s = s[i+1:]
    # print s
    i = s.find("(")
    funcname = s[:i]
    # print funcname
    res += funcname + ":\n * "
    s = s[i:]
    i = s.find("(")
    ii = s.find(")")
    s = s[i+1:ii]
    args = s.split(", ")
    print(s)
    for arg in args:
        if arg[:6] == "const ":
            arg = arg[6:]
        i = arg.find(" ")
        argtype = arg[:i].strip('*')
        # print argtype
        argname = arg[i+1:]
        res += "@" + argname + ": a #" + argtype + "\n * "
    res += "\n * FIXME\n * \n * Returns: FIXME \n * \n*/"
    print(res)
#


def create_fits(fitsname, var, header=None):
    """
    Writes the variables var to a FITS file (works for float variables !).

    `var` can be a dictionnary or a numpy.recarray.

    """
    if type(var) is dict :
        col = []
        for k, v in var :
            col.append(fits.Column(name=k,format='E',array=v))
        cols=fits.ColDefs(col)

        tbhdu=fits.TableHDU.from_columns(cols, header=header)

    if type(var) is numpy.recarray:
        tbhdu=fits.TableHDU.from_columns(var, header=header)

    tbhdu.writeto(fitsname, clobber=True)
#


def send_email(subject='Howdy !', message='This is Python talking to you !', to_addr_list=[], cc_addr_list=[]):
    """
    Send an email from dummypython address :)

    from http://stackoverflow.com/questions/10147455/how-to-send-an-email-with-gmail-as-provider-using-python

    """
    import os,smtplib

    header  = 'From: %s\n'%os.environ['CASTOREMAIL']
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(os.environ['CASTOREMAIL'],os.environ['CASTORPASS'])
    problems = server.sendmail(os.environ['CASTOREMAIL'], to_addr_list, message)
    server.quit()
#

##############################
# Use the tqdm package instead
##############################

# import sys, time
# try:
#     from IPython.core.display import clear_output
#     have_ipython = True
# except ImportError:
#     have_ipython = False
#
# class ProgressBar:
#     def __init__(self, iterations):
#         self.iterations = iterations
#         self.prog_bar = '[]'
#         self.fill_char = '*'
#         self.width = 40
#         self.__update_amount(0)
#         if have_ipython:
#             self.animate = self.animate_ipython
#         else:
#             self.animate = self.animate_noipython
#
#     def animate_ipython(self, iter):
#         try:
#             clear_output()
#         except Exception:
#             # terminal IPython has no clear_output
#             pass
#         print '\r', self,
#         sys.stdout.flush()
#         self.update_iteration(iter + 1)
#
#     def update_iteration(self, elapsed_iter):
#         self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
#         self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)
#
#     def __update_amount(self, new_amount):
#         percent_done = int(round((new_amount / 100.0) * 100.0))
#         all_full = self.width - 2
#         num_hashes = int(round((percent_done / 100.0) * all_full))
#         self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
#         pct_place = (len(self.prog_bar) / 2) - len(str(percent_done))
#         pct_string = '%d%%' % percent_done
#         self.prog_bar = self.prog_bar[0:pct_place] + \
#             (pct_string + self.prog_bar[pct_place + len(pct_string):])
#
#     def __str__(self):
#         return str(self.prog_bar)
# #
#
#
# import bitstring
#
# class bitflag:
#     """
#     Class to make bit-wise flags
#
#     """
#     def __init__(self):
#         self.counter = 0
#         self.bitlist = []
#
#     def add(self, test):
#         self.counter += 1
#         self.bitlist.append(int(test))
#         return test
#
#     def get(self):
#         if self.counter > 0 :
#             b = bitstring.BitArray(self.bitlist[::-1])
#             return b.uint
#         else :
#             return 0
# #
#
#
# def log_progress(sequence, every=None, size=None):
#     """
#     Progress bar compatible with Jupyter notebooks (from https://github.com/alexanderkuk/log-progress)
#
#     Usage :
#
#     for truc in log_pregress(trucs, every=nbr_de_trucs(None), size=nbr_total_de_trucs(None)):
#         do stuff...
#
#     Rmq : si trucs est un iterable, preciser every ou size
#
#     """
#     from ipywidgets import IntProgress, HTML, VBox
#     from IPython.display import display
#
#     is_iterator = False
#     if size is None:
#         try:
#             size = len(sequence)
#         except TypeError:
#             is_iterator = True
#     if size is not None:
#         if every is None:
#             if size <= 200:
#                 every = 1
#             else:
#                 every = size / 200     # every 0.5%
#     else:
#         assert every is not None, 'sequence is iterator, set every'
#
#     if is_iterator:
#         progress = IntProgress(min=0, max=1, value=1)
#         progress.bar_style = 'info'
#     else:
#         progress = IntProgress(min=0, max=size, value=0)
#     label = HTML()
#     box = VBox(children=[label, progress])
#     display(box)
#
#     index = 0
#     try:
#         for index, record in enumerate(sequence, 1):
#             if index == 1 or index % every == 0:
#                 if is_iterator:
#                     label.value = '{index} / ?'.format(index=index)
#                 else:
#                     progress.value = index
#                     label.value = u'{index} / {size}'.format(
#                         index=index,
#                         size=size
#                     )
#             yield record
#     except:
#         progress.bar_style = 'danger'
#         raise
#     else:
#         progress.bar_style = 'success'
#         progress.value = index
#         label.value = str(index or '?')
