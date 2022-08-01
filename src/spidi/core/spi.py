#
# Set of generic functions used by different python scripts in the dsuite 
#
# Emanuel Dutra, November 2017

from __future__ import print_function

import numpy as np 
import os
import sys
import scipy.stats as ss 
import scipy.special as sps
import eccodes as ec

## Generic variables 
PminDAY = 0.03  # minimum precipitation thrshold mm/day == 0.9 mm/month
ZeroMax = 1./3. # maximum frequency of zero to be accepted in the gamma fit 
MONHTAG="MON_HIND"
ZMISS=-99 # default missing value for grib encoding 


def load_hindY(fname,kidia=None,kfdia=None):
  """
  Load singe hindcast file and organize
  """
  xtmp,xkeys = load_grb_file(fname,retKeys=['dataDate','forecastMonth','number'],verbose=False)
  nfld,ngpTOT =xtmp.shape 
  forLead = np.sort(np.unique(xkeys['forecastMonth']))
  forENB = np.sort(np.unique(xkeys['number']))
  nleadF = len(forLead)
  nens = len(forENB)
  assert nleadF*nens == nfld ,"Number of fields in forecast files does not match nleadF*nens !"  
  xdata = np.zeros((nleadF,nens,ngpTOT),dtype=np.float32)
  for ilead,xlead in enumerate(forLead):
    for imemb,xensN in enumerate(forENB):
      ifld = np.nonzero((xkeys['number'] == xensN) & (xkeys['forecastMonth'] == xlead))[0]
      #print(xlead,xensN,ifld)
      xdata[ilead,imemb,:] = xtmp[ifld,:]
  xkeys['forLead']=forLead
  xkeys['forENB']=forENB
  return xdata[:,:,kidia:kfdia],xkeys


def gen_arg(args=[''],dlft=None):
  opt={}
  for i,arg in enumerate(args):
    if arg.startswith('--'):
      try:
        key_arg = arg[2:].split("=")
        opt[key_arg[0]] = key_arg[1]
      except:
        pass
  return opt       
  
def getENV(key,dflt=None,fail=True):
  """
  Get environment variable from OS
  """
  cvar = os.getenv(key,default=dflt)
  if fail and (cvar is None):
    print('Failed to get env. variable:',key)
    sys.exit(-1)
  return cvar 

def get_opt(OPTL=[''],OPTSS=['']):
  OPT={}
  for key in OPTL:
    OPT[key] = getENV(key,fail=False)
  #print(OPT)
  OPTS=gen_arg(OPTSS)
  #Override enviromnet by command line
  for key in OPT.keys():
    if key in OPTS:
      OPT[key] = OPTS[key]
  return OPT
  
def fspi_eval(D,zeromax,coef,q):
  #xspi = spi.fspi_eval(xprecTMP,coef,q)
  nt,ngp = D.shape
  xspi = np.zeros((nt,ngp))*np.nan


  for ip in xrange(ngp):
    if q[ip] > zeromax or np.isnan(coef[1,ip]) :
      continue
    xspi[:,ip] = q[ip]+(1.-q[ip])*ss.gamma.cdf(D[:,ip],coef[0,ip],scale=coef[1,ip])
  xspi[xspi < 0.001 ] = 0.001
  xspi[xspi > 0.999 ] = 0.999
  xspi[:,:] = ss.norm.ppf(xspi)

  xspi[np.isnan(D)]=np.nan
  return xspi

def fspi_fit(D,zeromax,dbg=-1):

  nt,ngp = D.shape
  coef = np.zeros((2,ngp))*np.nan
  q = np.zeros((ngp))*np.nan

  q = (nt - np.sum(D>0.,axis=0)) / float(nt)
  pp = np.nonzero(q<=zeromax)
  coef[0,pp[0]],coef[1,pp[0]] = fitgamma(D[:,pp[0]])
  pp = np.nonzero(coef[0,:] > 1000 )
  #print("Error in spi.fspi_fit:",len(pp[0]),ngp)
  coef[0,pp[0]] = np.nan
  coef[1,pp[0]] = np.nan

  if dbg >= 0:
    import matplotlib.pyplot as plt
    ip = dbg
    prob = q[ip]+(1.-q[ip])*ss.gamma.cdf(D[:,ip],coef[0,ip],scale=coef[1,ip])
    plt.plot(np.sort(D[:,ip]),np.arange(0,nt)/float(nt),'or')
    plt.plot(D[:,ip],prob,'.b')
    plt.show()

  return coef,q

def fitgamma ( samples ): 
  """fit a gamma distribution using maximum likelihood 
   http://psignifit.sourceforge.net/api/pypsignifit.psigsimultaneous-pysrc.html#fitgamma
     Parameters 
     ---------- 
     samples : array 
         array of samples on which the distribution should be fitted 
  
     Returns 
    ------- 
    prm : sequence 
        pair of k (shape) and theta (scale) parameters for the fitted gamma distribution 
   
  """
  #np.seterr(invalid='raise')
  nt,ngp = samples.shape
  xmean = np.zeros((ngp))
  xmeanl = np.zeros((ngp))
  for ip in xrange(ngp):
    xx = samples[samples[:,ip]>0.,ip]
    xmean[ip] = np.mean(xx)
    xmeanl[ip] = np.mean( np.log(xx))

  s = np.log ( xmean ) - xmeanl
  k = 3 - s + np.sqrt ( (s-3)**2 + 24*s)
  k /= 12 * s
  
  ppbad = np.nonzero(s==0)[0]
  ppbad = np.concatenate((ppbad,np.nonzero(k<=0)[0]))
  k[ppbad]=7.8
  s[ppbad]=0.06

  for i in xrange ( 5 ):
    k -= ( np.log(k) - sps.digamma ( k ) -s ) / ( 1./k - sps.polygamma( 1, k ) )
    ppbad = np.concatenate((ppbad,np.nonzero(k<=0)[0]))
    k[ppbad]=7.8
  th = xmean / k 
  
  if len(ppbad)> 0:
    print(" spi.fitgamma: mle failed,",np.unique(ppbad))
  k[ppbad]=np.nan
  th[ppbad]=np.nan

  return k,th

def rolling_sum(a, n=None,axis=0) :
  """
  Compute rolling sum 
  Addapted from:
  https://stackoverflow.com/questions/28288252/fast-rolling-sum
  """
  ret = np.cumsum(a, axis=axis, dtype=float)
  if axis==1:
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    ret[:,0:n-1]=np.nan
    return ret
  elif axis==0:
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    ret[0:n-1,:]=np.nan
    return ret


def compute_clim(FIN,FOUTN,key,extraKeys=None,extraKeysLimits={}):
  """
  Compute mean of all fields in FIN matching unique values of "key"
  Output saved into FOUTN. 
  """
  #FIN=core.gen_for_fname(AWDIR,'FOR',SEASVER,YMD,'ENS')
  #FOUTN=core.gen_for_fname(AWDIR,'FOR',SEASVER,YMD,'ENM1')
  #key='forecastMonth'
  
  print('Compute_clim:',FIN,key)
  fout = open(FOUTN,'w')
  iid = ec.codes_index_new_from_file(FIN, [key])
  key_vals = list(ec.codes_index_get(iid,key))
  key_vals.sort(key=int)
  for val in key_vals:
    ec.codes_index_select(iid, key, val)
    ikf=0
    while 1:
      gid = ec.codes_new_from_index(iid)
      if gid is None:
        break
      xtmp = ec.codes_get_values(gid)
      lpresent=True
      if extraKeys is not None:
        for kk in extraKeys:
          kval = ec.codes_get(gid,kk)
          if not (extraKeysLimits[kk][0] <= kval <= extraKeysLimits[kk][1] ): lpresent=False
      if lpresent:
        if ikf == 0 :
          xdata = np.zeros(xtmp.shape)
          clone_id = ec.codes_clone(gid)
        xdata = xdata+xtmp
        ikf=ikf+1
      #print(key,val,lpresent,ec.codes_get(gid,'number'))
      ec.codes_release(gid)
    if ikf > 0:
      xdata = xdata /float(ikf)
      #print('compute_clim:',val,ikf)
      ec.codes_set_values(clone_id, xdata)
      ec.codes_write(clone_id, fout)
      ec.codes_release(clone_id)
    print(key,val,ikf)
    
  ec.codes_index_release(iid)
  fout.close()
  print("File created:",FOUTN)

def add_months(m1,m2):
  """"
  Simple function to add add months
  """
  rmonth = int(m1)+int(m2)
  if rmonth>12:
    rmonth=rmonth-12
  return rmonth

def gen_for_fname(AWDIR,FTAG,SEASVER,YMD,FCONT):
  """"
  Generate file forecast file name in the form:
  "%s/%s%s.%s.%s.grb"%(AWDIR,FTAG,SEASVER,YMD,FCONT)
  """
  return "%s/%s%s.%s.%s.grb"%(AWDIR,FTAG,SEASVER,YMD,FCONT)

def load_grb_file(FNAME,retKeys=None,verbose=False):
  """
  Loads full grib file 
  xdata,xKeys=load_for_file(FNAME,retKeys=None,verbose=False)
  input:
   FNAME: file name (including full path) to read from 
   retKeys: list with extra keys to return (default: None)
   verbose: if true print some details 
  returns
   xdata : np.array: (nflds,npoints)
   xKeys : if retKeys is not None: dictionary with a list for each key requested 
  """
  
  if verbose:
    print('Reading:',FNAME)
  # open file 
  fgrb = open(FNAME)
  # find # of fields in file 
  nflds = ec.codes_count_in_file(fgrb)
  if verbose:
    print('Found ', nflds,' fields in,',FNAME)

  # create dictionary with empty lists for each requested key 
  xKeys={}
  if retKeys is not None:
    for key in retKeys:
      xKeys[key]=[]

  # load fields    
  for ikfld in range(nflds):
    gid = ec.codes_grib_new_from_file(fgrb)
    xtmp = ec.codes_get_values(gid)
    if ec.codes_get(gid,'bitmapPresent') == 1:
      zmiss=ec.codes_get(gid,'missingValue')
      xtmp[xtmp==zmiss]=np.nan
    if ikfld == 0  : 
      xdata = np.zeros((nflds,xtmp.shape[0]),dtype=np.float32)
    xdata[ikfld,:] = xtmp 
    if retKeys is not None:
      for key in retKeys:
        xKeys[key].append(ec.codes_get(gid,key))
    
    ec.codes_release(gid)
    
  fgrb.close()
  if retKeys is not None:
    for kk in xKeys.keys():
      xKeys[kk] = np.array(xKeys[kk])
    return xdata,xKeys
  else:
    return xdata
  
