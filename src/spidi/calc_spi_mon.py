# Compute SPI for the monitoring period

# Emanuel Dutra, November 2017


from __future__ import print_function

import numpy as np 
import datetime as dt
import sys
import traceback
import eccodes as ec 

from spidi import core

def save_gamma_params():
  FTEMPLATE="%s/%s.grb"%(AWDIR,MONHTAG)
  fin = open(FTEMPLATE)
  gid = ec.codes_grib_new_from_file(fin)
  clone_id = ec.codes_clone(gid)
  ec.codes_release(gid)
  ec.codes_set(clone_id,'bitmapPresent',1)
  ec.codes_set(clone_id,'missingValue',core.ZMISS)
  fin.close()
  ftags={0:'acoef',1:'bcoef',2:'pzero'}
  for ik in range(3): # loop on the 3 parameters 
    fnameOUT="%s/GFIT_SPI%i_%s_%s.grb"%(AWDIR,tscale,ftags[ik],MONHTAG)
    print("Writing to output:",fnameOUT)
    fout = open(fnameOUT,'w')
    for im in range(12):
      xtmp = np.ma.filled(np.ma.fix_invalid(GammaP[ik,im,:]),core.ZMISS)
      ec.codes_set_values(clone_id,xtmp)
      ec.codes_set(clone_id,'dataDate',int("2016%02i01"%(im+1)))
      ec.codes_write(clone_id, fout)
  ec.codes_release(clone_id)
  fout.close()
  return

def main(args=None):
  ##===================================
  ## Get required variables 
  OPT=core.get_opt(['AWDIR','SPITSCALE','HINDYSTART','HINDYEND'],args[1:])
  print(OPT)
  for key in OPT.keys():
    if OPT[key] is None:
      print('Variable: ',key,' is not defined')
      print('can be defined either in the calling environment: e.g export %s=value'%key)
      print('or in the command line as --%s=value'%key)
      print('Exiting')
      #sys.exit(-1)
    else:
      exec(key+'="'+OPT[key]+'"')


  # testing: run calc_spi_mon.py  --AWDIR=/disk1/data/work/dsuite/20160101/ --SPITSCALE=6 --HINDYEND=2016 --HINDYSTART=2007

  MONHTAG=core.MONHTAG
  tscale=int(SPITSCALE)


  ##=====================================
  ## Load MON HIND data 
  fnameMON="%s/%s.grb"%(AWDIR,MONHTAG) 
  mon_hindP,mon_keys = core.load_grb_file(fnameMON,retKeys=['year','month'],verbose=True)
  #mon_hindT=np.array([dt.datetime(yr,mon,1) for yr,mon in zip(mon_keys['year'],mon_keys['month'])])
  ntTOT,ngpTOT = mon_hindP.shape
  months_hind=np.array(mon_keys['month'])
  years_hind=np.array(mon_keys['year'])

  ## Set precip values bellow threshold to zero
  mon_hindP[mon_hindP< core.PminDAY ] = 0. 
  ## Accumulate precipitation for the specific time scale 
  xpreA=core.rolling_sum(mon_hindP,n=tscale,axis=0).astype(np.float32)
  months_hind[0:tscale-1]=9999  # set strange months in the beggining of accumulation so that the "nan" are not included in the fit 

  ## Do the fitting to the gamma function 
  GammaP=np.zeros((3,12,ngpTOT),dtype=np.float32) # Acoef,Bcoef,pzero

  for im in range(12):
    ttind = np.nonzero((months_hind == im+1) &
                      (years_hind >= int(HINDYSTART))&
                      (years_hind <= int(HINDYEND)) )[0]
    print('Fitting:tscale,month,samples:',tscale,im+1,len(ttind))
    xdata = xpreA[ttind,:]
    coef,q = core.fspi_fit(xdata,core.ZeroMax,-1)
    GammaP[0,im,:]=coef[0,:]
    GammaP[1,im,:]=coef[1,:]
    GammaP[2,im,:]=q.copy()

  ## save fitting parameters 
  save_gamma_params() 

  ## Apply transformation to spi 
  xspi = np.zeros(xpreA.shape,dtype=np.float32)
  ## loop on months
  for im in range(12):
    ttind = np.nonzero(months_hind == im+1)[0]
    xspi[ttind,:] = core.fspi_eval(xpreA[ttind,:],core.ZeroMax,
                                GammaP[0:2,im,:],GammaP[2,im,:])
    print("Computing SPI,tscale,calendar month:",tscale,im+1)

  ## write spi to output file (copy from precip...)
  FTEMPLATE="%s/%s.grb"%(AWDIR,MONHTAG)
  FOUTSPI="%s/SPI%i_%s.grb"%(AWDIR,tscale,MONHTAG)
  fin = open(FTEMPLATE)
  fout = open(FOUTSPI,'w')
  ikfld=0
  while 1:
    gid = ec.codes_grib_new_from_file(fin)
    if gid is None:
      break
    clone_id = ec.codes_clone(gid)
    ec.codes_set(clone_id,'bitsPerValue',12)
    ec.codes_set(clone_id,'bitmapPresent',1)
    ec.codes_set(clone_id,'missingValue',core.ZMISS)
    xtmp = np.ma.filled(np.ma.fix_invalid(xspi[ikfld,:]),core.ZMISS)
    ec.codes_set_values(clone_id, xtmp)
    ec.codes_write(clone_id, fout)
    ec.codes_release(clone_id)
    ec.codes_release(gid)
    ikfld=ikfld+1
  fin.close()
  fout.close()
  print(ikfld,' fields written to:',FOUTSPI)    


if __name__ == "__main__":
    main(sys.argv)
