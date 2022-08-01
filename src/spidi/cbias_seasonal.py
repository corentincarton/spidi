# compute and apply mean bias correction to seasonal forecasts 

# Emanuel Dutra, November 2017

from __future__ import print_function

import numpy as np 
import sys
import eccodes as ec 

from spidi import core


def compute_hind_climate():
  ystart=int(HINDYSTART)
  yend=int(HINDYEND)
  fmon=YMD[4:6]
  for yr in range(ystart,yend+1):
    fdate="%i%s01"%(yr,fmon)
    fnameENS=core.gen_for_fname(AWDIR,'FOR',SEASVER,fdate,'ENS')
    fnameENM=core.gen_for_fname(AWDIR,'FOR',SEASVER,fdate,'ENM')
    # compute ensemble mean:
    core.compute_clim(fnameENS,fnameENM,'forecastMonth')
    # load ensemble mean
    xtmp,xkeys = core.load_grb_file(fnameENM,retKeys=['dataDate','forecastMonth'],verbose=True)
    if yr == ystart:
      xdata = xtmp*0.
    xdata = xdata + xtmp 
  xdata = xdata*1./(yend-ystart+1)
  return xdata,xkeys

def main(args=None):

  ##==========================================
  ## Get variables from OSENV

  ##===================================
  ## Get required variables 
  OPT=core.get_opt(['AWDIR','SEASVER','HINDYSTART','HINDYEND','YMD'],args[1:])
  print(OPT)
  for key in OPT.keys():
    if OPT[key] is None:
      print('Variable: ',key,' is not defined')
      print('can be defined either in the calling environment: e.g export %s=value'%key)
      print('or in the command line as --%s=value'%key)
      print('Exiting')
      sys.exit(-1)
    else:
      exec(key+'="'+OPT[key]+'"')

  # testing: run cbias_seasonal.py  --AWDIR=/disk1/data/work/dsuite/20160101/ --SEASVER=5 --HINDYEND=2016 --HINDYSTART=2007 --YMD=20160101

  ##=====================================
  ## Load data 

  # 1: hindcast data mean climate 
  hind_climate,hind_keys = compute_hind_climate()
  nlead,npp = hind_climate.shape

  # compute ensemble mean of actual forecasts, required latter 
  fnameENS=core.gen_for_fname(AWDIR,'FOR',SEASVER,YMD,'ENS')
  fnameENM=core.gen_for_fname(AWDIR,'FOR',SEASVER,YMD,'ENM')
  core.compute_clim(fnameENS,fnameENM,'forecastMonth')

  # 2: Monitoring mean climate
  fnameMON="%s/MON_HIND.grb"%(AWDIR) 
  fnameMONC="%s/MON_HIND_CLIM.grb"%(AWDIR)
  core.compute_clim(fnameMON,fnameMONC,'month',
                  extraKeys=['year',],
                  extraKeysLimits={'year':[int(HINDYSTART),int(HINDYEND)]})
  mon_climate,mon_keys = core.load_grb_file(fnameMONC,retKeys=['month'],verbose=True)

  ##=====================================
  ## Compute multiplicative correction factor 
  mfact = np.zeros((nlead,npp))
  for il in range(nlead):
    icm = core.add_months(YMD[4:6],hind_keys['forecastMonth'][il]-1)
    indMON=icm-1
    print('Forecast Month,calendar month,monitoring climate month:',il,icm,mon_keys['month'][indMON])
    mfact[il,:] = np.maximum( 0.2, np.minimum( 5. ,
            ( np.maximum(core.PminDAY,mon_climate[indMON,:]) / 
            np.maximum(core.PminDAY,hind_climate[il,:]) ) ) )

  mfact = np.ma.fix_invalid(mfact)          
  ##=======================================
  ## Save multiplicative factor 
  FTEMPLATE=core.gen_for_fname(AWDIR,'FOR',SEASVER,YMD,'ENM')
  FOUTMF=core.gen_for_fname(AWDIR,'BCfFOR',SEASVER,YMD[4:6],'ENM')
  print('Writing Mfactor to:',FOUTMF)
  fin = open(FTEMPLATE)
  fout = open(FOUTMF,'w')
  for il in range(nlead):
    gid = ec.codes_grib_new_from_file(fin)
    fM = ec.codes_get(gid,'forecastMonth')
    print(il,fM)
    clone_id = ec.codes_clone(gid)
    ec.codes_set(clone_id,'bitmapPresent',1)
    ec.codes_set(clone_id,'missingValue',core.ZMISS)
    xtmp = np.ma.filled(np.ma.fix_invalid(mfact[il,:]),core.ZMISS)
    ec.codes_set_values(clone_id, xtmp)
    ec.codes_write(clone_id, fout)
    ec.codes_release(clone_id)
    ec.codes_release(gid)
  fin.close()
  fout.close()


  ##=========================================
  ## Apply bias correction factor to all fields 
  ystart=int(HINDYSTART)
  yend=int(HINDYEND)
  fmon=YMD[4:6]
  ## loop on hindcast years + actual forecast 
  for yr in np.unique(range(ystart,yend+1)+[int(YMD[0:4])]):
    fdate="%i%s01"%(yr,fmon)
    FTEMPLATE=core.gen_for_fname(AWDIR,'FOR',SEASVER,fdate,'ENS')
    FOUTBC=core.gen_for_fname(AWDIR,'BCFOR',SEASVER,fdate,'ENS')
    print('Processing:',FTEMPLATE)
    fin = open(FTEMPLATE)
    fout = open(FOUTBC,'w')
    ikfld=0
    while 1:
      gid = ec.codes_grib_new_from_file(fin)
      if gid is None:
        break
      xdata = ec.codes_get_values(gid)
      fM = ec.codes_get(gid,'forecastMonth')
      eM = ec.codes_get(gid,'number')

      # apply correction
      xdata = xdata*mfact[fM-1,:]
      clone_id = ec.codes_clone(gid)
      ec.codes_set(clone_id,'bitmapPresent',1)
      ec.codes_set(clone_id,'missingValue',core.ZMISS)
      xtmp = np.ma.filled(np.ma.fix_invalid(xdata),core.ZMISS)
      ec.codes_set_values(clone_id, xtmp)
      ec.codes_write(clone_id, fout)
      ec.codes_release(clone_id)
      ec.codes_release(gid)
      ikfld=ikfld+1
    fin.close()
    fout.close()
    print(ikfld,' fields written to:',FOUTBC)
    # compute ensemble mean
    FOUTBCENM=core.gen_for_fname(AWDIR,'BCFOR',SEASVER,fdate,'ENM')
    core.compute_clim(FOUTBC,FOUTBCENM,'forecastMonth')


if __name__ == "__main__":
    main(sys.argv)
