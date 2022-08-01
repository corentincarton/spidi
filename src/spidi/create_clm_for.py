## Generate a climatological seasonal forecast ensemble using the monitoring data

## Emanuel Dutra, November 2017

from __future__ import print_function

import numpy as np 
import sys
import eccodes as ec 

from spidi import core


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
      
  # testing: run create_clm_for.py  --AWDIR=/scratch/rd/need/dsuite/aaac/20170101 --SEASVER=5 --HINDYEND=2016 --HINDYSTART=1981 --YMD=20170101

  # 1. Load monitoring
  fnameMON="%s/MON_HIND.grb"%(AWDIR) 
  mon_hindF,mon_keys = core.load_grb_file("%s/%s.grb"%(AWDIR,core.MONHTAG),
                                          retKeys=['year','month'],verbose=False)

  ## Loop on forecast lead time
  FTEMPLATE=core.gen_for_fname(AWDIR,'BCFOR',SEASVER,YMD,'ENS')
  FOUTCLM=core.gen_for_fname(AWDIR,'CLMFOR',SEASVER,YMD,'ENS')
  fin = open(FTEMPLATE)
  fout = open(FOUTCLM,'w')
  while True:
    gid = ec.codes_grib_new_from_file(fin)
    if gid is None: break
    fN = ec.codes_get(gid,'number')
    
    if fN==0:
      fYR = ec.codes_get(gid,'year')
      fMN = ec.codes_get(gid,'month')
      fM = ec.codes_get(gid,'forecastMonth')
      
      clone_id = ec.codes_clone(gid)
      ec.codes_set(clone_id,'bitmapPresent',1)
      ec.codes_set(clone_id,'missingValue',core.ZMISS)
      
      Afmon=fMN+fM-1 # actual forecast month 
      Afyear=fYR     # actual forecast year
      yrA=0
      if Afmon > 12:
        Afmon = Afmon-12; Afyear=Afyear+1 ; yrA=1
      print(fN,fYR,fMN,fM,Afmon,Afyear)
      # loop on hindcast year searching for particular month
      numb=0
      for yr in range(int(HINDYSTART),int(HINDYEND)+1):
        if yr+yrA == Afyear: continue # skip same year 
        indME = np.nonzero((mon_keys['year'] == yr+yrA ) & (mon_keys['month'] == Afmon))[0][0]
        xtmp = np.ma.filled(np.ma.fix_invalid(mon_hindF[indME,:]),core.ZMISS)
        ec.codes_set_values(clone_id, xtmp)
        ec.codes_set(clone_id,'number',numb)
        ec.codes_write(clone_id, fout)
        print('  ',mon_keys['year'][indME],mon_keys['month'][indME],numb)
        numb=numb+1
      ec.codes_release(clone_id)
    ec.codes_release(gid)
  fin.close()
  fout.close()
  FOUTCLMENM=core.gen_for_fname(AWDIR,'CLMFOR',SEASVER,YMD,'ENM')
  core.compute_clim(FOUTCLM,FOUTCLMENM,'forecastMonth')


if __name__ == "__main__":
    main(sys.argv)