## Compute SPI seasonal forecasts 

# Emanuel Dutra, November 2017

from __future__ import print_function

import numpy as np 
import datetime as dt
import sys
import traceback
import eccodes as ec 

from spidi import core


def acc_precip(fyr,fmon,fclead,tscale,for_keys,for_hind,mon_keys,mon_hindF,
               kidia=None,kfdia=None,verbose=False):
      
  indHY = np.nonzero(for_keys['fdate']/10000 == fyr)[0][0] # year index in forecast hindcast array 
  indHLE = fclead # last lead index in forecast hindcast array 
  indHLS = np.maximum(0,indHLE-tscale) # start lead index in forecast hindcast array 
  
  fmonP = fmon-1
  fyrP = fyr+0
  if fmonP == 0:
    fmonP=12
    fyrP=fyr-1
    
  indME = np.nonzero((mon_keys['year'] == fyrP ) & (mon_keys['month'] == fmonP))[0][0] + 1
  indMS = np.minimum(indME,indME-tscale+fclead)
    
  lenF=for_hind[indHY,indHLS:indHLE,:,:].shape[0]
  lenM=mon_hindF[indMS:indME,:].shape[0]
  assert lenF+lenM == tscale , '# months do not match to spi time scale'
  if verbose:
    indMEp=np.minimum(indME,len(mon_keys['year'])-1)
    indMSp=np.minimum(indMS,len(mon_keys['year'])-1)
    print('Leadtime:',fclead,'Forecast year:',fyr,' Forecast Month:',fmon,' Tscale:',tscale)
    print('FOR:',indHY,for_keys['fdate'][indHY],indHLS,':',indHLE,'(',indHLE-indHLS,')')
    print('MON:',indMS,':',indME,'[',mon_keys['year'][indMSp],mon_keys['month'][indMSp],']'
                 ,'[',mon_keys['year'][indMEp],mon_keys['month'][indMEp],']',
                 '(',(indME-indMS),')','Ntfor:',lenF,'NtMon:',lenM)
    print('')
    
  return np.sum(for_hind[indHY,indHLS:indHLE,:,:],axis=0) + np.sum(mon_hindF[indMS:indME,kidia:kfdia],axis=0)

def load_hind(ystart,yend,kidia=None,kfdia=None):
  ##====================================
  ## 2. Load Hindcast data 
  #ystart=int(HINDYSTART)
  #yend=int(HINDYEND)
  nyearH=yend-ystart+1
  print('Loading hind',ystart,yend,kidia,kfdia)
  for yr in range(ystart,yend+1):
    ikY=yr-ystart
    fdate="%i%02i01"%(yr,fmon)
    fname=core.gen_for_fname(AWDIR,FTYPE,SEASVER,fdate,FORTYPE)
    print('Loading:',fname)
    xtmp,xkeys1 = core.load_hindY(fname,kidia=kidia,kfdia=kfdia)
    nleadF,nensF,ngpTOT = xtmp.shape
    if ( yr == ystart):
      xdata = np.zeros(( nyearH,nleadF,nensF,ngpTOT),dtype=np.float32)
      xkeys={}
      for kk in xkeys1.keys():
        xkeys[kk] = xkeys1[kk]
      xkeys['fdate']=[]
    xkeys['fdate'].append(xkeys1['dataDate'][0])
    xdata[ikY,:,:,:] = xtmp
  xkeys['fdate'] = np.array(xkeys['fdate'])
  return xdata,xkeys
  #sys.exit()
  
def save_gamma_params(GammaP):
  FTEMPLATE="%s/%s.grb"%(AWDIR,MONHTAG)
  fin = open(FTEMPLATE)
  gid = ec.codes_grib_new_from_file(fin)
  clone_id = ec.codes_clone(gid)
  ec.codes_release(gid)
  ec.codes_set(clone_id,'bitmapPresent',1)
  ec.codes_set(clone_id,'missingValue',core.ZMISS)
  fin.close()
  ftags={0:'acoef',1:'bcoef',2:'pzero'}
  nleadF,ii,ii = GammaP.shape
  for ik in range(3): # loop on the 3 parameters 
    fnameOUT="%s/GFIT_SPI%i_%s_%s_%s_%02i.grb"%(AWDIR,tscale,ftags[ik],FTYPE,FORTYPE,fmon)
    print("Writing to output:",fnameOUT)
    fout = open(fnameOUT,'w')
    for im in range(nleadF):
      xtmp = np.ma.filled(np.ma.fix_invalid(GammaP[im,ik,:]),core.ZMISS)
      ec.codes_set_values(clone_id,xtmp)
      ec.codes_set(clone_id,'dataDate',int("2016%02i01"%(im+1)))
      ec.codes_write(clone_id, fout)
  ec.codes_release(clone_id)
  fout.close()
  return

def fit_hind():

  ##===================================
  ## 1. Load monitoring 
  mon_hindF,mon_keys = core.load_grb_file("%s/%s.grb"%(AWDIR,MONHTAG),
                                        retKeys=['year','month'],verbose=False)
  ## Set precip values bellow threshold to zero
  mon_hindF[mon_hindF< core.PminDAY ] = 0. 
  ntHIND,ngpTOT = mon_hindF.shape

  ## Compute domain partitioning
  nslots = ngpTOT/npMAX + 1
  pslots = np.floor(np.linspace(0,ngpTOT,nslots+1)).astype(np.int)

  for iks in range(nslots):
    kidia = pslots[iks]
    kfdia = pslots[iks+1]

    ###=====================================
    ### Load MON HIND data 
    #kidia=None
    #kfdia=None
    for_hind,for_keys=load_hind(ystartH,yendH,kidia,kfdia)
    nyear,nleadF,nens,ngpF = for_hind.shape
    ## Set precip values bellow threshold to zero
    for_hind[for_hind< core.PminDAY ] = 0. 

    ##====================================
    ## Allocate GamaP array
    if iks == 0:
      GammaP=np.zeros((nleadF,3,ngpTOT),dtype=np.float32)-99999.


    ##=======================================
    ## Accumulate precipitation for a specific lead time and spi time-scale

    # Main loop on lead time 

    for ilead,fclead in enumerate(for_keys['forLead']):
      xprecA = np.zeros((nyearH,nens,ngpF))
      print('Fitting lead time',ilead)
      for iy,fyr in enumerate(range(ystartH,yendH+1)):
        xprecA[iy,:,:] = acc_precip(fyr,fmon,fclead,tscale,for_keys,for_hind,mon_keys,mon_hindF,
                                    kidia,kfdia,verbose=False)
      ## do the gamma fitting
      coef,q = core.fspi_fit(xprecA.reshape(nyearH*nens,ngpF),core.ZeroMax,-1)
      GammaP[ilead,0,kidia:kfdia]=coef[0,:]
      GammaP[ilead,1,kidia:kfdia]=coef[1,:]
      GammaP[ilead,2,kidia:kfdia]=q.copy()

  #save gamma fit parameters     
  save_gamma_params(GammaP)
  
def compute_spi():
  ##===================================
  ## 1. Load monitoring 
  mon_hindF,mon_keys = core.load_grb_file("%s/%s.grb"%(AWDIR,MONHTAG),
                                        retKeys=['year','month'],verbose=False)
  ## Set precip values bellow threshold to zero
  mon_hindF[mon_hindF< core.PminDAY ] = 0. 
  ntHIND,ngpTOT = mon_hindF.shape

  ###=====================================
  ### 2 . Load MON HIND data 
  kidia=None
  kfdia=None
  for_hind,for_keys=load_hind(fyear,fyear,kidia,kfdia)
  nyear,nleadF,nens,ngpF = for_hind.shape
  ## Set precip values bellow threshold to zero
  for_hind[for_hind< core.PminDAY ] = 0. 

  ##======================================0
  ### 3. Load Gamma Coefs
  GammaP=np.zeros((nleadF,3,ngpTOT),dtype=np.float32)-99999.
  ftags={0:'acoef',1:'bcoef',2:'pzero'}
  for ik in range(3): # loop on the 3 parameters 
    fname="%s/GFIT_SPI%i_%s_%s_%s_%02i.grb"%(AWDIR,tscale,ftags[ik],FTYPE,FORTYPE,fmon)
    xtmp,xkeys = core.load_grb_file(fname,retKeys=['year','month'],verbose=True)
    GammaP[:,ik,:] = xtmp
    

  ##=======================================
  # Main loop on lead time and write output 
  FTEMPLATE=core.gen_for_fname(AWDIR,FTYPE,SEASVER,YMD,FORTYPE)
  FOUTSPI=core.gen_for_fname(AWDIR,'SPI_%i_'%tscale,FTYPE,YMD,FORTYPE)
  fin = open(FTEMPLATE)
  fout = open(FOUTSPI,'w')
  print('Template from:',FTEMPLATE)
  print('Writting to:',FOUTSPI)
  for ilead,fclead in enumerate(for_keys['forLead']):
    print('Computing/writing lead time',fclead)
    xprecA = acc_precip(fyear,fmon,fclead,tscale,for_keys,for_hind,mon_keys,mon_hindF,
                                  kidia,kfdia,verbose=True)
    xspi = core.fspi_eval(xprecA,core.ZeroMax,
                        GammaP[ilead,0:2,:],GammaP[ilead,2,:])
    for imemb in range(nens):
      gid = ec.codes_grib_new_from_file(fin)
      fM = ec.codes_get(gid,'forecastMonth')
      eM = ec.codes_get(gid,'number')
      #print(ilead,imemb,fM,eM)
      clone_id = ec.codes_clone(gid)
      xdata = np.ma.filled(np.ma.fix_invalid(xspi[imemb,:]),core.ZMISS)
      ec.codes_set(clone_id,'bitsPerValue',12)
      ec.codes_set(clone_id,'bitmapPresent',1)
      ec.codes_set(clone_id,'missingValue',core.ZMISS)
      ec.codes_set_values(clone_id, xdata)
      ec.codes_write(clone_id, fout)
      ec.codes_release(clone_id)
      ec.codes_release(gid)
  fin.close()
  fout.close()


def main(args=None):

  ##===================================
  ## Get required variables 
  OPT=core.get_opt(['AWDIR','SPITSCALE','HINDYSTART','HINDYEND','FORTYPE','SEASVER','YMD','CONFIG','FTYPE'],args[1:])
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

  # testing: run calc_spi_for.py  --AWDIR=/disk1/data/work/dsuite/20160101/ --SPITSCALE=6 --HINDYEND=2016 --HINDYSTART=2007 --FORTYPE=ENS --SEASVER=5 --YMD=20160101 --CONFIG=fit_hind 
  #          run calc_spi_for.py  --AWDIR=/disk1/data/work/dsuite/20160101/ --SPITSCALE=6 --HINDYEND=2016 --HINDYSTART=2007 --FORTYPE=ENS --SEASVER=5 --YMD=20160101 --CONFIG=compute_spi

  ## generic / computed variables used at some point 
  MONHTAG=core.MONHTAG
  tscale=int(SPITSCALE)

  ystartH = int(HINDYSTART)
  yendH = int(HINDYEND)
  nyearH=yendH-ystartH+1
  fmon=int(YMD[4:6]) # Forecast start Month
  fyear=int(YMD[0:4])

  # max number of points nproma to avoid using too much RAM memory 
  npMAX=500000  
  if FORTYPE == "ENS":
    npMAX=50000
  if FORTYPE == "ENM":
    npMAX=500000
    
    
  if ( CONFIG == 'fit_hind') : 
    fit_hind()      
  elif (CONFIG == 'compute_spi'):
    compute_spi()
  else:
    print('Configuration requested not available!',CONFIG)
    sys.exit(-1)


if __name__ == "__main__":
    main(sys.argv)