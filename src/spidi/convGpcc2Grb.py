# Convert GPCC netcdf files to grib format 

# Emanuel Dutra, November 2017

from __future__ import print_function

import numpy as np 
import sys
import eccodes as ec 
from netCDF4 import Dataset,num2date
import datetime as dt 
import calendar 

from spidi import core


def main(args=None):

  OPT=core.get_opt(['IFILE','OFILE','YMDMIN'],args[1:])
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
  YMDMIN=int(YMDMIN)

  #run convGpcc2Grb  --IFILE=tmp.nc --OFILE=tmp.grb --YMDMIN=19790101 

  keys = {
          'dataDate': 19600101,
          'dataTime':0,
          'startStep': 0,
          'endStep': 12,
          'stepType': 'accum',
          'indicatorOfParameter': 228,
          'bitsPerValue':24,
          'bitmapPresent':1,
          'missingValue':core.ZMISS,
          'Ni':360,
          'Nj':180,
          'latitudeOfFirstGridPointInDegrees':89.5,
          'longitudeOfFirstGridPointInDegrees':-179.5,
          'latitudeOfLastGridPointInDegrees':-89.5,
          'longitudeOfLastGridPointInDegrees':179.5,
          'iScansNegatively':0,
          'jScansPositively':0,
          'jPointsAreConsecutive':0,
          'jDirectionIncrementInDegrees':1,
          'iDirectionIncrementInDegrees':1,
          'marsClass':'rd',
          'marsType':'fc',
          'marsStream':'moda',
          'experimentVersionNumber':1
      }


  ## Open input file
  nc = Dataset(IFILE,'r')
  ntI = len(nc.dimensions['time'])
  print(ntI)
  ## prepare grib output      
  sample_id = ec.codes_grib_new_from_samples("regular_ll_sfc_grib1")
  fout = open(OFILE, 'w')
  clone_id = ec.codes_clone(sample_id)
  for key in keys:
    #print(key)
    ec.codes_set(clone_id, key, keys[key])
    
  ## loop on time in input file 
  for ik in range(ntI):
    cunits=getattr(nc.variables['time'],'units')
    if "months since" in cunits:
      xtime=dt.datetime.strptime(cunits.split(' ')[2],"%Y-%m-%d")
      cdate=int(xtime.strftime("%Y%m%d"))
    elif "since" in cunits:
      xtime = num2date(nc.variables['time'][ik],cunits)
      cdate=int(xtime.strftime("%Y%m%d"))
    else:
      cdate=int(nc.variables['time'][ik])
    if ( cdate < YMDMIN ): continue
    
    mon=np.mod(cdate/100,100)
    yr=cdate/10000
    xx,ndays=calendar.monthrange(yr,mon)
    print(yr,mon,ndays)
    xtmp = np.ma.filled(np.ma.fix_invalid(nc.variables['p'][ik,:,:] / float(ndays)),core.ZMISS)
    ec.codes_set_values(clone_id,xtmp.ravel())
    ec.codes_set(clone_id,'dataDate',cdate)
    
    ec.codes_write(clone_id, fout)
    
  fout.close()
  nc.close()


if __name__ == "__main__":
    main(sys.argv)
