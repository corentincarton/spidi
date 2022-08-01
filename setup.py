from setuptools import setup, find_packages
import os

setup(
    name="spidi",
    version='0.1',
    author='ECMWF',
    description="ECMWF Standardized Precipitation Index and other Drought Indices library",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'python-eccodes'
    ],
    tests_require=[
    ],
    entry_points={
        "console_scripts": [
            "spidi-spi-for=spidi.calc_spi_for:main",
            "spidi-spi-mon=spidi.calc_spi_mon:main",
            "spidi-cbias-seasonal=spidi.cbias_seasonal:main",
            "spidi-gpcc2grib=spidi.convGpcc2Grb:main",
            "spidi-clim-for=spidi.create_clm_for:main",
        ],
    },
)
