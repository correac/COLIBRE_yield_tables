COLIBRE yield tables
=========

Python package that produces yield tables in the hdf5 format used by the swift/COLIBRE code.

Requirements
------------

The colibre yield tables package requires:

+ `python3.6` or above
+ see requirements.txt

Installing
----------

To get started using the package you need to set up a python virtual environment. The steps are as follows:

Clone repository
```
git clone https://github.com/correac/COLIBRE_yield_tables.git

cd COLIBRE_yield_tables

python3 -m venv colibre_tables_env
```

Now activate the virtual environment.

```
source colibre_tables_env/bin/activate
```

Update pip just in case
```
pip install pip --upgrade

pip install -r requirements.txt
```

How to use it
-------------

Simply type:

```
python yield_tables.py
```

to create yield tables for AGB stars, SNIa and CC-SNe. These will appear in the data folder. Additionally, plots showing the stellar yields as a function of zero-age main sequence mass will be produced in the folder figures.
