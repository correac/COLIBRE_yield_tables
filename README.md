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

Type 
```
python yield_tables.py
```
