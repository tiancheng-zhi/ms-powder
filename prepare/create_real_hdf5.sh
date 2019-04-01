mkdir -p ../real/bg
mkdir -p ../real/bgext
python create_hdf5_bg.py
python create_hdf5_bgext.py
python create_hdf5_trainext.py
python create_hdf5_test.py
python create_hdf5_testext.py
python create_hdf5_val.py
