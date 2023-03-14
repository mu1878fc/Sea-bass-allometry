# Sea-bass-allometry

# These data are an integrant part of our paper "Morphological traits for allometric scaling of the European Sea Bass Dicentrarchus labrax (Linnaeus, 1758) from Portugal"
# Azevedo, A., Navarro, L.C., Cavalheri, T., Santos, H.G., Martins, I., Ozório, R.

#.csv files comprise the raw data collected during our experiments

#to execute:

python .\CombVarFishAlloModel.py -i <input data>.csv -o <output folder> -f <function>

#<input data> = input CSV file full path
#<output folder> = output folder where results will be stored.
#<function> = lin for linear
#			 poly2 for polinomial degree 2
#			 pow for power
#			 loglin for log linear
       
#Installed Python should be version 3.8.

#Python libraries with versions listed in CombVarFish_pylib_versions.txt should be installed.

