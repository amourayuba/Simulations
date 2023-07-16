# Simulations
A Simulation class that can be used to read and derive a variety of halo properties. Some methods are based of
the Amiga Halo Finder type of file and might not work for other file formats. 

## Simulation objects
A Simulation object has to have: 
path: str, base path of the simulation, other data have to be in specific folder from the base path 
name: str, name of the simulation. data files will be saved and reference this name 
om0: float, the value of Omega_m of the simulation. 
s8: float, the value of s8 of the simulation. 

### files required 
A few methods will require specific files to exist to work : 

A list of snapshot redshifts (from most recent to oldest) that has to be in '[path]/redshifts/[name].txt'

A list of prefix names (from most recent to oldest) for each snapshot in 'path/[name]_prefixes.txt'. These 
are AHF type prefixes that are used in AHF files halos/particles/profiles. A typical AHF halo file will 
be [prefix1].AHF_halos. 

If one wants to read AHF halo files, they can use the read_halos() method, but the halos have to be in 
'path/halos'. 

If one wants to make mass accretion history files from AHF MAH output, they can use the make_mah() method but 
the MAH of individual halos output from AHF have to be in 'path/MAH' 

If one wants to use any method that uses the MAH (average_growth(), large_growth(), formation_time()...) they 
need to have a MAH file in 'path/data/' with a name mah_[name].npy and if the ids of each halo are needed, 
one also needs a ids_[name].npy 

