
from . import *
from ..attribute import InitializerMethod

from_ascii_file    = InitializerMethod( SPTDataInitializer.from_ascii_file    )
from_ascii_files   = InitializerMethod( SPTDataInitializer.from_ascii_files   )
from_dataframe     = InitializerMethod( SPTDataInitializer.from_dataframe     )
from_dataframes    = InitializerMethod( SPTDataInitializer.from_dataframes    )
from_mat_file      = InitializerMethod( SPTDataInitializer.from_mat_file      )
from_mat_files     = InitializerMethod( SPTDataInitializer.from_mat_files     )
from_rwa_file      = InitializerMethod( SPTDataInitializer.from_rwa_file      )
from_rwa_files     = InitializerMethod( SPTDataInitializer.from_rwa_files     )
from_rw_generator  = InitializerMethod( SPTDataInitializer.from_rw_generator  )
from_analysis_tree = InitializerMethod( SPTDataInitializer.from_analysis_tree )
#from_tracker      = InitializerMethod( SPTDataInitializer.from_tracker       )

