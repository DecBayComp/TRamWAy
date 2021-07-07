
from . import *
from ..attribute import initializer_method

from_ascii_file    = initializer_method( SPTDataInitializer.from_ascii_file    )
from_ascii_files   = initializer_method( SPTDataInitializer.from_ascii_files   )
from_dataframe     = initializer_method( SPTDataInitializer.from_dataframe     )
from_dataframes    = initializer_method( SPTDataInitializer.from_dataframes    )
from_mat_file      = initializer_method( SPTDataInitializer.from_mat_file      )
from_mat_files     = initializer_method( SPTDataInitializer.from_mat_files     )
from_rwa_file      = initializer_method( SPTDataInitializer.from_rwa_file      )
from_rwa_files     = initializer_method( SPTDataInitializer.from_rwa_files     )
from_rw_generator  = initializer_method( SPTDataInitializer.from_rw_generator  )
from_analysis_tree = initializer_method( SPTDataInitializer.from_analysis_tree )
#from_tracker      = initializer_method( SPTDataInitializer.from_tracker       )

