
from . import *
from . import utils
from ..attribute import initializer_method

from_bounding_boxes = initializer_method( ROIInitializer.from_bounding_boxes )
from_squares        = initializer_method( ROIInitializer.from_squares        )
from_ascii_file     = initializer_method( ROIInitializer.from_ascii_file     )
from_ascii_files    = initializer_method( ROIInitializer.from_ascii_files    )
from_dedicated_rwa_record  = initializer_method( ROIInitializer.from_dedicated_rwa_record  )
from_dedicated_rwa_records = initializer_method( ROIInitializer.from_dedicated_rwa_records )

