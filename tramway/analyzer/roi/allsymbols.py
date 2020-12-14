
from . import *
from . import utils
from ..attribute import InitializerMethod

from_bounding_boxes = InitializerMethod( ROIInitializer.from_bounding_boxes )
from_squares        = InitializerMethod( ROIInitializer.from_squares        )
from_ascii_file     = InitializerMethod( ROIInitializer.from_ascii_file     )
from_ascii_files    = InitializerMethod( ROIInitializer.from_ascii_files    )
from_dedicated_rwa_record  = InitializerMethod( ROIInitializer.from_dedicated_rwa_record  )
from_dedicated_rwa_records = InitializerMethod( ROIInitializer.from_dedicated_rwa_records )

