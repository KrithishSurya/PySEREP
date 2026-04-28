"""pyserep.models — built-in synthetic FE model generators for testing."""
from pyserep.models.synthetic import (
    euler_beam,
    model_info,
    plate_2d,
    random_symmetric_pd,
    spring_chain,
)

__all__ = ["spring_chain","euler_beam","plate_2d","random_symmetric_pd","model_info"]
