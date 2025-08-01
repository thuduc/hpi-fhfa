"""Census tract data structures and operations."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CensusTract:
    """Represents a census tract with geographic and demographic properties.
    
    Attributes:
        tract_code: 11-digit census tract code (2 state + 3 county + 6 tract)
        cbsa_code: Core Based Statistical Area code
        state_code: 2-digit state FIPS code
        county_code: 3-digit county FIPS code
        tract_number: 6-digit tract number
        centroid_lat: Latitude of tract centroid
        centroid_lon: Longitude of tract centroid
        distance_to_cbd: Distance to central business district in miles
        population: Total population (optional)
        housing_units: Total housing units (optional)
        college_share: Share of college-educated population (optional)
        nonwhite_share: Share of non-white population (optional)
        metadata: Additional tract properties
    """
    
    tract_code: str
    cbsa_code: str
    state_code: str
    county_code: str
    tract_number: str
    centroid_lat: float
    centroid_lon: float
    distance_to_cbd: float
    population: Optional[int] = None
    housing_units: Optional[int] = None
    college_share: Optional[float] = None
    nonwhite_share: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate tract code and extract components."""
        if len(self.tract_code) != 11:
            raise ValueError(f"Tract code must be 11 digits, got: {self.tract_code}")
        
        if not self.tract_code.isdigit():
            raise ValueError(f"Tract code must contain only digits, got: {self.tract_code}")
        
        # Extract components if not provided
        if not self.state_code:
            self.state_code = self.tract_code[:2]
        if not self.county_code:
            self.county_code = self.tract_code[2:5]
        if not self.tract_number:
            self.tract_number = self.tract_code[5:]
        
        # Validate coordinates
        if not -90 <= self.centroid_lat <= 90:
            raise ValueError(f"Invalid latitude: {self.centroid_lat}")
        if not -180 <= self.centroid_lon <= 180:
            raise ValueError(f"Invalid longitude: {self.centroid_lon}")
        
        # Validate optional fields
        if self.college_share is not None and not 0 <= self.college_share <= 1:
            raise ValueError(f"College share must be between 0 and 1, got: {self.college_share}")
        if self.nonwhite_share is not None and not 0 <= self.nonwhite_share <= 1:
            raise ValueError(f"Nonwhite share must be between 0 and 1, got: {self.nonwhite_share}")
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Return centroid as (lat, lon) tuple."""
        return (self.centroid_lat, self.centroid_lon)
    
    @property
    def full_fips(self) -> str:
        """Return full FIPS code (state + county)."""
        return self.state_code + self.county_code
    
    def is_adjacent_to(self, other: 'CensusTract') -> bool:
        """Check if this tract is in the same county as another tract.
        
        Note: True adjacency would require boundary data. This method
        checks for same-county membership as a proxy.
        """
        return self.full_fips == other.full_fips
    
    def get_demographic_weight(self, weight_type: str) -> Optional[float]:
        """Get demographic weight value for specified type.
        
        Args:
            weight_type: Type of weight ('college', 'nonwhite', 'population', 'housing_units')
            
        Returns:
            Weight value or None if not available
        """
        weight_map = {
            'college': self.college_share,
            'nonwhite': self.nonwhite_share,
            'population': self.population,
            'housing_units': self.housing_units
        }
        
        if weight_type not in weight_map:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        return weight_map[weight_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tract to dictionary representation."""
        return {
            'tract_code': self.tract_code,
            'cbsa_code': self.cbsa_code,
            'state_code': self.state_code,
            'county_code': self.county_code,
            'tract_number': self.tract_number,
            'centroid_lat': self.centroid_lat,
            'centroid_lon': self.centroid_lon,
            'distance_to_cbd': self.distance_to_cbd,
            'population': self.population,
            'housing_units': self.housing_units,
            'college_share': self.college_share,
            'nonwhite_share': self.nonwhite_share,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CensusTract':
        """Create CensusTract from dictionary.
        
        Args:
            data: Dictionary containing tract data
            
        Returns:
            CensusTract instance
        """
        # Extract known fields
        known_fields = {
            'tract_code', 'cbsa_code', 'state_code', 'county_code',
            'tract_number', 'centroid_lat', 'centroid_lon', 'distance_to_cbd',
            'population', 'housing_units', 'college_share', 'nonwhite_share'
        }
        
        tract_data = {k: v for k, v in data.items() if k in known_fields}
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        if metadata:
            tract_data['metadata'] = metadata
        
        return cls(**tract_data)
    
    def __str__(self) -> str:
        """String representation of tract."""
        return f"CensusTract({self.tract_code}, CBSA: {self.cbsa_code})"
    
    def __repr__(self) -> str:
        """Detailed representation of tract."""
        return (f"CensusTract(tract_code='{self.tract_code}', "
                f"cbsa_code='{self.cbsa_code}', "
                f"centroid=({self.centroid_lat:.4f}, {self.centroid_lon:.4f}), "
                f"distance_to_cbd={self.distance_to_cbd:.2f})")
    
    def __eq__(self, other) -> bool:
        """Check equality based on tract code."""
        if not isinstance(other, CensusTract):
            return False
        return self.tract_code == other.tract_code
    
    def __hash__(self) -> int:
        """Hash based on tract code."""
        return hash(self.tract_code)