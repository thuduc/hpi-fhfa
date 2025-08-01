"""Unit tests for census_tract module."""

import pytest
from hpi_fhfa.geography.census_tract import CensusTract


class TestCensusTract:
    """Test CensusTract class functionality."""
    
    def test_valid_tract_creation(self):
        """Test creating a valid census tract."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        assert tract.tract_code == "12345678901"
        assert tract.cbsa_code == "12345"
        assert tract.state_code == "12"
        assert tract.county_code == "345"
        assert tract.tract_number == "678901"
        assert tract.centroid_lat == 40.7128
        assert tract.centroid_lon == -74.0060
        assert tract.distance_to_cbd == 5.5
    
    def test_auto_extract_components(self):
        """Test automatic extraction of state/county/tract components."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="",
            county_code="",
            tract_number="",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        assert tract.state_code == "12"
        assert tract.county_code == "345"
        assert tract.tract_number == "678901"
    
    def test_invalid_tract_code_length(self):
        """Test validation of tract code length."""
        with pytest.raises(ValueError, match="Tract code must be 11 digits"):
            CensusTract(
                tract_code="12345",  # Too short
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5
            )
    
    def test_invalid_tract_code_format(self):
        """Test validation of tract code format."""
        with pytest.raises(ValueError, match="must contain only digits"):
            CensusTract(
                tract_code="1234567890A",  # Contains letter
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5
            )
    
    def test_invalid_latitude(self):
        """Test validation of latitude bounds."""
        with pytest.raises(ValueError, match="Invalid latitude"):
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=95.0,  # Invalid
                centroid_lon=-74.0060,
                distance_to_cbd=5.5
            )
    
    def test_invalid_longitude(self):
        """Test validation of longitude bounds."""
        with pytest.raises(ValueError, match="Invalid longitude"):
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-200.0,  # Invalid
                distance_to_cbd=5.5
            )
    
    def test_optional_fields(self):
        """Test optional demographic fields."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5,
            population=10000,
            housing_units=4000,
            college_share=0.35,
            nonwhite_share=0.25
        )
        
        assert tract.population == 10000
        assert tract.housing_units == 4000
        assert tract.college_share == 0.35
        assert tract.nonwhite_share == 0.25
    
    def test_invalid_college_share(self):
        """Test validation of college share bounds."""
        with pytest.raises(ValueError, match="College share must be between 0 and 1"):
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5,
                college_share=1.5  # Invalid
            )
    
    def test_invalid_nonwhite_share(self):
        """Test validation of nonwhite share bounds."""
        with pytest.raises(ValueError, match="Nonwhite share must be between 0 and 1"):
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5,
                nonwhite_share=-0.1  # Invalid
            )
    
    def test_centroid_property(self):
        """Test centroid property returns tuple."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        assert tract.centroid == (40.7128, -74.0060)
    
    def test_full_fips_property(self):
        """Test full FIPS code property."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        assert tract.full_fips == "12345"
    
    def test_is_adjacent_to(self):
        """Test adjacency check (same county)."""
        tract1 = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        tract2 = CensusTract(
            tract_code="12345678902",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678902",
            centroid_lat=40.7228,
            centroid_lon=-74.0160,
            distance_to_cbd=5.0
        )
        
        tract3 = CensusTract(
            tract_code="12346678901",
            cbsa_code="12345",
            state_code="12",
            county_code="346",  # Different county
            tract_number="678901",
            centroid_lat=40.7328,
            centroid_lon=-74.0260,
            distance_to_cbd=6.0
        )
        
        assert tract1.is_adjacent_to(tract2)
        assert not tract1.is_adjacent_to(tract3)
    
    def test_get_demographic_weight(self):
        """Test getting demographic weights."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5,
            population=10000,
            housing_units=4000,
            college_share=0.35,
            nonwhite_share=0.25
        )
        
        assert tract.get_demographic_weight('college') == 0.35
        assert tract.get_demographic_weight('nonwhite') == 0.25
        assert tract.get_demographic_weight('population') == 10000
        assert tract.get_demographic_weight('housing_units') == 4000
    
    def test_get_demographic_weight_invalid(self):
        """Test getting invalid demographic weight type."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        with pytest.raises(ValueError, match="Unknown weight type"):
            tract.get_demographic_weight('invalid')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5,
            metadata={'source': 'census'}
        )
        
        result = tract.to_dict()
        assert result['tract_code'] == "12345678901"
        assert result['cbsa_code'] == "12345"
        assert result['centroid_lat'] == 40.7128
        assert result['source'] == 'census'
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'tract_code': "12345678901",
            'cbsa_code': "12345",
            'state_code': "12",
            'county_code': "345",
            'tract_number': "678901",
            'centroid_lat': 40.7128,
            'centroid_lon': -74.0060,
            'distance_to_cbd': 5.5,
            'extra_field': 'extra_value'
        }
        
        tract = CensusTract.from_dict(data)
        assert tract.tract_code == "12345678901"
        assert tract.cbsa_code == "12345"
        assert tract.metadata['extra_field'] == 'extra_value'
    
    def test_string_representations(self):
        """Test string and repr methods."""
        tract = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        assert str(tract) == "CensusTract(12345678901, CBSA: 12345)"
        assert "tract_code='12345678901'" in repr(tract)
        assert "cbsa_code='12345'" in repr(tract)
    
    def test_equality(self):
        """Test equality comparison."""
        tract1 = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        tract2 = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        tract3 = CensusTract(
            tract_code="12345678902",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678902",
            centroid_lat=40.7228,
            centroid_lon=-74.0160,
            distance_to_cbd=5.0
        )
        
        assert tract1 == tract2
        assert tract1 != tract3
        assert tract1 != "not a tract"
    
    def test_hash(self):
        """Test hash functionality."""
        tract1 = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        tract2 = CensusTract(
            tract_code="12345678901",
            cbsa_code="12345",
            state_code="12",
            county_code="345",
            tract_number="678901",
            centroid_lat=40.7128,
            centroid_lon=-74.0060,
            distance_to_cbd=5.5
        )
        
        # Same tract code should have same hash
        assert hash(tract1) == hash(tract2)
        
        # Can be used in sets
        tract_set = {tract1, tract2}
        assert len(tract_set) == 1