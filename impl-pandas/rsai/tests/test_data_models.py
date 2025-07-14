"""Unit tests for data models"""

import pytest
from datetime import date
from pydantic import ValidationError

from rsai.src.data.models import (
    Transaction, RepeatSalePair, GeographicData, 
    WeightingData, SupertractDefinition, IndexValue
)


class TestTransaction:
    """Test Transaction model"""
    
    def test_valid_transaction(self):
        """Test creating a valid transaction"""
        trans = Transaction(
            property_id="PROP001",
            transaction_date=date(2020, 1, 15),
            transaction_price=250000.0,
            census_tract_2010="06037123456",
            cbsa_id="31080"
        )
        assert trans.property_id == "PROP001"
        assert trans.transaction_price == 250000.0
    
    def test_negative_price_validation(self):
        """Test that negative prices are rejected"""
        with pytest.raises(ValidationError):
            Transaction(
                property_id="PROP001",
                transaction_date=date(2020, 1, 15),
                transaction_price=-100.0,
                census_tract_2010="06037123456",
                cbsa_id="31080"
            )
    
    def test_zero_price_validation(self):
        """Test that zero prices are rejected"""
        with pytest.raises(ValidationError):
            Transaction(
                property_id="PROP001",
                transaction_date=date(2020, 1, 15),
                transaction_price=0.0,
                census_tract_2010="06037123456",
                cbsa_id="31080"
            )


class TestRepeatSalePair:
    """Test RepeatSalePair model"""
    
    def test_valid_repeat_sale(self):
        """Test creating a valid repeat sale pair"""
        pair = RepeatSalePair(
            property_id="PROP001",
            first_sale_date=date(2018, 1, 15),
            first_sale_price=200000.0,
            second_sale_date=date(2020, 1, 15),
            second_sale_price=250000.0,
            census_tract_2010="06037123456",
            cbsa_id="31080",
            log_price_relative=0.223,
            annual_growth_rate=0.118,
            years_between_sales=2.0
        )
        assert pair.property_id == "PROP001"
        assert pair.years_between_sales == 2.0
    
    def test_sale_order_validation(self):
        """Test that second sale must be after first sale"""
        with pytest.raises(ValidationError):
            RepeatSalePair(
                property_id="PROP001",
                first_sale_date=date(2020, 1, 15),
                first_sale_price=200000.0,
                second_sale_date=date(2018, 1, 15),  # Before first sale
                second_sale_price=250000.0,
                census_tract_2010="06037123456",
                cbsa_id="31080",
                log_price_relative=0.223,
                annual_growth_rate=0.118,
                years_between_sales=2.0
            )
    
    def test_same_date_validation(self):
        """Test that same date sales are rejected"""
        with pytest.raises(ValidationError):
            RepeatSalePair(
                property_id="PROP001",
                first_sale_date=date(2020, 1, 15),
                first_sale_price=200000.0,
                second_sale_date=date(2020, 1, 15),  # Same date
                second_sale_price=250000.0,
                census_tract_2010="06037123456",
                cbsa_id="31080",
                log_price_relative=0.0,
                annual_growth_rate=0.0,
                years_between_sales=0.0
            )


class TestGeographicData:
    """Test GeographicData model"""
    
    def test_valid_geographic_data(self):
        """Test creating valid geographic data"""
        geo = GeographicData(
            census_tract_2010="06037123456",
            centroid_lat=34.0522,
            centroid_lon=-118.2437,
            cbsa_id="31080"
        )
        assert geo.census_tract_2010 == "06037123456"
        assert geo.centroid_lat == 34.0522
    
    def test_invalid_latitude(self):
        """Test latitude bounds validation"""
        with pytest.raises(ValidationError):
            GeographicData(
                census_tract_2010="06037123456",
                centroid_lat=91.0,  # Invalid latitude
                centroid_lon=-118.2437,
                cbsa_id="31080"
            )
    
    def test_invalid_longitude(self):
        """Test longitude bounds validation"""
        with pytest.raises(ValidationError):
            GeographicData(
                census_tract_2010="06037123456",
                centroid_lat=34.0522,
                centroid_lon=181.0,  # Invalid longitude
                cbsa_id="31080"
            )


class TestWeightingData:
    """Test WeightingData model"""
    
    def test_valid_weighting_data(self):
        """Test creating valid weighting data"""
        weight = WeightingData(
            census_tract_2010="06037123456",
            year=2020,
            total_housing_units=1500,
            total_housing_value=750000000.0,
            total_upb=600000000.0,
            college_population=3000,
            non_white_population=2500
        )
        assert weight.total_housing_units == 1500
    
    def test_optional_fields(self):
        """Test that optional fields can be None"""
        weight = WeightingData(
            census_tract_2010="06037123456",
            year=2020
        )
        assert weight.total_housing_units is None
        assert weight.total_housing_value is None
    
    def test_negative_values_rejected(self):
        """Test that negative values are rejected"""
        with pytest.raises(ValidationError):
            WeightingData(
                census_tract_2010="06037123456",
                year=2020,
                total_housing_units=-100  # Negative value
            )


class TestSupertractDefinition:
    """Test SupertractDefinition model"""
    
    def test_valid_supertract(self):
        """Test creating valid supertract definition"""
        supertract = SupertractDefinition(
            supertract_id="31080_2020_ST0001",
            year=2020,
            cbsa_id="31080",
            component_tracts=["06037123456", "06037123457"],
            half_pairs_count=45
        )
        assert len(supertract.component_tracts) == 2
        assert supertract.half_pairs_count == 45
    
    def test_empty_tracts_rejected(self):
        """Test that empty component tracts list is rejected"""
        with pytest.raises(ValidationError):
            SupertractDefinition(
                supertract_id="31080_2020_ST0001",
                year=2020,
                cbsa_id="31080",
                component_tracts=[],  # Empty list
                half_pairs_count=45
            )
    
    def test_duplicate_tracts_rejected(self):
        """Test that duplicate tracts are rejected"""
        with pytest.raises(ValidationError):
            SupertractDefinition(
                supertract_id="31080_2020_ST0001",
                year=2020,
                cbsa_id="31080",
                component_tracts=["06037123456", "06037123456"],  # Duplicate
                half_pairs_count=45
            )


class TestIndexValue:
    """Test IndexValue model"""
    
    def test_valid_index_value(self):
        """Test creating valid index value"""
        index = IndexValue(
            cbsa_id="31080",
            year=2020,
            index_value=105.5,
            appreciation_rate=0.055,
            weighting_scheme="sample",
            observations=1500
        )
        assert index.index_value == 105.5
        assert index.weighting_scheme == "sample"
    
    def test_negative_index_rejected(self):
        """Test that negative index values are rejected"""
        with pytest.raises(ValidationError):
            IndexValue(
                cbsa_id="31080",
                year=2020,
                index_value=-10.0,  # Negative index
                appreciation_rate=0.055,
                weighting_scheme="sample",
                observations=1500
            )
    
    def test_negative_observations_rejected(self):
        """Test that negative observations are rejected"""
        with pytest.raises(ValidationError):
            IndexValue(
                cbsa_id="31080",
                year=2020,
                index_value=105.5,
                appreciation_rate=0.055,
                weighting_scheme="sample",
                observations=-10  # Negative observations
            )