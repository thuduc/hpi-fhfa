"""
Unit tests for data models.

Tests the Pydantic models and their validation logic.
"""

import pytest
from datetime import date, datetime
from pydantic import ValidationError

from rsai.src.data.models import (
    PropertyType,
    TransactionType,
    GeographyLevel,
    WeightingScheme,
    PropertyCharacteristics,
    GeographicLocation,
    Transaction,
    RepeatSalePair,
    IndexValue,
    BMNRegressionResult,
    SupertractDefinition,
    QualityMetrics,
    RSAIConfig
)


class TestEnumerations:
    """Test enumeration values."""
    
    def test_property_type_values(self):
        assert PropertyType.SINGLE_FAMILY.value == "single_family"
        assert PropertyType.CONDO.value == "condo"
        assert PropertyType.TOWNHOUSE.value == "townhouse"
        assert PropertyType.MULTI_FAMILY.value == "multi_family"
        assert PropertyType.OTHER.value == "other"
    
    def test_transaction_type_values(self):
        assert TransactionType.ARMS_LENGTH.value == "arms_length"
        assert TransactionType.FORECLOSURE.value == "foreclosure"
        assert TransactionType.SHORT_SALE.value == "short_sale"
    
    def test_geography_level_values(self):
        assert GeographyLevel.PROPERTY.value == "property"
        assert GeographyLevel.TRACT.value == "tract"
        assert GeographyLevel.COUNTY.value == "county"
        assert GeographyLevel.NATIONAL.value == "national"
    
    def test_weighting_scheme_values(self):
        assert WeightingScheme.EQUAL.value == "equal"
        assert WeightingScheme.VALUE.value == "value"
        assert WeightingScheme.CASE_SHILLER.value == "case_shiller"
        assert WeightingScheme.BMN.value == "bmn"


class TestPropertyCharacteristics:
    """Test PropertyCharacteristics model."""
    
    def test_valid_characteristics(self):
        chars = PropertyCharacteristics(
            living_area=2000.0,
            lot_size=8000.0,
            bedrooms=3,
            bathrooms=2.5,
            age=20,
            stories=2,
            garage_spaces=2,
            pool=True,
            property_type=PropertyType.SINGLE_FAMILY
        )
        assert chars.living_area == 2000.0
        assert chars.bathrooms == 2.5
        assert chars.pool is True
    
    def test_optional_fields(self):
        chars = PropertyCharacteristics(property_type=PropertyType.CONDO)
        assert chars.living_area is None
        assert chars.bedrooms is None
        assert chars.pool is None
    
    def test_validation_boundaries(self):
        # Test negative values
        with pytest.raises(ValidationError):
            PropertyCharacteristics(living_area=-100)
        
        # Test excessive bedrooms
        with pytest.raises(ValidationError):
            PropertyCharacteristics(bedrooms=25)
        
        # Test excessive age
        with pytest.raises(ValidationError):
            PropertyCharacteristics(age=600)


class TestGeographicLocation:
    """Test GeographicLocation model."""
    
    def test_valid_location(self):
        loc = GeographicLocation(
            property_id="P123456",
            latitude=34.0522,
            longitude=-118.2437,
            address="123 Main St",
            zip_code="90001",
            tract="060371234.00",
            county_fips="06037",
            state="CA"
        )
        assert loc.property_id == "P123456"
        assert loc.latitude == 34.0522
    
    def test_zip_code_validation(self):
        # Valid 5-digit ZIP
        loc1 = GeographicLocation(property_id="P1", zip_code="90001")
        assert loc1.zip_code == "90001"
        
        # Valid 9-digit ZIP
        loc2 = GeographicLocation(property_id="P2", zip_code="900011234")
        assert loc2.zip_code == "900011234"
        
        # Valid ZIP+4
        loc3 = GeographicLocation(property_id="P3", zip_code="90001-1234")
        assert loc3.zip_code == "90001-1234"
        
        # Invalid ZIP
        with pytest.raises(ValidationError):
            GeographicLocation(property_id="P4", zip_code="123")
    
    def test_coordinate_validation(self):
        # Invalid latitude
        with pytest.raises(ValidationError):
            GeographicLocation(property_id="P1", latitude=91.0)
        
        # Invalid longitude
        with pytest.raises(ValidationError):
            GeographicLocation(property_id="P2", longitude=181.0)


class TestTransaction:
    """Test Transaction model."""
    
    def test_valid_transaction(self):
        trans = Transaction(
            transaction_id="T123456",
            property_id="P123456",
            sale_price=500000.0,
            sale_date=date(2023, 6, 15),
            transaction_type=TransactionType.ARMS_LENGTH,
            recording_date=date(2023, 6, 20),
            buyer_name="John Doe",
            seller_name="Jane Smith",
            mortgage_amount=400000.0
        )
        assert trans.sale_price == 500000.0
        assert trans.transaction_type == "arms_length"
    
    def test_price_validation(self):
        # Zero price
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="T1",
                property_id="P1",
                sale_price=0,
                sale_date=date(2023, 1, 1)
            )
        
        # Negative price
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="T2",
                property_id="P2",
                sale_price=-1000,
                sale_date=date(2023, 1, 1)
            )
    
    def test_date_validation(self):
        # Recording date before sale date
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="T1",
                property_id="P1",
                sale_price=100000,
                sale_date=date(2023, 6, 15),
                recording_date=date(2023, 6, 10)
            )
        
        # Valid: recording date after sale date
        trans = Transaction(
            transaction_id="T2",
            property_id="P2",
            sale_price=100000,
            sale_date=date(2023, 6, 15),
            recording_date=date(2023, 6, 20)
        )
        assert trans.recording_date > trans.sale_date


class TestRepeatSalePair:
    """Test RepeatSalePair model."""
    
    def test_valid_pair(self):
        pair = RepeatSalePair(
            pair_id="P123_T1_T2",
            property_id="P123",
            sale1_transaction_id="T1",
            sale1_price=400000.0,
            sale1_date=date(2020, 1, 15),
            sale2_transaction_id="T2",
            sale2_price=500000.0,
            sale2_date=date(2023, 6, 15),
            price_ratio=1.25,
            log_price_ratio=0.223,
            holding_period_days=1246,
            annualized_return=0.069,
            is_valid=True
        )
        assert pair.price_ratio == 1.25
        assert pair.holding_period_days == 1246
    
    def test_date_validation(self):
        # Sale2 before sale1
        with pytest.raises(ValidationError):
            RepeatSalePair(
                pair_id="P1_T1_T2",
                property_id="P1",
                sale1_transaction_id="T1",
                sale1_price=400000.0,
                sale1_date=date(2023, 6, 15),
                sale2_transaction_id="T2",
                sale2_price=500000.0,
                sale2_date=date(2020, 1, 15),
                price_ratio=1.25,
                log_price_ratio=0.223,
                holding_period_days=100,
                annualized_return=0.069
            )
    
    def test_price_ratio_validation(self):
        # Incorrect price ratio
        with pytest.raises(ValidationError):
            RepeatSalePair(
                pair_id="P1_T1_T2",
                property_id="P1",
                sale1_transaction_id="T1",
                sale1_price=400000.0,
                sale1_date=date(2020, 1, 15),
                sale2_transaction_id="T2",
                sale2_price=500000.0,
                sale2_date=date(2023, 6, 15),
                price_ratio=2.0,  # Should be 1.25
                log_price_ratio=0.693,
                holding_period_days=1246,
                annualized_return=0.069
            )


class TestIndexValue:
    """Test IndexValue model."""
    
    def test_valid_index_value(self):
        idx = IndexValue(
            geography_level=GeographyLevel.COUNTY,
            geography_id="06037",
            period=date(2023, 6, 1),
            index_value=125.5,
            num_pairs=150,
            num_properties=120,
            median_price=650000.0,
            standard_error=2.5,
            confidence_lower=120.6,
            confidence_upper=130.4
        )
        assert idx.index_value == 125.5
        assert idx.geography_level == GeographyLevel.COUNTY
    
    def test_validation(self):
        # Zero index value
        with pytest.raises(ValidationError):
            IndexValue(
                geography_level=GeographyLevel.TRACT,
                geography_id="123456",
                period=date(2023, 1, 1),
                index_value=0,
                num_pairs=10,
                num_properties=8
            )
        
        # Negative pairs
        with pytest.raises(ValidationError):
            IndexValue(
                geography_level=GeographyLevel.TRACT,
                geography_id="123456",
                period=date(2023, 1, 1),
                index_value=100,
                num_pairs=-1,
                num_properties=8
            )


class TestSuperTractDefinition:
    """Test SupertractDefinition model."""
    
    def test_valid_supertract(self):
        supertract = SupertractDefinition(
            supertract_id="06037_ST_001",
            name="Downtown LA Supertract",
            county_fips="06037",
            tract_ids=["060371001.00", "060371002.00", "060371003.00"],
            num_properties=500,
            num_transactions=1200,
            median_price=750000.0,
            min_lat=34.0,
            max_lat=34.1,
            min_lon=-118.3,
            max_lon=-118.2
        )
        assert len(supertract.tract_ids) == 3
        assert supertract.county_fips == "06037"
    
    def test_unique_tract_validation(self):
        # Duplicate tract IDs
        with pytest.raises(ValidationError):
            SupertractDefinition(
                supertract_id="06037_ST_001",
                county_fips="06037",
                tract_ids=["060371001.00", "060371002.00", "060371001.00"],
                num_properties=500,
                num_transactions=1200
            )
    
    def test_empty_tract_list(self):
        # Empty tract list
        with pytest.raises(ValidationError):
            SupertractDefinition(
                supertract_id="06037_ST_001",
                county_fips="06037",
                tract_ids=[],
                num_properties=0,
                num_transactions=0
            )


class TestQualityMetrics:
    """Test QualityMetrics model."""
    
    def test_valid_metrics(self):
        metrics = QualityMetrics(
            total_records=1000,
            valid_records=950,
            invalid_records=50,
            missing_counts={"price": 10, "date": 5},
            invalid_counts={"price": 20},
            completeness_score=0.95,
            validity_score=0.95,
            consistency_score=0.98,
            overall_score=0.96,
            issues=["Missing prices", "Invalid dates"]
        )
        assert metrics.total_records == 1000
        assert metrics.completeness_score == 0.95
    
    def test_count_validation(self):
        # Valid + invalid != total
        with pytest.raises(ValidationError):
            QualityMetrics(
                total_records=1000,
                valid_records=900,
                invalid_records=50,  # Should be 100
                completeness_score=0.95,
                validity_score=0.95,
                consistency_score=0.98,
                overall_score=0.96
            )
    
    def test_score_bounds(self):
        # Score > 1
        with pytest.raises(ValidationError):
            QualityMetrics(
                total_records=100,
                valid_records=100,
                invalid_records=0,
                completeness_score=1.5,
                validity_score=0.95,
                consistency_score=0.98,
                overall_score=0.96
            )


class TestRSAIConfig:
    """Test RSAIConfig model."""
    
    def test_valid_config(self):
        config = RSAIConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            frequency="monthly",
            geography_levels=[GeographyLevel.TRACT, GeographyLevel.COUNTY],
            weighting_scheme=WeightingScheme.BMN,
            min_pairs_threshold=10,
            max_holding_period_years=20,
            min_price=10000,
            max_price=10000000,
            outlier_std_threshold=3.0,
            output_format="parquet",
            include_diagnostics=True
        )
        assert config.frequency == "monthly"
        assert len(config.geography_levels) == 2
    
    def test_default_values(self):
        config = RSAIConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31)
        )
        assert config.frequency == "monthly"
        assert config.weighting_scheme == WeightingScheme.BMN
        assert config.min_pairs_threshold == 10
    
    def test_validation(self):
        # Zero min pairs
        with pytest.raises(ValidationError):
            RSAIConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                min_pairs_threshold=0
            )
        
        # Negative outlier threshold
        with pytest.raises(ValidationError):
            RSAIConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                outlier_std_threshold=-1.0
            )