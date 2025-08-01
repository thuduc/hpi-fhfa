# House Price Index (HPI) - FHFA Method Product Requirements Document

## 1. Overview
This PRD describes the implementation requirements for the Federal Housing Finance Agency (FHFA) Repeat-Sales Aggregation Index (RSAI) method for constructing house price indices, based on Contat & Larson (2022).

## 2. Business Context
The RSAI method addresses limitations of traditional city-level house price indices by:
- Creating balanced panel of Census tract-level indices
- Handling low transaction counts through dynamic aggregation
- Supporting flexible weighting schemes for different use cases
- Avoiding sampling bias and composition effects

## 3. Functional Requirements

### 3.1 Core Data Structures

#### 3.1.1 Transaction Data
- **Fields Required:**
  - `property_id`: Unique identifier for each property
  - `transaction_date`: Date of transaction
  - `transaction_price`: Sale price
  - `census_tract`: 2010 Census tract identifier
  - `cbsa_code`: Core-Based Statistical Area code
  - `distance_to_cbd`: Distance from central business district

#### 3.1.2 Geographic Hierarchy
- **Census Tract**: Base geographic unit
- **Supertract**: Dynamic aggregation of census tracts
- **CBSA**: City-level aggregation

### 3.2 Mathematical Components

#### 3.2.1 Price Relatives Calculation
For repeat-sales pair of property `i` between times `t` and `τ`:
```
p_itτ = log(price_t) - log(price_τ)
```

#### 3.2.2 Data Filters
Remove pairs where:
- Same 12-month transaction period
- Compound annual growth rate > |30%|: `|(V1/V0)^(1/(t1-t0)) - 1| > 0.30`
- Cumulative appreciation > 10x or < 0.25x previous sale

#### 3.2.3 Bailey-Muth-Nourse (BMN) Regression
**Equation 1**: Basic repeat-sales regression
```
p_itτ = D'_tτ * δ_tτ + ε_itτ
```
Where:
- `p_itτ`: Log price difference for property i
- `D_tτ`: Dummy variable matrix for time periods
- `δ_tτ`: Coefficient vector (appreciation rates)
- `ε_itτ`: Error term (IID, mean zero)

**Equation 2**: Pooled index calculation
```
p̂_pooled(t,t-1) = δ̂_t - δ̂_t-1
```

### 3.3 Aggregation Framework

#### 3.3.1 Weighted Aggregation
**Equation 4**: General target index
```
p = Σ(n=1 to N) w_n * p_n
```
Where:
- `w_n`: Weight for submarket n (Σw_n = 1)
- `p_n`: Appreciation rate for submarket n
- `N`: Total number of submarkets

#### 3.3.2 Index Differences
**Equation 5**: Difference between indices
```
p - p' = Σ(w_n - w'_n) * p_n
```

### 3.4 Supertract Algorithm

#### 3.4.1 Parameters
- **MIN_HALF_PAIRS**: Minimum 40 half-pairs per year
- **AGGREGATION_METHOD**: Nearest centroid distance

#### 3.4.2 Algorithm Steps
For each period `t`:
1. Calculate half-pairs for each tract in periods `t` and `t-1`
2. If tract has < MIN_HALF_PAIRS in either period:
   - Merge with nearest tract/supertract by centroid distance
   - Repeat until threshold met
3. Estimate BMN regression for each supertract
4. Extract δ̂_n,t and δ̂_n,t-1 coefficients
5. Calculate appreciation: `p̂_n = δ̂_n,t - δ̂_n,t-1`

### 3.5 Weighting Schemes

#### 3.5.1 Available Weight Types
1. **Sample Weights** (`w_sample`): Share of half-pairs
2. **Value Weights** (`w_value`): Share of aggregate housing value (Laspeyres)
3. **Unit Weights** (`w_unit`): Share of housing units
4. **UPB Weights** (`w_upb`): Share of unpaid principal balance
5. **College Weights** (`w_college`): Share of college-educated population
6. **Non-White Weights** (`w_nonwhite`): Share of non-white population

#### 3.5.2 Weight Calculation
For time-varying weights (1-4):
```
w_n,t = measure_n,t / Σ(measure_n,t)
```

For static weights (5-6):
```
w_n = measure_n,2010 / Σ(measure_n,2010)
```

### 3.6 City-Level Index Construction

#### 3.6.1 Annual Index Algorithm
Initialize: `P_a(t=0) = 1`

For each period `t = 1...T`:
1. Construct N_t supertracts for period t
2. Calculate BMN indices for each supertract
3. Capture δ̂_n,t and δ̂_n,t-1
4. Calculate/specify weights w_n
5. Aggregate: `p̂_a(t) = Σ(n=1 to Nt) w_n * (δ̂_n,t - δ̂_n,t-1)`
6. Update index: `P̂_a(t) = P̂_a(t-1) * exp(p̂_a(t))`

## 4. Data Processing Requirements

### 4.1 Half-Pairs Calculation
For property with transactions at times [t1, t2, t3]:
- Half-pairs at t1: 1
- Half-pairs at t2: 2
- Half-pairs at t3: 1

### 4.2 Geographic Processing
- Census tract boundaries: 2010 definitions
- Distance calculations: Tract centroid to CBD
- Adjacency determination for supertract formation

## 5. Performance Requirements

### 5.1 Scale
- Handle 63.3 million repeat-sales pairs
- Process 63,122 census tracts
- Support 581 CBSAs
- Time range: 1975-2021 (indices from 1989-2021)

### 5.2 Computational Complexity
- Total regressions: ~1,037,465 supertract-year combinations
- Total parameters: ~47 million

## 6. Output Specifications

### 6.1 Tract-Level Indices
- Annual frequency
- Base year normalization (typically 1989=100)
- Balanced panel (all tracts, all years)

### 6.2 City-Level Indices
- Six index variants per CBSA
- Annual frequency
- Cumulative and year-over-year changes

## 7. Validation Requirements

### 7.1 Quality Checks
- Appreciation rates within reasonable bounds
- Sufficient observations per supertract
- Weight normalization (sum to 1)
- Index continuity

### 7.2 Comparison Metrics
- Correlation with existing indices
- Cross-sectional variance analysis
- Gradient analysis by distance to CBD