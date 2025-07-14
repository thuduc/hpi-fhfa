## **Product Requirements Document: Repeat-Sales Aggregation Index (RSAI) Model**

**Version:** 1.0

**Date:** July 11, 2025

**Author:** Gemini AI

**Source Document:** "A Flexible Method of House Price Index Construction using Repeat-Sales Aggregates" (FHFA Working Paper 21-01) 1

### **1\. Introduction**

This document specifies the requirements for a Python-based application that implements the **Repeat-Sales Aggregation Index (RSAI)** methodology. The goal is to produce robust, city-level house price indices by first estimating granular price changes in small, localized submarkets (Census tracts) and then aggregating them using various weighting schemes222. This approach is designed to be flexible and mitigate biases from non-random sampling that affect traditional house price indices3.

The final product will be a code library capable of processing raw housing transaction data to generate these tailored indices.

---

### **2\. Goals and Objectives**

* **Primary Goal:** To develop a programmatic implementation of the RSAI model that calculates annual house price indices for Core-Based Statistical Areas (CBSAs).  
* **Key Objectives:**  
  * To construct a balanced panel of house price indices for thousands of Census tracts over a long time horizon (e.g., 1989-2021)4.

  * To implement a novel "supertract" algorithm that ensures feasible index estimation even in areas with low transaction counts5555.

  * To provide the flexibility to aggregate tract-level indices into city-level indices using multiple, user-defined weighting schemes6666.

  * To create a tool that researchers, policymakers, and financial analysts can use to generate bespoke house price indices tailored to specific use-cases, such as tracking collateral value or minority housing wealth7777.

---

### **3\. Scope**

#### **In-Scope**

* **Data Processing:** Ingestion and filtering of raw repeat-sales housing transaction data.  
* **Algorithm Implementation:** The full, multi-step algorithm as described in the paper, including:  
  1. Identification and filtering of repeat-sales pairs.  
  2. Dynamic creation of "supertracts" on an annual basis.  
  3. Estimation of annual appreciation rates for each supertract using the Bailey, Muth, and Nourse (BMN) regression method.  
  4. Aggregation of supertract appreciation rates to the CBSA level.  
  5. Chaining of annual rates to form continuous index series.  
* **Weighting Schemes:** Implementation of the six specific weighting schemes mentioned in the paper: Sample, Value (Laspeyres), Unit, UPB, College, and Non-White8. The system should be extensible to allow for new, custom weights.

* **Output:** Generation of final CBSA-level house price index data as a time series.

#### **Out-of-Scope**

* **Data Acquisition:** The system will assume that all necessary input data (transaction data, geographic data, weighting data) is provided by the user.  
* **Graphical User Interface (GUI):** This is a code-based library/application, not a visual tool.  
* **Real-time Processing:** The model is designed for batch processing of historical data on an annual basis.

---

### **4\. Functional Requirements**

The system shall be broken down into the following functional components:

#### **Feature 1: Data Ingestion and Preparation**

* **FR-1.1: Identify Repeat Sales:** The system must process a list of transactions and identify all pairs of sales for the same unique property ID. The data spans from 1975 to 2021, though transactions before 1989 primarily serve as controls9.

* **FR-1.2: Calculate Price Relatives:** For each repeat-sale pair, the system shall calculate the log-difference of the transaction prices10.

* **FR-1.3: Filter Transaction Pairs:** The system must apply the following three filters to the set of all repeat-sale pairs to remove outliers and transactions suggestive of significant quality changes11:

  * Remove pairs where the two transactions occurred within the same 12-month period12.

  * Remove pairs with a calculated compound annual growth rate exceeding 30%13.

  * Remove pairs where the cumulative appreciation is greater than 10 times the prior sale price or less than 25% of the prior sale price (a 75% drop)14.

#### **Feature 2: Supertract Generation Algorithm**

* **FR-2.1: Define Half-Pairs:** The system must be able to calculate "half-pairs" for any given geographic area (tract or supertract) and time period (year). A half-pair for a given year is the count of all repeat-sale transactions where either the first or second sale occurred in that year15.

* **FR-2.2: Implement Annual Supertract Creation:** For each year t in the analysis period (1989-2021), the system must perform the following iterative aggregation for each CBSA16161616:

  * **a. Set Threshold:** Use a minimum threshold of 40 half-pairs for a geography to be considered independently estimable. This check must be performed for both the current year  
    t and the prior year t-117171717.

  * b. Initial Check:  
    For a given CBSA, identify all Census tracts that do *not* meet the 40 half-pair threshold in either year t or t-118181818.

  * **c. Iterative Merging:** For each tract that fails the threshold, merge it with its single nearest neighbor (based on centroid distance). This creates a "supertract"19.

  * **d. Re-evaluate:** Recalculate the half-pair count for the newly formed supertract.  
  * e. Repeat:  
    If the new supertract still fails the threshold, continue merging it with its nearest neighbor (which may be a tract or another supertract) until the minimum observation count is met20.

  * f. Finalize Period-Specific Geographies:  
    This process results in a set of mutually exclusive and exhaustive "supertracts" for each CBSA that is unique to that specific year t21.

#### **Feature 3: Submarket (Supertract) Index Calculation**

* **FR-3.1: Run BMN Regressions:** For each CBSA and for each year t, the system must run a separate Bailey, Muth, and Nourse (BMN) repeat-sales regression for every supertract defined in **FR-2.2**22.

  * The regression model is:  
    pitτ​=Dtτ′​δtτ​+ϵitτ​23.

  * This must be the unweighted BMN (1963) formulation, not a version correcting for heteroskedasticity24.

  * The regression for a given supertract must use the  
    *entire time sample* of available repeat-sales data for the properties within that supertract25.

* **FR-3.2: Extract Appreciation Rates:** From each regression, the system shall extract the coefficients δ^n,t​ and δ^n,t−1​ for the current and prior year. The annual appreciation rate for that supertract  
  n is calculated as the difference: (δ^n,t​−δ^n,t−1​)262626.

#### **Feature 4: City-Level Index Aggregation (RSAI)**

* **FR-4.1: Calculate Weights:** The system must be able to calculate various weighting schemes, wn​, for each supertract n within a CBSA for a given year t. The weights must sum to 1\. Required schemes are:  
  * **Sample:** Share of half-pairs in the supertract relative to the CBSA total272727.

  * **Value (Laspeyres):** Share of aggregate single-family housing value using initial-period (t-1) values282828.

  * **Unit:** Share of the number of single-family housing units29.

  * **UPB:** Share of aggregate outstanding Unpaid Principal Balance of Enterprise loans30.

  * **College/Non-White (Lowe):** Share based on static 2010 Census demographic data31.

* **FR-4.2: Aggregate Appreciation:** The system shall calculate the city-level appreciation rate for year t by taking the weighted sum of all supertract appreciation rates within that city: p^​ta​=∑n=1Nt​​wn​(δ^n,t​−δ^n,t−1​)32. This must be done for each weighting scheme.

#### **Feature 5: Output Generation**

* **FR-5.1: Chain Index:** The system must chain the annual city-level appreciation rates together to form a continuous index series over time. The formula is  
  Pta​=P^t−1a​×exp(p^​ta​), with an initial value of P0a​=100 (or 1\)33.

* **FR-5.2: Generate Output Files:** The system shall output the final time series data (e.g., in CSV format), with columns for date, CBSA ID, index value, and weighting scheme.

---

### **5\. Data & Technical Requirements**

#### **Data Inputs**

1. **Transaction Data:** A table of all property transactions.  
   * property\_id: Unique identifier for each housing unit.  
   * transaction\_date: Date of sale.  
   * transaction\_price: Sale price in USD.  
   * census\_tract\_2010: The 2010 Census Tract ID for the property.  
   * cbsa\_id: The Core-Based Statistical Area ID.  
2. **Geographic Data:** Census Tract centroids for calculating nearest neighbors.  
   * census\_tract\_2010: Unique ID.  
   * centroid\_lat: Latitude of the tract's centroid.  
   * centroid\_lon: Longitude of the tract's centroid.  
3. **Weighting Data:** Tract-level data for constructing aggregation weights. Required for each year or as a static file (for Lowe indices).  
   * census\_tract\_2010: Unique ID.  
   * year: The year the data applies to.  
   * total\_housing\_units: Count of single-family units.  
   * total\_housing\_value: Aggregate value of single-family housing stock.  
   * total\_upb: Aggregate unpaid principal balance.  
   * college\_population: Count of college-educated population (static).  
   * non\_white\_population: Count of non-white population (static).

#### **Technical Stack**

* **Language:** Python (3.8+)  
* **Core Libraries:**  
  * **Pandas:** For all data manipulation and management.  
  * **NumPy:** For numerical calculations.  
  * **Statsmodels / Scikit-learn:** For implementing the BMN (Ordinary Least Squares) regression.  
  * **SciPy / Scikit-learn:** For efficiently calculating centroid distances to find nearest neighbors.