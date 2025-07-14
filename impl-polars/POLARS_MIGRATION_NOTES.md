# Polars Migration Notes

## Key API Differences Encountered

### 1. Date/Time Operations
**Pandas:**
```python
(df['date2'] - df['date1']).dt.days
```

**Polars:**
```python
(df['date2'] - df['date1']).dt.total_days()
# or for integer days:
((df['date2'] - df['date1']).dt.total_milliseconds() / (24 * 60 * 60 * 1000)).cast(pl.Int32)
```

### 2. DataFrame Creation
**Pandas:**
```python
pd.DataFrame({'col': []})  # Empty with column
```

**Polars:**
```python
pl.DataFrame({'col': []}, schema={'col': pl.Float64})  # Need schema for empty
```

### 3. JSON Export
**Pandas:**
```python
df.to_json(path, orient='records', row_oriented=True)
```

**Polars:**
```python
df.write_json(path, orient='records')  # No row_oriented parameter
```

### 4. Column Selection
**Pandas:**
```python
df[['col1', 'col2']]
```

**Polars:**
```python
df.select(['col1', 'col2'])
# or
df[['col1', 'col2']]  # Also works but select is preferred
```

### 5. Aggregation
**Pandas:**
```python
df.groupby('col').agg({'value': ['mean', 'sum']})
```

**Polars:**
```python
df.group_by('col').agg([
    pl.col('value').mean().alias('value_mean'),
    pl.col('value').sum().alias('value_sum')
])
```

### 6. Index Operations
**Pandas:**
```python
df.set_index('col')
df.reset_index()
```

**Polars:**
```python
# No index concept - use sort for ordering
df.sort('col')
# Add row number if needed:
df.with_row_index()
```

### 7. Missing Value Handling
**Pandas:**
```python
df.fillna(0)
df.dropna()
```

**Polars:**
```python
df.fill_null(0)
df.drop_nulls()
```

### 8. Type Casting
**Pandas:**
```python
df['col'].astype(float)
```

**Polars:**
```python
df.with_columns(pl.col('col').cast(pl.Float64))
```

### 9. Conditional Operations
**Pandas:**
```python
df.loc[df['col'] > 0, 'col'] = 1
```

**Polars:**
```python
df.with_columns(
    pl.when(pl.col('col') > 0).then(1).otherwise(pl.col('col')).alias('col')
)
```

### 10. String Operations
**Pandas:**
```python
df['col'].str.contains('pattern')
```

**Polars:**
```python
df['col'].str.contains('pattern')  # Same API!
```

## Performance Benefits Observed

1. **Lazy Evaluation**: Can build complex query plans that are optimized
2. **Parallel Processing**: Automatic use of multiple cores
3. **Memory Efficiency**: Columnar storage reduces memory usage
4. **Type Safety**: Stronger typing prevents runtime errors

## Migration Tips

1. **Start with Schema**: Define schemas explicitly for better performance
2. **Use Lazy Frames**: For complex operations, use `.lazy()` and `.collect()`
3. **Avoid Loops**: Polars excels at vectorized operations
4. **Expression API**: Learn the expression API for complex transformations
5. **Check Documentation**: Many methods have slightly different names/parameters

## Common Gotchas

1. **No Inplace Operations**: Polars returns new DataFrames
2. **Different Null Handling**: Polars distinguishes between null and NaN
3. **Column Order**: Column order matters more in Polars
4. **No MultiIndex**: Use multiple columns instead
5. **Different Datetime Parsing**: May need to specify format explicitly