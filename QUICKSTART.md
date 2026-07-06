# 🚀 Quick Start Guide - Benchmarking Dashboard

## Installation (5 minutes)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Data Files
Ensure your `trained_parameters` directory contains:
```
trained_parameters/
├── level1_neural_network.keras
├── linear_models.json
├── metadata.json
├── books_200M_uint32          (binary dataset)
├── test_keys.npy
├── test_indices.npy
└── benchmark_results.json
```

### Step 3: Start the Dashboard
```bash
python app.py
```

You'll see:
```
================================================================================
HYBRID LEARNED INDEX BENCHMARKING DASHBOARD
================================================================================

📊 Dashboard initialized with 200,000,000 keys
🔧 RMI: 1000 linear models with neural network router
🌳 B+ Tree: Binary search implementation

🚀 Starting server at http://127.0.0.1:8050
================================================================================
```

### Step 4: Open in Browser
Navigate to: **http://127.0.0.1:8050**

---

## Dashboard Walkthrough

### 📊 Header Metrics (Top of Page)
Shows at a glance:
- **Dataset Size**: 200,000,000 Keys
- **Key Range**: 0 to 4,294,967,294
- **RMI Configuration**: 1,000 Linear Models
- **Mean Error**: ~283 positions

### 🔍 Interactive Search Section
```
1. Enter a key value (or click "Random Key")
2. Click "Search Both"
3. View instant latency for RMI vs B+ Tree
```

### ⚡ Benchmark Operations
```
1. Set query count (10-10,000)
2. Click "Run Benchmark"
3. Wait for completion (~10-30 seconds)
4. View results in all tabs automatically
```

---

## Tab Guide

### 📈 Performance Metrics (Default)
**Best for**: Comparing lookup speed
- **Area Chart**: Lookup times for first 100 queries
- **Cumulative Time**: Total execution time growth
- **Latency Histogram**: Distribution of lookup times
- **Percentile Chart**: P50, P90, P95, P99, P99.9 comparison

**Key Insight**: See if RMI's learning provides speed advantage

### 📊 Dataset Explorer
**Best for**: Understanding data distribution
- **Data Table**: Sample of 20 records
- **Scatter Plot**: Keys across dataset range
- **Histogram**: Key value distribution

**Key Insight**: Verify dataset is evenly distributed

### 🎯 Prediction Quality
**Best for**: RMI accuracy analysis
- **Error Distribution**: Histogram of model errors
- **Box Plot**: Mean vs Max errors
- **Radar Chart**: Quality metrics

**Key Insight**: Understand prediction accuracy trade-offs

### 🏗️ Structural Analysis
**Best for**: Index structure comparison
- **Model Distribution**: Keys per linear model
- **Error Heatmap**: Error across all 1000 models
- **Scaling Curve**: Performance with dataset size

**Key Insight**: See how well models partition the data

### ⚔️ Head-to-Head Comparison
**Best for**: Final performance verdict
- **Radar Chart**: Multi-dimensional comparison
- **Box Plot**: Latency distributions side-by-side
- **Metrics Table**: Numerical comparison

**Key Insight**: Which index wins on your dataset?

---

## Common Tasks

### Task 1: Quick Performance Check (5 min)
```
1. Open dashboard (already running from Step 3)
2. Click "Performance Metrics" tab (default)
3. Enter query count: 100
4. Click "Run Benchmark"
5. Review the 4 charts
```

### Task 2: Search Specific Key (1 min)
```
1. Scroll to "Interactive Search" section
2. Enter key value (e.g., 2500000000)
3. Click "Search Both"
4. Compare RMI vs B+ Tree latency
```

### Task 3: Random Key Testing (2 min)
```
1. In "Interactive Search", click "Random Key"
2. Repeat 5-10 times
3. Note latency patterns
```

### Task 4: Full Benchmark Analysis (30 min)
```
1. Set query count: 5000
2. Run benchmark
3. Check all 5 tabs for insights
4. Screenshot interesting results
```

### Task 5: Dataset Analysis (10 min)
```
1. Go to "Dataset Explorer" tab
2. View sample data and distribution plots
3. Check if data is balanced
```

---

## Understanding the Results

### ✅ What Good Results Look Like

**RMI Advantages:**
- Lower average latency (typically 2-5x faster)
- Lower P95/P99 percentiles
- Higher throughput (QPS)

**B+ Tree Advantages:**
- More consistent latency (lower std dev)
- Guaranteed structure
- Better for dynamic inserts/deletes

### 📊 Metric Meanings

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| Avg Latency | Average lookup time | < 0.01 ms |
| P99 Latency | 99th percentile time | < 0.1 ms |
| Throughput | Queries per second | > 100,000 |
| Accuracy | % of keys found | 100% |
| Std Dev | Consistency measure | Low < High |

---

## Customization

### Change Sample Size for Visualizations
Edit `app.py`, line ~430:
```python
SAMPLE_SIZE = 100  # Change to desired number
```

### Adjust UI Colors
Edit `COLORS` dictionary in `app.py`:
```python
COLORS = {
    "accent_primary": "#3b82f6",  # Change RMI color
    "accent_secondary": "#10b981",  # Change B+ Tree color
}
```

### Modify RMI Search Parameters
Edit `RMIIndex.predict()` method in `app.py`:
```python
neighbor_models = 25  # Increase for more candidates
error_bound = 64      # Increase search window
```

---

## Troubleshooting

### Problem: "FileNotFoundError: books_200M_uint32"
**Solution:**
```bash
# Verify file exists
ls -lh trained_parameters/books_200M_uint32

# Check file size (should be ~800MB)
```

### Problem: Dashboard loads but charts are empty
**Solution:**
```bash
# Run a benchmark first
# Click "Run Benchmark" button
# Wait for "✅ Benchmark Completed" message
```

### Problem: "MemoryError" during benchmark
**Solution:**
```bash
# Reduce benchmark size
# Change from 10000 to 1000 queries
```

### Problem: Slow performance
**Solution:**
```bash
# Reduce visualization sample size
# Close other applications
# Use smaller benchmark sizes
```

---

## Keyboard Shortcuts

| Action | Method |
|--------|--------|
| Random Key | Click button or press R |
| Search | Click button or press Enter in input |
| Run Benchmark | Click button or press Space |
| Tab Navigation | Click tabs or use arrow keys |

---

## Tips & Tricks

### 💡 Tip 1: Batch Testing
Run multiple benchmarks with different sizes:
- 100, 500, 1000, 5000, 10000 queries
- Compare how performance scales

### 💡 Tip 2: Edge Case Testing
Test specific keys:
- Minimum key: 0
- Maximum key: 4,294,967,294
- Middle key: 2,147,483,647

### 💡 Tip 3: Screenshot for Papers
- Hover over charts
- Click camera icon
- Save as PNG for publications

### 💡 Tip 4: Repeat Benchmarks
Run same size benchmark twice:
- Check consistency
- Identify variance
- Better for statistical analysis

### 💡 Tip 5: Export Data
Tables are interactive:
- Copy metrics for spreadsheets
- Paste into papers/presentations
- Use for statistical analysis

---

## Next Steps

### For Academic Research:
1. ✅ Run benchmark with 10,000 queries
2. ✅ Screenshot all tabs
3. ✅ Record metrics from Comparison tab
4. ✅ Compare with related work
5. ✅ Include results in paper

### For Performance Analysis:
1. ✅ Test with different query distributions
2. ✅ Modify RMI neighbor parameters
3. ✅ Change B+ Tree order
4. ✅ Compare results
5. ✅ Document findings

### For Production Deployment:
1. ✅ Run comprehensive benchmark (10K+ queries)
2. ✅ Verify accuracy metrics
3. ✅ Test on production dataset size
4. ✅ Document performance characteristics
5. ✅ Create deployment playbook

---

## Need Help?

1. **Check README.md** for detailed documentation
2. **Review config.py** for customization options
3. **Examine app.py comments** for code explanations
4. **Test with sample queries** from dataset

---

## Performance Baseline

For reference, typical results on 200M key dataset:

```
RMI Lookup:
  Avg Latency: ~0.003 ms
  P99 Latency: ~0.05 ms
  Throughput: ~300,000+ QPS

B+ Tree Lookup:
  Avg Latency: ~0.01 ms
  P99 Latency: ~0.1 ms
  Throughput: ~100,000+ QPS
```

*Actual results depend on hardware and query distribution*

---

**Happy Benchmarking! 🚀📊**
