# 🚀 Hybrid Learned Index vs B+ Tree Benchmarking Dashboard

A professional, research-grade benchmarking platform for comparing Hybrid Learned Index (Recursive Model Index) against traditional B+ Tree implementations. Built with Dash, Plotly, and modern UI/UX principles for academic research and commercial analytics.

## 📋 Overview

This dashboard provides a comprehensive environment to:

- **Compare Performance** of RMI and B+ Tree across identical datasets
- **Interactive Search** with real-time performance metrics
- **Comprehensive Benchmarking** with configurable workload sizes
- **Detailed Analytics** covering 40+ metrics and visualizations
- **Dataset Exploration** with visual statistics
- **Prediction Quality Analysis** for the learned index
- **Structural Analysis** of both index types
- **Exportable Reports** for academic publications

## 🎯 Key Features

### 1. **Performance Metrics**
- Lookup latency (avg, median, P95, P99, max)
- Throughput (queries per second)
- Latency distribution analysis
- Percentile comparisons
- Cumulative execution time

### 2. **Interactive Search**
- Search any key in the dataset
- Real-time performance comparison
- Random key sampling
- Position verification

### 3. **Dataset Explorer**
- 100-sample dataset visualization
- Key distribution plots
- Range statistics
- Sample data table

### 4. **Prediction Quality Analysis**
- Error distribution histograms
- Model-level error statistics
- Box plots and violin plots
- Radar charts for quality metrics

### 5. **Structural Analysis**
- Key distribution across linear models
- Model error heatmap
- Dataset scaling curves
- Tree structure statistics

### 6. **Head-to-Head Comparison**
- Multi-dimensional radar charts
- Side-by-side metrics tables
- Distribution comparisons
- Performance speedup analysis

## 📊 Architecture

```
app.py
├── RMIIndex
│   ├── Neural Network Router (Level 1)
│   └── 1000 Linear Models (Level 2)
├── BPlusTree
│   └── Binary Search Implementation
├── BenchmarkEngine
│   ├── Unified lookup interface
│   ├── Performance metrics computation
│   └── Event logging
└── Dash Application
    ├── Interactive UI with Glassmorphism
    ├── Real-time visualizations
    └── 6 interactive tabs
```

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
cd Final
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ensure Data Files Exist
The `trained_parameters` directory should contain:
- `level1_neural_network.keras` - Trained router network
- `linear_models.json` - 1000 linear model coefficients
- `metadata.json` - Dataset and model metadata
- `books_200M_uint32` - Binary dataset file (200M keys)
- `test_keys.npy` - Test key subset
- `test_indices.npy` - Corresponding indices

## 🚀 Running the Dashboard

```bash
python app.py
```

Then navigate to: **http://127.0.0.1:8050**

## 📖 Dashboard Tabs

### 📈 **Performance Metrics**
- Area chart of lookup times
- Cumulative execution time
- Latency distribution histogram
- Percentile comparison chart

### 📊 **Dataset Explorer**
- Sample dataset table (first 20 records)
- Key distribution scatter plot
- Key value histogram

### 🎯 **Prediction Quality**
- Error distribution across models
- Box plot of error statistics
- Radar chart of prediction metrics

### 🏗️ **Structural Analysis**
- Keys per model distribution
- Model error heatmap
- Dataset scaling curves (log-log)

### ⚔️ **Head-to-Head Comparison**
- Multi-dimensional radar chart
- Box plot comparison
- Comprehensive metrics table

### ⚙️ **Interactive Controls**
- Real-time search with latency measurement
- Random key selection
- Configurable benchmark size (10-10,000 queries)
- Report generation button

## 🎨 Design Philosophy

### Glassmorphism UI
- Semi-transparent cards with backdrop blur
- Modern dark theme (slate blue background)
- Accent colors: Blue, Green, Orange
- Smooth animations and transitions
- Professional typography

### Color Scheme
```
Primary Background:   #0f172a (Dark Blue)
Secondary Background: #1e293b (Slate)
Card Background:      #334155 (Lighter Slate)
Primary Accent:       #3b82f6 (Blue) - RMI
Secondary Accent:     #10b981 (Green) - B+ Tree
Tertiary Accent:      #f59e0b (Orange) - Metrics
Text Primary:         #f1f5f9 (Light Gray)
Text Secondary:       #cbd5e1 (Gray)
```

## 📊 Benchmark Metrics Computed

### Latency Metrics (milliseconds)
- Average, Median
- P95, P99 percentiles
- Min, Max, Standard Deviation

### Throughput Metrics
- Queries per second (QPS)
- Total execution time
- Cumulative analysis

### Accuracy Metrics
- Found count percentage
- Position correctness
- Result consistency

### RMI-Specific Metrics
- Prediction error distribution
- Router accuracy
- Model-level statistics
- Search window analysis

### B+ Tree Metrics
- Tree height
- Binary search iterations
- Node access patterns

## 🔍 Interactive Search Feature

1. **Manual Search**
   - Enter any key value in the input field
   - Click "Search Both" to compare
   - View instant latency measurements

2. **Random Sampling**
   - Click "Random Key" to select from dataset
   - Perfect for testing edge cases

3. **Results Display**
   - RMI position and latency
   - B+ Tree position and latency
   - Visual comparison

## 📈 Benchmark Execution

1. **Configure Workload**
   - Set query count (10-10,000)
   - Larger benchmarks = longer runtime

2. **Run Benchmark**
   - Click "Run Benchmark"
   - Monitor progress indicator
   - View real-time metrics update

3. **Analyze Results**
   - All visualizations update automatically
   - Toggle between tabs for different views
   - Export metrics from tables

## 💾 Data Requirements

### Dataset Format
- **Type**: Unsigned 32-bit integers (uint32)
- **Size**: 200 million keys
- **Format**: Binary file (books_200M_uint32)
- **Properties**: Pre-sorted, distinct keys

### Model Artifacts
```json
{
  "N": 200000000,
  "key_min": 0,
  "key_max": 4294967294,
  "num_linear_models": 1000,
  "global_mean_error": 283.88,
  "global_max_error": 4315537,
  "global_p95_error": 731.24,
  "global_p99_error": 1584.05
}
```

## 🔧 Customization

### Modify Index Behavior
Edit `RMIIndex` class parameters:
```python
neighbor_models = 25  # Candidate model range
error_bound = 64      # Binary search correction window
```

### Adjust B+ Tree Order
Edit `BPlusTree.MAX_KEYS`:
```python
MAX_KEYS = 32  # Increase for different branching
```

### Change UI Colors
Edit `COLORS` dictionary in app.py:
```python
COLORS = {
    "bg_primary": "#0f172a",
    "accent_primary": "#3b82f6",
    # ... etc
}
```

### Modify Sample Size
Change `SAMPLE_SIZE` variable:
```python
SAMPLE_SIZE = 100  # Adjust for dataset preview
```

## 📊 Exporting Results

### Table Export
- Click on any table and use browser's built-in tools
- Copy metrics for spreadsheets
- Screenshot capability for presentations

### Chart Export
- Hover over any Plotly chart
- Use the camera icon to save PNG
- Download in high-quality format

## 🎓 Academic Use

This dashboard is designed for:
- **Thesis Research** - Comprehensive benchmarking evidence
- **Conference Presentations** - Professional visualizations
- **Performance Papers** - Detailed metric tables
- **Comparative Analysis** - Head-to-head evaluation

## 🔬 Research Metrics Included

- Lookup performance scaling
- Memory efficiency analysis
- Prediction accuracy evaluation
- Error distribution analysis
- Model distribution statistics
- Throughput under load
- Latency percentile analysis
- Consistency metrics

## ⚡ Performance Characteristics

### RMI (Hybrid Learned Index)
- **Strengths**: Superior average latency, logarithmic complexity
- **Tradeoffs**: Training overhead, model maintenance
- **Use Case**: Read-heavy workloads, static datasets

### B+ Tree
- **Strengths**: Guaranteed balance, insertion/deletion support
- **Tradeoffs**: Higher latency variance, node overhead
- **Use Case**: Dynamic data, general-purpose indexing

## 🐛 Troubleshooting

### Dashboard Won't Load
```bash
# Verify TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"

# Check Dash installation
python -c "import dash; print(dash.__version__)"
```

### Data File Not Found
```bash
# Ensure file exists with correct path
ls -lh trained_parameters/books_200M_uint32
```

### Out of Memory
- Reduce `SAMPLE_SIZE` for visualization
- Use smaller benchmark sizes (100-1000 queries)
- Close other applications

## 📈 Visualization Types

- **Area Charts** - Lookup time progression
- **Line Charts** - Cumulative performance
- **Histograms** - Latency and error distributions
- **Box Plots** - Statistical distributions
- **Bar Charts** - Percentile comparisons
- **Scatter Plots** - Dataset distribution
- **Heatmaps** - Multi-dimensional model analysis
- **Radar Charts** - Multi-metric comparison
- **Tables** - Detailed metric values

## 🔐 Requirements

- Python 3.8+
- 8GB RAM (for 200M dataset)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for Plotly CDN)

## 📝 Citation

If using this dashboard for research, cite as:
```bibtex
@software{rmi_dashboard_2024,
  title = {Hybrid Learned Index vs B+ Tree Benchmarking Dashboard},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourrepo/rmi-dashboard}
}
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all data files are present
3. Ensure Python dependencies are correctly installed
4. Check browser console for JavaScript errors

## 📄 License

Academic research project. Free to use for educational purposes.

---

**Built with**: Dash • Plotly • Bootstrap • TensorFlow • NumPy • Pandas

**Design Inspiration**: Grafana, Datadog, Vercel Analytics, GitHub Insights, Stripe Dashboard
