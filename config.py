"""
Configuration file for the Benchmarking Dashboard
Easily customize behavior without modifying app.py
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Path to trained parameters directory
TRAINED_PARAMS_DIR = "trained_parameters"

# Path to binary dataset file
DATA_FILE_PATH = "trained_parameters/books_200M_uint32"

# Sample size for dataset visualization (in Performance tab)
SAMPLE_SIZE = 100

# ============================================================================
# RMI CONFIGURATION
# ============================================================================

# Number of neighbor models to check during search
RMI_NEIGHBOR_MODELS = 25

# Error bound for binary search window
RMI_ERROR_BOUND = 64

# ============================================================================
# B+ TREE CONFIGURATION
# ============================================================================

# Maximum keys per node (order of tree)
BPLUS_MAX_KEYS = 32

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

# Default benchmark size (number of queries)
DEFAULT_BENCHMARK_SIZE = 1000

# Minimum benchmark size
MIN_BENCHMARK_SIZE = 10

# Maximum benchmark size
MAX_BENCHMARK_SIZE = 10000

# ============================================================================
# UI THEME CONFIGURATION
# ============================================================================

# Dark theme color palette
UI_COLORS = {
    "bg_primary": "#0f172a",        # Dark navy blue
    "bg_secondary": "#1e293b",      # Slate 800
    "bg_tertiary": "#334155",       # Slate 700
    "accent_primary": "#3b82f6",    # Blue 500 (RMI)
    "accent_secondary": "#10b981",  # Green 500 (B+ Tree)
    "accent_tertiary": "#f59e0b",   # Amber 500 (Metrics)
    "text_primary": "#f1f5f9",      # Slate 100
    "text_secondary": "#cbd5e1",    # Slate 300
    "border_color": "#475569",      # Slate 600
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Number of bins for histograms
HISTOGRAM_BINS = 30

# Number of percentiles to display
PERCENTILES = [50, 90, 95, 99, 99.9]

# Enable interactive hover information
ENABLE_HOVER_INFO = True

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# Server host
SERVER_HOST = "127.0.0.1"

# Server port
SERVER_PORT = 8050

# Debug mode (set to False in production)
DEBUG_MODE = False

# Suppress callback exceptions (for optional callbacks)
SUPPRESS_CALLBACK_EXCEPTIONS = True

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Cache results (improves responsiveness)
ENABLE_CACHING = True

# Number of threads for parallel operations
NUM_THREADS = 4

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

# Export format (csv, json, xlsx)
EXPORT_FORMAT = "json"

# Include raw data in exports
INCLUDE_RAW_DATA = True

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Enable detailed logging
ENABLE_LOGGING = True

# Log file path
LOG_FILE_PATH = "benchmark.log"

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

# Number of test samples for initial load
INITIAL_TEST_SAMPLES = 100

# Precision for numerical outputs
FLOAT_PRECISION = 4

# Maximum data points to display in charts
MAX_CHART_POINTS = 5000

# Enable real-time progress updates
REAL_TIME_UPDATES = True
