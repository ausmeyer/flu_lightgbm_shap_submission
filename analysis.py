# -*- coding: utf-8 -*-

# Standard library imports
import collections
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shap
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

# Bioinformatics libraries
from Bio import SeqIO, Phylo
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import flupan

# Machine learning libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from bayes_opt import BayesianOptimization

# Geolocation
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import GoogleV3

# Statistical utilities
from scipy.stats import entropy, pearsonr, spearmanr, chi2_contingency, rankdata

# Configure logging
logging.getLogger('flupan').setLevel(logging.CRITICAL)
logging.getLogger('root').setLevel(logging.WARNING)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('LightGBM').setLevel(logging.ERROR)

# --- Configuration & Constants ---

# Set your Google Geocoding API Key here if using GoogleV3
GOOGLE_API_KEY = "api_here"

# File Paths
BASE_DIR = '.'
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_figures')
METADATA_FILE = os.path.join(DATA_DIR, 'gisaid_metadata_raw.csv')
SEQUENCE_FILE = os.path.join(DATA_DIR, 'H3_prot.fasta')
PASSAGES_CACHE = os.path.join(DATA_DIR, 'passages.pkl')
LOCATIONS_CACHE = os.path.join(DATA_DIR, 'locations.json')
COUNTRIES_CACHE = os.path.join(DATA_DIR, 'countries_lookup.json')
LEISR_FILE = os.path.join(DATA_DIR, 'H3_prot_leisr.csv')
LEISR_UNPASSAGED_FILE = os.path.join(DATA_DIR, 'H3_prot_unpassaged_leisr.csv')
FUBAR_FILE = os.path.join(DATA_DIR, 'H3_nuc_unpassaged_fubar.csv')
BAYES_OPT_LOG_PASSAGE = os.path.join(DATA_DIR, 'bayes_opt_passage_logs.json')
BAYES_OPT_CACHE_PASSAGE = os.path.join(DATA_DIR, 'bayes_fit_passage_classification_lgbm.pkl')
TREE_FILE = os.path.join(DATA_DIR, "H3_nuc_unpassaged.fasta.treefile")

# Control Flags
PARSE_PASSAGES_FRESH = True  # Set to False to use cached passage data (much faster!)
GEOCODE_FRESH = False
RUN_BAYES_OPT_PASSAGE = False
LOAD_BAYES_OPT_PASSAGE = True

# Set up publication-quality plotting style
def set_publication_style():
    """Set up matplotlib parameters for publication-quality figures."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Figure aesthetics
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.figsize'] = (8, 6)  # Default figure size
    
    # Seaborn style
    sns.set_style("ticks")
    
    # Define a colorblind-friendly palette
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'
    ])

# Other Constants
RANDOM_STATE = 42
N_SEQUENCE_SITES = 566  # HA protein length for H3
SITE_OFFSET = 16  # Offset for sequence site numbering due to signal peptide cleavage

# --- Utility Functions ---

def create_output_dir(dir_path):
    """Creates the output directory if it doesn't exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Output directory ready: {dir_path}")
    except OSError as e:
        print(f"✗ Error creating directory {dir_path}: {e}")
        raise

def write_cache(data, filename):
    """Writes data to a pickle file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Data cached to: {filename}")
    except (IOError, pickle.PicklingError) as e:
        print(f"✗ Error writing cache file {filename}: {e}")

def read_cache(filename):
    """Reads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Cache loaded: {filename}")
        return data
    except FileNotFoundError:
        print(f"⚠ Cache file not found: {filename}")
        return None
    except (IOError, pickle.UnpicklingError) as e:
        print(f"✗ Error reading cache file {filename}: {e}")
        return None

# --- Geocoding Functions ---

def findGeocode(city, api_key):
    """Geocodes a city/country name using Google Maps Geocoding API.
    
    Args:
        city (str): City or country name to geocode
        api_key (str): Google Maps API key
        
    Returns:
        tuple: (address, (latitude, longitude)) or (None, (None, None)) if failed
    """
    if not api_key or api_key.startswith("AIza") == False:
        print("⚠ Warning: Invalid Google API key - using dummy coordinates (0.0, 0.0)")
        return ("DummyLocation, Country", (0.0, 0.0))
    try:
        geolocator = GoogleV3(api_key=api_key, timeout=10)
        location = geolocator.geocode(city)
        if location:
            return location.address, (location.latitude, location.longitude)
        else:
            return None, (None, None)
    except GeocoderTimedOut:
        return findGeocode(city, api_key)
    except Exception as e:
        return None, (None, None)

def get_locs(countries_series, api_key):
    """Geocodes unique countries in the pandas Series.
    
    Args:
        countries_series (pd.Series): Series containing country names
        api_key (str): Google Maps API key
        
    Returns:
        tuple: (location_map, country_lookup) dictionaries
    """
    unique_countries = countries_series.dropna().unique()
    print(f"Geocoding {len(unique_countries)} unique countries...")
    
    locs = {}
    cnt_lookup = {}
    geocoded_count = 0
    failed_count = 0
    for country in unique_countries:
        if country == 'nan':
            continue
        address, latlon = findGeocode(country, api_key)
        if address and latlon[0] is not None:
            locs[address] = latlon
            cnt_lookup[country] = address
            geocoded_count += 1
        else:
            failed_count += 1
            cnt_lookup[country] = None

    print(f"✓ Geocoding complete: {geocoded_count} successful, {failed_count} failed")
    return locs, cnt_lookup

def pull_locs_data_online(md_df, locs_file, cntri_file, api_key):
    """Performs online geocoding and caches results to JSON files.
    
    Args:
        md_df (pd.DataFrame): Metadata DataFrame with 'countries' column
        locs_file (str): Path to save location mappings
        cntri_file (str): Path to save country lookup mappings
        api_key (str): Google Maps API key
        
    Returns:
        tuple: (location_map, country_lookup) dictionaries
    """
    loc_map, country_lookup = get_locs(md_df['countries'], api_key)
    try:
        with open(locs_file, 'w') as f:
            json.dump(loc_map, f)
        print(f"✓ Location mappings saved to: {locs_file}")
    except IOError as e:
        print(f"✗ Error saving location map: {e}")

    try:
        with open(cntri_file, 'w') as f:
            json.dump(country_lookup, f)
        print(f"✓ Country lookup saved to: {cntri_file}")
    except IOError as e:
        print(f"✗ Error saving country lookup: {e}")

    return loc_map, country_lookup

def import_locs_data(locs_file, cntri_file):
    """Loads cached geocoding data from JSON files.
    
    Args:
        locs_file (str): Path to location mappings file
        cntri_file (str): Path to country lookup mappings file
        
    Returns:
        tuple: (location_map, country_lookup) dictionaries or (None, None) if failed
    """
    loc_map, country_lookup = None, None
    try:
        with open(locs_file, 'r') as f:
            loc_map = json.load(f)
        print(f"✓ Location mappings loaded from cache")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"⚠ Could not load location mappings: {e}")

    try:
        with open(cntri_file, 'r') as f:
            country_lookup = json.load(f)
        print(f"✓ Country lookup loaded from cache")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"⚠ Could not load country lookup: {e}")

    if loc_map is None or country_lookup is None:
        print("⚠ Geocoding cache incomplete - fresh geocoding may be needed")
        return None, None

    return loc_map, country_lookup

def append_latlong(md_df, loc_map, country_lookup):
    """Adds latitude and longitude columns to the DataFrame.
    
    Args:
        md_df (pd.DataFrame): Metadata DataFrame with 'countries' column
        loc_map (dict): Location mappings from geocoding
        country_lookup (dict): Country to address mappings
        
    Returns:
        pd.DataFrame: DataFrame with added 'latitude' and 'longitude' columns
    """
    latitudes = []
    longitudes = []
    missing_count = 0
    for country in md_df['countries']:
        address = country_lookup.get(str(country))
        if address and address in loc_map:
            lat, lon = loc_map[address]
            latitudes.append(lat)
            longitudes.append(lon)
        else:
            latitudes.append(np.nan)
            longitudes.append(np.nan)
            if pd.notna(country) and (country not in country_lookup or country_lookup.get(country) is None):
                missing_count += 1

    md_df['latitude'] = latitudes
    md_df['longitude'] = longitudes
    
    total_entries = len(md_df)
    geocoded_entries = total_entries - missing_count
    print(f"✓ Geographic coordinates added: {geocoded_entries}/{total_entries} entries geocoded")
    if missing_count > 0:
        print(f"  ⚠ {missing_count} entries missing coordinates due to failed geocoding")
    
    return md_df

# --- Data Loading and Preprocessing Functions ---

def import_data(metadata_path, sequence_path):
    """Loads metadata and sequence data from files.
    
    Args:
        metadata_path (str): Path to metadata CSV file (tilde-separated)
        sequence_path (str): Path to FASTA sequence file
        
    Returns:
        tuple: (metadata_df, sequence_dict) where sequence_dict maps IDs to sequences
    """
    print(f"Loading metadata from: {metadata_path}")
    try:
        md = pd.read_table(metadata_path, sep='~', dtype=str, lineterminator='\n')
        print(f"✓ Metadata loaded: {md.shape[0]:,} sequences, {md.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ Error: Metadata file not found at {metadata_path}")
        raise
    except Exception as e:
        print(f"✗ Error: Could not read metadata file {metadata_path}: {e}")
        raise

    print(f"Loading sequences from: {sequence_path}")
    try:
        sqs = list(SeqIO.parse(sequence_path, 'fasta'))
        ds = {record.id: str(record.seq) for record in sqs}
        print(f"✓ Sequences loaded: {len(sqs):,} records")
    except FileNotFoundError:
        print(f"✗ Error: Sequence file not found at {sequence_path}")
        raise
    except Exception as e:
        print(f"✗ Error: Could not read sequence file {sequence_path}: {e}")
        raise

    num_metadata_ids = md['Isolate_Id'].nunique()
    num_fasta_ids = len(ds)
    print(f"Data validation:")
    print(f"  Unique IDs in metadata: {num_metadata_ids:,}")
    print(f"  Unique IDs in sequences: {num_fasta_ids:,}")
    if num_metadata_ids == 0 or num_fasta_ids == 0:
        raise ValueError("Metadata or sequence file seems empty or lacks expected IDs.")
    if num_metadata_ids != md.shape[0]:
        print("  ⚠ Warning: Duplicate Isolate_Id found in metadata")

    return md, ds

def parse_pasg(passage_history_series):
    """Parses passage history strings using flupan library.
    
    Args:
        passage_history_series (pd.Series): Series containing passage history strings
        
    Returns:
        pd.DataFrame: DataFrame with columns ['pass1', 'pass2', 'pass3', 'pass4']
    """
    parsed_passages = []
    
    # Temporarily suppress flupan's verbose logging
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        pp = flupan.PassageParser()
        for passage_str in passage_history_series:
            passage_str = str(passage_str) if pd.notna(passage_str) else ''
            try:
                specific_passages = pp.parse_passage(passage_str).specific_passages
                parsed_passages.append(specific_passages + [None] * (4 - len(specific_passages)))
            except Exception as e:
                # Can't print while stdout is redirected, store for later
                parsed_passages.append([None] * 4)
    finally:
        # Restore stdout/stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return pd.DataFrame(parsed_passages, columns=['pass1', 'pass2', 'pass3', 'pass4'])

def parse_history(metadata_df):
    """Parses passage history for all records in the DataFrame.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame with 'Passage_History' column
        
    Returns:
        pd.DataFrame: DataFrame with parsed passage columns
    """
    print(f"Parsing passage history for {len(metadata_df):,} sequences...")
    if 'Passage_History' not in metadata_df.columns:
        raise KeyError("Metadata DataFrame must contain 'Passage_History' column.")
    metadata_df['Passage_History'] = metadata_df['Passage_History'].fillna('').astype(str)
    passage_df = parse_pasg(metadata_df['Passage_History'])
    print("✓ Passage history parsing complete")
    return passage_df

def split_location(location_string, pos):
    """Extracts location component at specified position.
    
    Args:
        location_string (str): Location string with '/' separators
        pos (int): Position index to extract (0-based)
        
    Returns:
        str: Location component or 'nan' if not available
    """
    if pd.isna(location_string):
        return 'nan'
    parts = [p.strip() for p in str(location_string).split('/')]
    return parts[pos] if len(parts) > pos else 'nan'

def consolidate_data(md, pa_df):
    """Combines metadata with parsed passage data and location components.
    
    Args:
        md (pd.DataFrame): Raw metadata DataFrame
        pa_df (pd.DataFrame): Parsed passage DataFrame
        
    Returns:
        tuple: (consolidated_df, first_date) where first_date is the earliest collection date
    """
    print("Consolidating metadata with location and time data...")
    
    # More efficient: split location once and extract all components
    location_parts = md['Location'].str.split('/', expand=True)
    location_parts = location_parts.map(lambda x: x.strip() if isinstance(x, str) else 'nan')
    
    # Assign column names based on available parts
    location_cols = ['continents', 'countries', 'states', 'cities']
    for i, col in enumerate(location_cols):
        if i < location_parts.shape[1]:
            md[col] = location_parts[i]
        else:
            md[col] = 'nan'

    md['Collection_Date'] = pd.to_datetime(md['Collection_Date'], errors='coerce')
    first_date = md['Collection_Date'].min()
    if pd.isna(first_date):
        raise ValueError("Could not determine the minimum collection date. Check date format/column.")
    print(f"  Date range: {first_date.strftime('%Y-%m-%d')} to {md['Collection_Date'].max().strftime('%Y-%m-%d')}")
    md['times'] = (md['Collection_Date'] - first_date).dt.days

    # More efficient: directly concatenate without creating intermediate DataFrames
    mdf = pd.concat([md.reset_index(drop=True),
                     pa_df.reset_index(drop=True)], axis=1)

    country_replacements = {
        'Congo, the Democatic Republic of': 'DRC',
        'Hong Kong (SAR)': 'Hong Kong',
        'Palestinian Territory': 'Palestine',
        'Jordan': 'Kingdom of Jordan'
    }
    mdf['countries'] = mdf['countries'].replace(country_replacements)
    mdf['countries'] = mdf['countries'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)

    print(f"✓ Metadata consolidation complete: {len(mdf):,} records")
    return mdf, first_date

def get_clean_multipass(md_full, sequence_dict, n_sequence_sites):
    """Filters and cleans data for machine learning model training.
    
    Args:
        md_full (pd.DataFrame): Full consolidated metadata
        sequence_dict (dict): Dictionary mapping sequence IDs to sequences
        n_sequence_sites (int): Expected sequence length
        
    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame ready for modeling
    """
    print("Filtering and cleaning data for model training...")
    
    # Convert passage columns to uppercase strings once
    pass_cols = ['pass1', 'pass2', 'pass3', 'pass4']
    for col in pass_cols:
        md_full[col] = md_full[col].fillna('').astype(str).str.upper()
    
    mdf = md_full[
        (md_full['pass1'] != 'UNKNOWNCELL') &
        (md_full['pass2'] != 'UNKNOWNCELL') &
        (md_full['pass3'] != 'UNKNOWNCELL') &
        (md_full['pass4'] != 'UNKNOWNCELL') &
        (md_full['pass1'] != '') &
        (md_full['Isolate_Id'].isin(sequence_dict.keys())) &
        (md_full['Collection_Date'].notna()) &
        (md_full['countries'].notna()) & (md_full['countries'] != 'nan')
    ].copy()

    print(f"  After initial filtering: {mdf.shape[0]:,} sequences")

    # Mark sequences with multiple passages as 'MULTI' (original logic)
    multi_condition = (mdf['pass2'] != '') & (mdf['pass2'] != 'UNPASSAGED')
    mdf.loc[multi_condition, 'pass1'] = 'MULTI'

    unpassaged_condition = (mdf['pass1'] == 'UNPASSAGED') | \
                           (mdf['pass2'] == 'UNPASSAGED') | \
                           (mdf['pass3'] == 'UNPASSAGED') | \
                           (mdf['pass4'] == 'UNPASSAGED')
    mdf.loc[unpassaged_condition, 'pass1'] = 'UNPASSAGED'
    
    # Filter out MULTI sequences
    print(f"  Removing {(mdf['pass1'] == 'MULTI').sum():,} sequences with multiple passages")
    mdf = mdf[mdf['pass1'] != 'MULTI'].copy()

    valid_seqs_idx = [
        idx for idx, iso_id in mdf['Isolate_Id'].items()
        if sequence_dict.get(iso_id) is not None and
           '-' not in sequence_dict[iso_id] and
           len(sequence_dict[iso_id]) == n_sequence_sites
    ]
    mdf = mdf.loc[valid_seqs_idx]
    print(f"  After sequence validation: {mdf.shape[0]:,} sequences (length={n_sequence_sites}, no gaps)")

    mdf.reset_index(inplace=True, drop=True)
    
    print(f"✓ Data filtering complete: {len(mdf):,} sequences ready for modeling")
    print("\nPassage type distribution:")
    passage_counts = mdf['pass1'].value_counts()
    for passage_type, count in passage_counts.items():
        percentage = (count / len(mdf)) * 100
        print(f"  {passage_type}: {count:,} ({percentage:.1f}%)")
    
    return mdf

# --- Sequence Analysis Functions ---

def Shannon_entropy(sequence_list):
    """Calculates Shannon entropy for a collection of items.
    
    Args:
        sequence_list: List or Series of items (e.g., amino acids)
        
    Returns:
        float: Shannon entropy in bits
    """
    if not isinstance(sequence_list, pd.Series):
        sequence_list = pd.Series(sequence_list)
    if sequence_list.empty or sequence_list.isna().all():
        return 0.0

    counts = collections.Counter(sequence_list.dropna())
    total_count = sum(counts.values())
    if total_count == 0:
        return 0.0
    probabilities = [count / total_count for count in counts.values()]
    return entropy(probabilities, base=2)

def find_site(onehot_encoder_categories, feature_index, site_offset):
    """
    Finds the protein site number and amino acid from a OneHotEncoder feature index.

    Args:
        onehot_encoder_categories: The `categories_` attribute of the fitted OneHotEncoder.
        feature_index: The numerical index of the feature in the transformed array.
        site_offset: Offset to subtract from FASTA position for protein numbering.

    Returns:
        tuple: (protein_site_number, amino_acid, all_options_at_site)
               Returns (-1, 'N/A', []) if index corresponds to a cleaved site (protein_site < 1).
               Returns (-2, 'NonSequenceFeature', []) if index is out of bounds of sequence features (e.g., time, lat, lon).
               Site number is 1-based protein numbering (FASTA position - offset).
    """
    current_index = 0
    for site_idx_fasta, category_list in enumerate(onehot_encoder_categories): # site_idx_fasta is 0-based index corresponding to FASTA s1, s2...
        num_categories = len(category_list)
        if feature_index < current_index + num_categories:
            amino_acid = category_list[feature_index - current_index]
            # Calculate protein site number: 1-based FASTA position - offset
            fasta_position = site_idx_fasta + 1
            protein_site_num = fasta_position - site_offset
            # Only return positive protein site numbers; sites within the offset region are considered non-protein (-1)
            if protein_site_num < 1:
                 protein_site_num = -1 # Indicator for cleaved sites
                 amino_acid = 'N/A'      # Label for cleaved sites

            return protein_site_num, amino_acid, category_list
        current_index += num_categories
    # If index is beyond sequence features (e.g., time, lat, lon)
    # We will assign the actual feature name later in the processing functions
    return -2, 'NonSequenceFeature', [] # Indicate feature is not a sequence site

# --- Feature Preparation ---

def prepare_features(metadata_df, sequence_dict, n_sequence_sites, include_time=True, fit_encoder=True, encoder=None):
    """Prepares features (sequences, time, lat/lon) for modeling."""
    # Reset index of the input metadata to ensure consistent 0, 1, 2... indexing internally
    metadata_df_reset = metadata_df.reset_index(drop=True)
    print(f"Preparing features for {len(metadata_df_reset):,} sequences (include_time={include_time})...")

    # Extract sequences corresponding to the metadata
    sequences = np.array([sequence_dict[iso_id] for iso_id in metadata_df_reset['Isolate_Id']])
    sequence_df = pd.DataFrame([list(seq) for seq in sequences],
                               columns=[f's{i+1}' for i in range(n_sequence_sites)])

    # Define feature columns
    feature_cols_seq = list(sequence_df.columns)
    feature_cols_other = []
    if include_time:
        feature_cols_other.append('times')
    feature_cols_other.extend(['latitude', 'longitude'])

    # Select relevant columns from the reset metadata
    X_other = metadata_df_reset[feature_cols_other].copy()

    # Handle potential missing values (impute with median for numerical features)
    for col in feature_cols_other:
        if X_other[col].isnull().any():
            median_val = X_other[col].median()
            if pd.isna(median_val):
                print(f"  ⚠ Cannot impute '{col}' (median is NaN) - filling with 0")
                median_val = 0 # Fallback imputation value
            else:
                print(f"  ⚠ Imputing missing values in '{col}' with median ({median_val:.4f})")
            X_other[col].fillna(median_val, inplace=True)

    # One-Hot Encode sequence features
    if fit_encoder:
        print("  Fitting One-Hot Encoder for sequence features...")
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
        X_seq_encoded = ohe.fit_transform(sequence_df[feature_cols_seq])
    elif encoder:
        print("  Transforming sequence features with existing encoder...")
        ohe = encoder
        X_seq_encoded = ohe.transform(sequence_df[feature_cols_seq])
    else:
        raise ValueError("Must either fit a new encoder or provide an existing one.")

    ohe_feature_names = ohe.get_feature_names_out(feature_cols_seq)

    # Combine encoded sequence features with other features
    X_final = pd.concat([
        pd.DataFrame(X_seq_encoded, columns=ohe_feature_names),
        X_other # Already has default 0, 1, ... index from metadata_df_reset
    ], axis=1)

    print(f"✓ Feature preparation complete: {X_final.shape[0]:,} samples, {X_final.shape[1]:,} features")
    return X_final, ohe

# --- Plotting Functions ---

def plot_collection_date_histogram(metadata_df, filename):
    """Plots a histogram of collection dates with publication-quality styling."""
    print(f"Generating collection date histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dates = pd.to_datetime(metadata_df['Collection_Date'], errors='coerce').dropna()
    dates_filtered = dates[dates > pd.Timestamp('1990-01-01')]

    if dates_filtered.empty:
        print("  ⚠ No dates found after 1990 - skipping histogram")
        plt.close()
        return

    # Plot with improved styling
    sns.histplot(data=dates_filtered, bins=32, ax=ax, 
                 color='#8FBBD9', edgecolor='#2B5F8E', alpha=0.8, linewidth=0.8)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Labels and styling
    ax.set_xlabel('collection date')
    ax.set_ylabel('number of sequences')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.grid(False, axis='x')
    
    # Rotate x-tick labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved: {filename}")

def plot_combined_variability(entropies, distinct_counts, filename):
    """Plots entropy, distinct AAs, and distribution together with publication-quality styling."""
    print(f"Generating sequence variability plots...")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False, constrained_layout=True)
    positions = range(len(entropies))

    # Plot 1: Shannon Entropy (scatter)
    axes[0].scatter(positions, entropies, s=8, color='#4477AA', alpha=0.7, marker='o', edgecolors='none')
    axes[0].set_ylabel('Shannon entropy')
    axes[0].set_ylim(bottom=0)
    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    axes[0].grid(False, axis='x')
    axes[0].set_xlim(-0.5, len(positions) - 0.5)

    # Plot 2: Distinct Amino Acids (scatter)
    axes[1].scatter(positions, distinct_counts, s=8, color='#EE6677', alpha=0.7, marker='o', edgecolors='none')
    axes[1].set_ylabel('number of distinct amino acids')
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel('sequence position')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    axes[1].grid(False, axis='x')
    axes[1].set_xlim(-0.5, len(positions) - 0.5)

    # Plot 3: Distribution of Distinct AA Counts (histogram)
    sns.histplot(distinct_counts, discrete=True, shrink=0.8, color='#228833', alpha=0.7, ax=axes[2])
    axes[2].set_xlabel('number of distinct amino acids')
    axes[2].set_ylabel('number of sites')
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    axes[2].grid(False, axis='x')
    max_distinct_count = int(np.max(distinct_counts)) if len(distinct_counts) > 0 else 0
    integer_ticks = np.arange(0, max_distinct_count + 1)
    axes[2].set_xticks(integer_ticks)
    axes[2].set_xticklabels([str(tick) for tick in integer_ticks])

    plt.savefig(filename)
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def plot_confusion_matrix(y_true, y_pred, classes, normalize, filename, title_suffix):
    """Plots and saves a confusion matrix with publication-quality styling."""
    print(f"Generating {'normalized' if normalize else 'raw count'} confusion matrix...")
    if normalize:
        # To calculate recall (true positives / all actual positives), we normalize by the true label counts.
        # The sklearn confusion_matrix with `normalize='true'` does this, creating a matrix where
        # true labels are rows and its rows sum to 1.
        cm = confusion_matrix(y_true, y_pred, normalize='true')

        # For a plot with 'predicted' on the y-axis and 'true' on the x-axis, where columns
        # (true labels) should sum to 1, we must transpose the matrix from sklearn.
        cm = cm.T
        fmt = '.2f'
    else:
        # For consistency, we also transpose the raw counts matrix to match the axis labels.
        cm = confusion_matrix(y_true, y_pred, normalize=None)
        cm = cm.T
        fmt = 'd'

    legend_name_map = { # Map internal names to display names
        'EGG': 'Egg', 'MDCK': 'MDCK', 'MONKEYKIDNEY': 'Monkey Kidney',
        'MULTI': 'Multiple', 'SIAT': 'SIAT-MDCK', 'UNPASSAGED': 'Unpassaged'
    }
    display_classes = [legend_name_map.get(c, c) for c in classes]
    
    # After transposing, the matrix rows correspond to predicted labels and columns to true labels.
    df_cm = pd.DataFrame(cm, index=display_classes, columns=display_classes)

    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Custom colormap with better contrast
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Plot heatmap. The axes now correctly correspond to the labels.
    sns.heatmap(df_cm, annot=True, fmt=fmt, cmap=cmap, 
                annot_kws={"size": 12}, 
                linewidths=0.5, linecolor='white', cbar=False, ax=ax)
    
    # Set aspect ratio to make cells square
    ax.set_aspect('equal', adjustable='box')
    
    # Labels are now consistent with the plotted data.
    ax.set_ylabel('predicted label', fontsize=12)
    ax.set_xlabel('true label', fontsize=12)
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved: {filename}")

def plot_shap_summary(shap_values, features, feature_names, filename, max_display=20, title=""):
    """Generates and saves a SHAP summary plot with publication-quality styling."""
    print(f"Generating SHAP summary plot...")
    plt.close('all') # Ensure clean slate

    try:
        # Set SHAP plot style
        plt.figure(figsize=(10, 8))
        
        if isinstance(shap_values, list): # Multi-class case
            print("  Using SHAP values for first class (multi-class model)")
            shap.summary_plot(shap_values[0], features, feature_names=feature_names,
                             plot_type="dot", max_display=max_display, show=False,
                             plot_size=(10, 8), color_bar=True, alpha=0.8)
        else: # Regression or binary classification
            shap.summary_plot(shap_values, features, feature_names=feature_names,
                             plot_type="dot", max_display=max_display, show=False,
                             plot_size=(10, 8), color_bar=True, alpha=0.8)

        # Get the current figure and improve styling
        fig = plt.gcf()
        if not fig.get_axes():
            print(f"  ⚠ SHAP plot generation failed - skipping save")
            plt.close('all')
            return

        # Improve styling of the current axes
        for ax in fig.get_axes():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")

    except Exception as e:
        print(f"  ✗ Error during SHAP plot generation: {e}")
        plt.close('all')

def plot_diverging_stacked_importance(site_agg_shap_per_class, site_agg_shap_total, class_names, filename, top_n=20):
    """
    Plots top N features as a diverging stacked horizontal bar chart.
    Positive SHAP values are plotted to the right of zero, negative to the left.
    """
    print(f"Generating stacked feature importance plot...")

    # Colorblind-friendly palette and legend mapping
    color_map = {
        'EGG': '#4477AA', 'MDCK': '#EE6677', 'MONKEYKIDNEY': '#228833',
        'MULTI': '#CCBB44', 'SIAT': '#66CCEE', 'UNPASSAGED': '#AA3377'
    }
    legend_name_map = {
        'EGG': 'Egg', 'MDCK': 'MDCK', 'MONKEYKIDNEY': 'Monkey Kidney',
        'MULTI': 'Multiple', 'SIAT': 'SIAT-MDCK', 'UNPASSAGED': 'Unpassaged'
    }

    # 1. Select top N features based on total importance magnitude
    top_features = site_agg_shap_total.head(top_n).index
    plot_data = site_agg_shap_per_class.loc[top_features].copy()

    # 2. Separate data into positive and negative contributions
    pos_data = plot_data[plot_data > 0].fillna(0)
    neg_data = plot_data[plot_data < 0].fillna(0)

    # 3. Ensure consistent column order and sort for plotting (most important at top)
    class_names_list = [str(c) for c in class_names]
    ordered_plot_cols = [c for c in class_names_list if c in plot_data.columns]
    ordered_plot_cols += [c for c in plot_data.columns if c not in ordered_plot_cols]

    pos_data = pos_data[ordered_plot_cols].iloc[::-1]
    neg_data = neg_data[ordered_plot_cols].iloc[::-1]
    plot_colors = [color_map.get(col, '#BBBBBB') for col in ordered_plot_cols]

    # 4. Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot positive and negative bars on the same axes
    pos_data.plot(kind='barh', stacked=True, ax=ax, color=plot_colors, edgecolor='white', linewidth=0.5, legend=False)
    neg_data.plot(kind='barh', stacked=True, ax=ax, color=plot_colors, edgecolor='white', linewidth=0.5, legend=False)

    # 5. Formatting and Aesthetics
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.set_ylabel("most important predictive variables", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add a central line at zero
    ax.axvline(0, color='black', linestyle='-', linewidth=1.0, zorder=3)

    # Clean up y-tick labels
    current_labels = [item.get_text() for item in ax.get_yticklabels()]
    new_labels = [label.replace('_', '') for label in current_labels]
    ax.set_yticklabels(new_labels)

    # Create a single, clean legend
    handles, labels = ax.get_legend_handles_labels()
    # The legend gets populated twice (once for pos, once for neg), so we deduplicate
    unique_handles_labels = dict(zip(labels, handles))
    display_labels = [legend_name_map.get(lbl, lbl) for lbl in unique_handles_labels.keys()]
    ax.legend(handles=unique_handles_labels.values(), labels=display_labels, title='Passage Type',
              title_fontsize='12', fontsize='10', frameon=False,
              loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Grid and spines
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='gray', zorder=0)
    ax.grid(axis='y', linestyle='')
    ax.spines[['top', 'right', 'left']].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def plot_diverging_stacked_importance_regressor(shap_df, ranking_series, filename, top_n=20):
    """
    Plots top N sites for a regressor as a diverging stacked horizontal bar chart.
    Sites are ranked by the sum of absolute SHAP values.
    The plot shows the signed contribution of each amino acid at that site.
    """
    print(f"Generating regressor stacked importance plot...")

    # 1. Define a color map for all 20 standard amino acids using Tableau20
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    cmap = plt.cm.get_cmap('tab20', len(amino_acids))
    color_map = {aa: cmap(i) for i, aa in enumerate(amino_acids)}
    # Add colors for non-amino acid features if they appear
    other_features = ['times', 'latitude', 'longitude']
    other_colors = ['#8C564B', '#9467BD', '#E377C2'] # From Tableau10 continuation
    for i, feat in enumerate(other_features):
        if feat not in color_map:
            color_map[feat] = other_colors[i]


    # 2. Select top N features based on the ranking series (sum of absolute SHAP values)
    top_sites = ranking_series.head(top_n).index
    plot_data_filtered = shap_df[shap_df['protein_site'].isin(top_sites)].copy()

    # 3. Pivot the data to get sites as rows and amino acids as columns
    pivot_df = plot_data_filtered.pivot_table(
        index='protein_site',
        columns='amino_acid',
        values='mean_shap',
        fill_value=0
    )

    # Reorder the pivoted data to match the ranking order (most important at top)
    pivot_df = pivot_df.reindex(top_sites).iloc[::-1] # Reverse for barh

    # 4. Separate data into positive and negative contributions
    pos_data = pivot_df[pivot_df > 0].fillna(0)
    neg_data = pivot_df[pivot_df < 0].fillna(0)

    # 5. Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get consistent colors for the amino acids present in the data
    plot_colors = [color_map.get(col, '#BBBBBB') for col in pivot_df.columns]

    # Plot positive and negative bars on the same axes
    pos_data.plot(kind='barh', stacked=True, ax=ax, color=plot_colors, edgecolor='white', linewidth=0.5, legend=False)
    neg_data.plot(kind='barh', stacked=True, ax=ax, color=plot_colors, edgecolor='white', linewidth=0.5, legend=False)

    # 6. Formatting and Aesthetics
    ax.set_xlabel("SHAP value (impact on date prediction)", fontsize=12)
    ax.set_ylabel("most important predictive sites", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.0, zorder=3)

    # Create a single, clean legend for amino acids
    handles = [plt.Rectangle((0,0),1,1, color=color_map[aa]) for aa in pivot_df.columns if aa in color_map]
    labels = [aa for aa in pivot_df.columns if aa in color_map]
    ax.legend(handles=handles, labels=labels, title='Amino Acid',
              title_fontsize='12', fontsize='10', frameon=False,
              loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax.grid(axis='x', linestyle='--', alpha=0.4, color='gray', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def generate_contingency_table(protein_site_num, metadata_df, sequence_dict, site_offset):
    """
    Generates a contingency table for a given protein site, prints it,
    and performs a Chi-squared test.

    Args:
        protein_site_num (int): The protein site number (1-based, after offset).
        metadata_df (pd.DataFrame): DataFrame with 'Isolate_Id' and 'pass1' columns.
        sequence_dict (dict): Dictionary mapping Isolate_Id to sequence string.
        site_offset (int): The offset for sequence site numbering.
    """
    print(f"\n=== Contingency Table Analysis: Site {protein_site_num} ===")

    # Validate input
    if not isinstance(protein_site_num, int) or protein_site_num < 1:
        print("✗ Error: protein_site_num must be a positive integer")
        return

    # Convert protein site number to 0-based sequence index
    # FASTA position = protein Site + Offset
    fasta_position = protein_site_num + site_offset
    sequence_index = fasta_position - 1

    # Extract passage type and the specific amino acid for each sequence - vectorized approach
    isolate_ids = metadata_df['Isolate_Id'].values
    passage_types = metadata_df['pass1'].values
    
    amino_acids = []
    valid_indices = []
    
    for i, isolate_id in enumerate(isolate_ids):
        sequence = sequence_dict.get(isolate_id)
        if sequence and 0 <= sequence_index < len(sequence):
            amino_acids.append(sequence[sequence_index])
            valid_indices.append(i)
    
    if not amino_acids:
        print("✗ Error: No data available for contingency table analysis")
        return
    
    # Create a DataFrame directly from arrays
    analysis_df = pd.DataFrame({
        'passage': passage_types[valid_indices],
        'amino_acid': amino_acids
    })
    contingency_table = pd.crosstab(analysis_df['passage'], analysis_df['amino_acid'])

    print("\nContingency Table (observed counts):")
    print(contingency_table)

    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print("\nChi-squared independence test:")
        print(f"  χ² = {chi2:.4f}")
        print(f"  p-value = {p:.4g}")
        print(f"  df = {dof}")
        if p < 0.001:
            print("  Result: Highly significant association (p < 0.001)")
        elif p < 0.05:
            print("  Result: Significant association (p < 0.05)")
        else:
            print("  Result: No significant association (p ≥ 0.05)")
    except ValueError as e:
        print(f"\n⚠ Chi-squared test failed: {e}")
        print("  (Often due to sparse data or empty cells)")

def plot_feature_importance_bar(importance_series, filename, top_n=20, xlabel="Overall feature importance"):
    """
    Plots top N features as a diverging horizontal bar chart based on signed importance values.
    Assumes input `importance_series` has already been sorted by magnitude.
    """
    print(f"Generating feature importance bar plot...")
    if not isinstance(importance_series, pd.Series) or importance_series.empty:
        print(f"  ⚠ Empty importance data - skipping plot")
        return

    # Select top N from the pre-sorted series and reverse for plotting (highest importance at top)
    plot_data = importance_series.head(top_n).iloc[::-1]
    if plot_data.empty:
        print(f"  ⚠ No data remaining after selecting top {top_n} - skipping plot")
        plt.close()
        return

    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(8, 8)) # Adjusted size for better label visibility

    # Assign colors based on sign (using the same palette as the stacked plot)
    colors = ['#EE6677' if x < 0 else '#4477AA' for x in plot_data.values]

    # Plot horizontal bars
    ax.barh(plot_data.index, plot_data.values, color=colors,
            edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add a vertical line at x=0 for reference
    ax.axvline(0, color='black', linestyle='-', linewidth=1.0, zorder=0)

    # Labels and styling
    ax.set_ylabel("amino acid site", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Grid and spines
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray', zorder=-1)
    ax.grid(axis='y', linestyle='')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved: {filename}")

def plot_predicted_vs_actual(pred_actual_df, filename, first_date):
    """Plots predicted vs actual dates with publication-quality styling."""
    print(f"Generating predicted vs actual date plot...")
    # Convert days since first date back to actual dates
    pred_actual_df['real_date'] = pred_actual_df['real_time_days'].apply(lambda d: first_date + timedelta(days=int(round(d))))
    pred_actual_df['predicted_date'] = pred_actual_df['predicted_time_days'].apply(lambda d: first_date + timedelta(days=int(round(d))))

    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot scatter with improved styling
    sns.scatterplot(data=pred_actual_df, x='real_date', y='predicted_date', 
                    alpha=0.6, s=30, color='#4477AA', 
                    edgecolor='#2B5F8E', linewidth=0.5, ax=ax)
    
    # Add identity line
    min_date = pred_actual_df[['real_date', 'predicted_date']].min().min()
    max_date = pred_actual_df[['real_date', 'predicted_date']].max().max()
    ax.plot([min_date, max_date], [min_date, max_date], 
            color='#EE6677', linestyle='--', alpha=0.7, linewidth=1.5, 
            label='y=x')
    
    # Labels and styling
    ax.set_xlabel('actual collection date', fontsize=12)
    ax.set_ylabel('predicted collection date', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Legend and grid
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved: {filename}")

def plot_residuals(pred_actual_df, filename_prefix):
    """Plots residual distribution and residuals vs predicted with publication-quality styling."""
    print(f"Generating residual analysis plots...")
    pred_actual_df['residuals_log'] = pred_actual_df['real_log_time'] - pred_actual_df['predicted_log_time']

    # Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter with improved styling
    sns.scatterplot(data=pred_actual_df, x='predicted_log_time', y='residuals_log', 
                    alpha=0.6, s=30, color='#4477AA', 
                    edgecolor='#2B5F8E', linewidth=0.5, ax=ax)
    
    # Add zero line
    ax.axhline(0, color='#EE6677', linestyle='--', linewidth=1.5)
    
    # Labels and styling
    ax.set_xlabel('predicted log(time + 1)', fontsize=12)
    ax.set_ylabel('residuals (log scale)', fontsize=12)
    
    # Grid and background
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_residuals_vs_predicted.pdf")
    plt.close()

    # Histogram of Residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with improved styling
    sns.histplot(pred_actual_df['residuals_log'], kde=True, bins=30, 
                 color='#4477AA', alpha=0.7, edgecolor='#2B5F8E', 
                 linewidth=0.5, ax=ax)
    
    # Labels and styling
    ax.set_xlabel('residuals (log scale)', fontsize=12)
    
    # Grid and background
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.grid(False, axis='x')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_residuals_histogram.pdf")
    plt.close()
    print(f"  ✓ Saved residual plots")

def plot_correlation_scatter(corr_df, x_col, y_col, xlabel, ylabel, filename, use_ranks=True):
    """Plots a scatter plot for correlation analysis with publication-quality styling.
    
    Args:
        corr_df: DataFrame with columns to correlate
        x_col: Name of x column
        y_col: Name of y column  
        xlabel: Label for x axis
        ylabel: Label for y axis
        filename: Output filename
        use_ranks: If True, plot rank-transformed data for Spearman correlation
    """
    print(f"Generating correlation plot ({y_col} vs {x_col})...")
    plot_data = corr_df.dropna(subset=[x_col, y_col])
    if plot_data.empty:
        print(f"  ⚠ No overlapping data for {y_col} vs {x_col} - skipping plot")
        return None # Return None if no data

    # Calculate Spearman correlation first
    if len(plot_data) > 1:
        corr, p_val = spearmanr(plot_data[x_col], plot_data[y_col])
        print(f"  Spearman correlation: ρ = {corr:.3f}, p = {p_val:.3g}")
    else:
        print(f"  ⚠ Insufficient data ({len(plot_data)} points) for correlation analysis")
        return None, None
    
    # If using ranks, transform data for visualization
    if use_ranks:
        plot_data = plot_data.copy()
        plot_data[f'{x_col}_rank'] = rankdata(plot_data[x_col])
        plot_data[f'{y_col}_rank'] = rankdata(plot_data[y_col])
        x_plot = f'{x_col}_rank'
        y_plot = f'{y_col}_rank'
        xlabel = f"{xlabel} (rank)"
        ylabel = f"{ylabel} (rank)"
    else:
        x_plot = x_col
        y_plot = y_col
    
    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot scatter with improved styling
    sns.scatterplot(data=plot_data, x=x_plot, y=y_plot, 
                    color='#4477AA', alpha=0.7, 
                    edgecolor='#2B5F8E', linewidth=0.5, s=40, ax=ax)
    
    # Add regression line on ranked data
    if len(plot_data) > 2:
        sns.regplot(data=plot_data, x=x_plot, y=y_plot, 
                    scatter=False, ci=None, line_kws={'color': '#EE6677', 'lw': 1.5}, 
                    ax=ax)
    
    # Labels and styling
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    ax.tick_params(labelsize=14)
    
    # Add correlation text to plot (positioned at 0.1 from the left)
    ax.text(0.1, 0.95, f'ρ = {corr:.3f}\np = {p_val:.3g}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Grid and background
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  ✓ Saved rank correlation plot: {filename}")

    return corr, p_val

def calculate_frequency_user_style(group):
    """
    Calculates frequency based on user's previous style (np.unique).
    Args:
        group (pd.Series): Series of amino acids for one site in one time bin.
    Returns:
        pd.Series: Series with amino acids as index and frequencies as values.
    """
    if group.empty:
        return pd.Series(dtype=float)

    group_cleaned = group.dropna()
    if group_cleaned.empty:
        return pd.Series(dtype=float)

    counts_val, counts_num = np.unique(group_cleaned, return_counts=True)
    total_count = np.sum(counts_num)

    if total_count == 0:
         return pd.Series(dtype=float)

    freq_series = pd.Series(counts_num / total_count, index=counts_val)
    return freq_series

def plot_frequency_over_time(data_df, site_col, time_bin_col, aa_categories, filename, title):
    """Calculates and plots amino acid frequency over time bins with publication-quality styling."""
    print(f"Generating frequency over time plot...")

    all_aas = set()
    for cat_list in aa_categories:
        all_aas.update(cat_list)

    try:
        # Calculate frequencies
        print(f"  Calculating frequencies for {title}...")
        freq_df = data_df.groupby(time_bin_col, observed=False)[site_col].apply(calculate_frequency_user_style).unstack(fill_value=0).reset_index()
        print(f"  ✓ Frequency calculation complete")

        freq_df_melt = freq_df.melt(id_vars=time_bin_col, var_name='AA', value_name='frequency')

        # Filter out AAs that are always zero frequency
        aa_totals = freq_df_melt.groupby('AA')['frequency'].sum()
        aas_to_plot = aa_totals[aa_totals > 0].index.tolist()
        plot_data = freq_df_melt[freq_df_melt['AA'].isin(aas_to_plot)]

        if plot_data.empty:
            print(f"  ⚠ No frequency data to plot for {title} - skipping")
            return

        # Sort data by time bin for correct line plotting
        plot_data = plot_data.sort_values(by=time_bin_col)

        # Create figure with improved styling
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Use a colorblind-friendly palette
        color_palette = sns.color_palette("colorblind", n_colors=len(aas_to_plot))
        
        # Plot lines with improved styling
        sns.lineplot(
            data=plot_data, x=time_bin_col, y='frequency', hue='AA',
            hue_order=sorted(aas_to_plot), marker='o', palette=color_palette,
            linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=0.5,
            ax=ax
        )
        
        # Improve legend with larger font
        ax.legend(title='amino acid', bbox_to_anchor=(1.02, 0.5), 
                  loc='center left', borderaxespad=0., frameon=False, 
                  fontsize=22, title_fontsize=22)
        
        # Labels and styling with larger fonts
        ax.set_xlabel('time', fontsize=22)
        ax.set_ylabel('frequency', fontsize=22)
        ax.set_ylim(0, 1.05)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Grid and spines
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.savefig(filename)
        plt.close()
        print(f"  ✓ Saved: {filename}")

    except Exception as e:
        print(f"  ✗ Error generating frequency plot for {title}: {e}")
        plt.close('all')

# --- SHAP Processing Functions ---

def process_shap_values_classifier(shap_values_data, X_test_df, encoder_cats, class_names, site_offset):
    """
    Processes SHAP values for a classifier to aggregate importance by protein site,
    preserving the direction (positive/negative contribution) of SHAP values.
    """
    num_classes = len(class_names)
    mean_shap_per_class = []

    if isinstance(shap_values_data, list):
        if len(shap_values_data) != num_classes:
             raise ValueError(f"SHAP list length ({len(shap_values_data)}) != num classes ({num_classes}).")
        mean_shap_per_class = [shap_values_data[i].mean(axis=0) for i in range(num_classes)]
    elif isinstance(shap_values_data, np.ndarray) and shap_values_data.ndim == 3:
        if shap_values_data.shape[2] != num_classes:
             raise ValueError(f"SHAP array classes dim ({shap_values_data.shape[2]}) != num classes ({num_classes}).")
        mean_shap_per_class = [shap_values_data[:, :, i].mean(axis=0) for i in range(num_classes)]
    else:
        raise TypeError(f"Unsupported SHAP values format: {type(shap_values_data)}. Expected list or 3D numpy array.")

    if not mean_shap_per_class:
         raise ValueError("Could not calculate SHAP values.")

    feature_importances = pd.DataFrame(
        np.array(mean_shap_per_class).T,
        index=X_test_df.columns,
        columns=class_names
    )

    feature_importances['total_shap_magnitude'] = feature_importances[class_names].abs().sum(axis=1)
    feature_importances['total_shap_signed_sum'] = feature_importances[class_names].sum(axis=1)

    # Add protein site and amino acid information
    num_total_features = X_test_df.shape[1]
    site_info = [find_site(encoder_cats, i, site_offset) for i in range(num_total_features)]
    num_seq_features_ohe = sum(len(cat) for cat in encoder_cats)
    num_non_seq_features = num_total_features - num_seq_features_ohe
    if num_non_seq_features > 0:
        non_seq_feature_names = X_test_df.columns[-num_non_seq_features:].tolist()
        for i in range(num_non_seq_features):
             non_seq_idx = num_seq_features_ohe + i
             if 0 <= non_seq_idx < len(site_info):
                 site_info[non_seq_idx] = (-2 - i, non_seq_feature_names[i], [])

    feature_importances['protein_site'] = [s[0] for s in site_info]
    feature_importances['aa'] = [s[1] for s in site_info]
    feature_importances['site_aa_label'] = [
        f"{s[0]}{s[1]}" if s[0] >= 1 else s[1] for s in site_info
    ]

    site_agg_shap_per_class = feature_importances.groupby('site_aa_label')[class_names].sum()
    site_agg_shap_total = site_agg_shap_per_class.abs().sum(axis=1).sort_values(ascending=False)

    return feature_importances, site_agg_shap_total, site_agg_shap_per_class

def process_shap_values_regressor(shap_values, features_df, ohe_categories, site_offset):
    """
    Processes SHAP values for a regressor, preserving the sign of the contributions.
    Returns:
        shap_df (pd.DataFrame): DataFrame with signed mean_shap for each feature (e.g., 158K).
        site_agg_shap (pd.Series): Series with sum of signed SHAP values per protein site.
        site_agg_shap_magnitude (pd.Series): Series with sum of ABSOLUTE SHAP values per protein site, for ranking.
    """
    if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
        raise ValueError("shap_values for regressor must be a 2D numpy array (samples x features)")

    mean_shap = np.mean(shap_values, axis=0)
    feature_names = features_df.columns

    if len(feature_names) != len(mean_shap):
         raise ValueError(f"Mismatch between feature names ({len(feature_names)}) and SHAP values ({len(mean_shap)})")

    shap_df = pd.DataFrame({'feature': feature_names, 'mean_shap': mean_shap})
    
    shap_df['mean_abs_shap'] = shap_df['mean_shap'].abs()

    try:
        site_info = [find_site(ohe_categories, i, site_offset) for i in range(len(feature_names))]
        num_seq_features_ohe = sum(len(cat) for cat in ohe_categories)
        num_non_seq_features = len(feature_names) - num_seq_features_ohe
        if num_non_seq_features > 0:
            non_seq_feature_names = features_df.columns[-num_non_seq_features:].tolist()
            for i in range(num_non_seq_features):
                non_seq_idx = num_seq_features_ohe + i
                if 0 <= non_seq_idx < len(site_info):
                    site_info[non_seq_idx] = (-2 - i, non_seq_feature_names[i], [])
        shap_df['protein_site'] = [s[0] for s in site_info]
        shap_df['amino_acid'] = [s[1] for s in site_info]
        shap_df['feature_label'] = shap_df.apply(
            lambda row: f"{int(row['protein_site'])}{row['amino_acid']}" if pd.notna(row['protein_site']) and row['protein_site'] >= 1 else row['amino_acid'],
            axis=1
        )
    except Exception as e:
        print(f"ERROR during site/label processing in process_shap_values_regressor: {e}")
        raise

    # Sort the main DataFrame by magnitude of mean SHAP value
    shap_df = shap_df.reindex(shap_df['mean_shap'].abs().sort_values(ascending=False).index)

    # Filter for valid protein sites before aggregation
    seq_sites_shap = shap_df[shap_df['protein_site'] >= 1].copy()
    seq_sites_shap['protein_site'] = seq_sites_shap['protein_site'].astype(int)

    # Aggregate by protein site
    if not seq_sites_shap.empty:
        site_agg_shap = seq_sites_shap.groupby('protein_site')['mean_shap'].sum()

        site_agg_shap_magnitude = seq_sites_shap.groupby('protein_site')['mean_shap'].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
    else:
        print("Warning: No protein sequence site features found for site aggregation.")
        site_agg_shap = pd.Series(dtype=float)
        site_agg_shap_magnitude = pd.Series(dtype=float)

    return shap_df, site_agg_shap, site_agg_shap_magnitude

def save_shap_values_classifier_to_csv(shap_df, entropies, df_erates, output_dir, model_name, site_offset):
    """
    Saves SHAP values from classifier model to CSV with required columns.
    
    Args:
        shap_df (pd.DataFrame): DataFrame with SHAP values from process_shap_values_classifier
        entropies (list): List of entropy values for each gene site
        df_erates (pd.DataFrame): DataFrame with leisr values (original_site, erate columns)
        output_dir (str): Directory to save CSV file
        model_name (str): Name of the model (for filename)
        site_offset (int): Offset to convert gene site to protein site
    """
    # Filter for sequence-based features (protein_site >= 1)
    seq_features = shap_df[shap_df['protein_site'] >= 1].copy()
    
    if seq_features.empty:
        print(f"Warning: No sequence features found for {model_name} model CSV export")
        return
    
    # Create output DataFrame
    output_rows = []
    
    for _, row in seq_features.iterrows():
        gene_site = int(row['protein_site']) + site_offset
        protein_site = int(row['protein_site'])
        aa = row['aa']
        shap_value = row['total_shap_magnitude']  # Use magnitude for classifier
        
        # Get entropy for this gene site
        entropy_val = entropies[gene_site - 1] if 0 <= gene_site - 1 < len(entropies) else np.nan
        
        # Get leisr value for this gene site
        leisr_val = np.nan
        if df_erates is not None:
            leisr_row = df_erates[df_erates['site'] == gene_site]
            if not leisr_row.empty:
                leisr_val = leisr_row['erate'].iloc[0]
        
        output_rows.append({
            'gene_site': gene_site,
            'protein_site': protein_site,
            'aa': aa,
            'shap': shap_value,
            'leisr': leisr_val,
            'entropy': entropy_val
        })
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(['gene_site', 'aa'])
    
    filename = os.path.join(output_dir, f'{model_name.lower()}_shap_values.csv')
    output_df.to_csv(filename, index=False)
    print(f"✓ Saved {model_name} SHAP values to {filename} ({len(output_df)} rows)")

def save_shap_values_regressor_to_csv(shap_df, entropies, df_erates, output_dir, model_name, site_offset):
    """
    Saves SHAP values from regressor model to CSV with required columns.
    
    Args:
        shap_df (pd.DataFrame): DataFrame with SHAP values from process_shap_values_regressor
        entropies (list): List of entropy values for each gene site
        df_erates (pd.DataFrame): DataFrame with leisr values (original_site, erate columns)
        output_dir (str): Directory to save CSV file
        model_name (str): Name of the model (for filename)
        site_offset (int): Offset to convert gene site to protein site
    """
    # Filter for sequence-based features (protein_site >= 1)
    seq_features = shap_df[shap_df['protein_site'] >= 1].copy()
    
    if seq_features.empty:
        print(f"Warning: No sequence features found for {model_name} model CSV export")
        return
    
    # Create output DataFrame
    output_rows = []
    
    for _, row in seq_features.iterrows():
        gene_site = int(row['protein_site']) + site_offset
        protein_site = int(row['protein_site'])
        aa = row['amino_acid']
        shap_value = row['mean_shap']  # Use signed value for regressor
        
        # Get entropy for this gene site
        entropy_val = entropies[gene_site - 1] if 0 <= gene_site - 1 < len(entropies) else np.nan
        
        # Get leisr value for this gene site
        leisr_val = np.nan
        if df_erates is not None:
            leisr_row = df_erates[df_erates['site'] == gene_site]
            if not leisr_row.empty:
                leisr_val = leisr_row['erate'].iloc[0]
        
        output_rows.append({
            'gene_site': gene_site,
            'protein_site': protein_site,
            'aa': aa,
            'shap': shap_value,
            'leisr': leisr_val,
            'entropy': entropy_val
        })
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(['gene_site', 'aa'])
    
    filename = os.path.join(output_dir, f'{model_name.lower()}_shap_values.csv')
    output_df.to_csv(filename, index=False)
    print(f"✓ Saved {model_name} SHAP values to {filename} ({len(output_df)} rows)")

# --- Model Training and Evaluation ---

def run_bayes_opt(X_train, y_train, log_path, cache_path):
    """Performs Bayesian optimization for LightGBM classifier hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        log_path (str): Path to save optimization log
        cache_path (str): Path to cache optimization results
        
    Returns:
        dict: Best hyperparameters found
    """
    print("Running Bayesian optimization for hyperparameter tuning...")

    def lgb_evaluate_passage(minData, maxDepth, numLeaves, nEstimators, regLambda):
        params = {
            'boosting_type': 'gbdt', 'objective': 'multiclass', 'metric': 'multi_logloss',
            'feature_fraction': 1.0, 'learning_rate': 0.1, 'reg_alpha': 0,
            'reg_lambda': float(regLambda), 'min_child_samples': int(minData),
            'n_estimators': int(nEstimators), 'num_leaves': int(numLeaves),
            'max_depth': int(maxDepth), 'class_weight': 'balanced',
            'n_jobs': -1, 'random_state': RANDOM_STATE, 'verbosity': -1, 'force_row_wise': True
        }
        clf = LGBMClassifier(**params)
        # Use Stratified K-Fold for balanced accuracy evaluation in classification
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        return np.mean(scores)

    lgbBO = BayesianOptimization(
        lgb_evaluate_passage,
        pbounds={
            'minData': (2, 30), 'maxDepth': (5, 15), 'numLeaves': (100, 1000),
            'nEstimators': (100, 2000), 'regLambda': (100, 1000)
        },
        random_state=RANDOM_STATE, verbose=2
    )
    # JSONLogger not available in newer bayes_opt versions
    # Logging is handled internally by BayesianOptimization

    lgbBO.maximize(init_points=50, n_iter=100)

    best_params = lgbBO.max['params']
    best_score = lgbBO.max['target']
    print(f"✓ Bayesian optimization complete - best score: {best_score:.4f}")
    write_cache(lgbBO.res, cache_path)
    return best_params

def load_best_bayes_params(cache_path):
    """Loads best hyperparameters from cached Bayesian optimization results.
    
    Args:
        cache_path (str): Path to cached results file
        
    Returns:
        dict or None: Best parameters if found, None otherwise
    """
    print("Loading cached Bayesian optimization results...")
    bayes_results_list = read_cache(cache_path)

    if isinstance(bayes_results_list, list) and len(bayes_results_list) > 0:
        try:
            best_result_dict = max(bayes_results_list, key=lambda x: x.get('target', -np.inf))
            best_params = best_result_dict.get('params')
            if best_params and isinstance(best_params, dict):
                print(f"✓ Best parameters loaded from cache (score={best_result_dict.get('target', 'N/A'):.4f})")
                return best_params
            else:
                print("⚠ Cache loaded but could not extract valid parameters")
        except (KeyError, TypeError, ValueError) as e:
            print(f"⚠ Error processing cached results: {e}")
    elif isinstance(bayes_results_list, dict):
        print("✓ Parameters loaded from cache")
        return bayes_results_list
    else:
        print("⚠ Cache not found or invalid format")

    return None

def standardize_lgbm_params(params_dict):
    """Converts and standardizes LightGBM parameters to proper format.
    
    Args:
        params_dict (dict): Dictionary of raw parameters
        
    Returns:
        dict: Standardized parameters for LightGBM
    """
    if not params_dict or not isinstance(params_dict, dict):
        raise ValueError("Invalid input: params_dict must be a non-empty dictionary.")

    standardized = {}
    try:
        standardized['reg_lambda'] = float(params_dict.get('regLambda', params_dict.get('reg_lambda', 0.0)))
        standardized['reg_alpha'] = float(params_dict.get('regAlpha', params_dict.get('reg_alpha', 0.0)))
        standardized['num_leaves'] = int(params_dict.get('numLeaves', params_dict.get('num_leaves', 31)))
        standardized['n_estimators'] = int(params_dict.get('nEstimators', params_dict.get('n_estimators', 100)))
        standardized['min_child_samples'] = int(params_dict.get('minData', params_dict.get('min_child_samples', params_dict.get('min_data_in_leaf', 20))))
        standardized['max_depth'] = int(params_dict.get('maxDepth', params_dict.get('max_depth', -1)))

        standardized['learning_rate'] = float(params_dict.get('learning_rate', 0.1))
        standardized['colsample_bytree'] = float(params_dict.get('colsample_bytree', 1.0))
        standardized['objective'] = params_dict.get('objective', 'multiclass')
        standardized['metric'] = params_dict.get('metric', 'multi_logloss')
        standardized['boosting_type'] = params_dict.get('boosting_type', 'gbdt')
        standardized['class_weight'] = params_dict.get('class_weight', None)
        standardized['n_jobs'] = params_dict.get('n_jobs', -1)
        standardized['random_state'] = params_dict.get('random_state', RANDOM_STATE)
        standardized['verbosity'] = params_dict.get('verbosity', params_dict.get('verbose', -1))
        standardized['force_row_wise'] = True  # Suppress warning about row/column access

        current_objective = standardized.get('objective', '').lower()
        if 'regression' in current_objective or current_objective in ['regression_l1', 'regression_l2', 'mae', 'mse', 'rmse']:
            standardized['metric'] = params_dict.get('metric', 'rmse')
            standardized.pop('class_weight', None)
        elif 'multiclass' in current_objective:
            standardized['metric'] = params_dict.get('metric', 'multi_logloss')

        return standardized

    except (KeyError, ValueError, TypeError) as e:
        print(f"Error standardizing parameters: {e}. Original params: {params_dict}")
        raise ValueError("Parameter standardization failed.") from e

def train_evaluate_classifier(X_train, y_train, X_test, y_test, params):
    """Trains and evaluates a LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        params (dict): Model parameters
        
    Returns:
        tuple: (trained_model, class_labels)
    """
    print("\n=== Training LightGBM Classifier ===")
    model = LGBMClassifier(**params)
    
    # Suppress warnings during fit - redirect stderr completely
    import sys
    import io
    
    # Save original stderr
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X_train, y_train)
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = old_stderr

    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)
    classes = model.classes_

    # Calculate metrics
    test_acc = accuracy_score(y_test, pred_test)
    train_acc = accuracy_score(y_train, pred_train)
    test_bal_acc = balanced_accuracy_score(y_test, pred_test)
    train_bal_acc = balanced_accuracy_score(y_train, pred_train)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred_test, average='macro', zero_division=0)
    
    print(f"✓ Model training complete")
    print(f"\nClassification Performance:")
    print(f"  Test Set:  Accuracy = {test_acc:.3f}, Balanced Accuracy = {test_bal_acc:.3f}")
    print(f"  Train Set: Accuracy = {train_acc:.3f}, Balanced Accuracy = {train_bal_acc:.3f}")
    print(f"  Macro-averaged: Precision = {prec:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}")

    return model, classes

def train_evaluate_regressor(X_train, y_train, X_test, y_test, params):
    """Trains and evaluates a LightGBM regressor.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        params (dict): Model parameters
        
    Returns:
        tuple: (trained_model, prediction_results_df)
    """
    print("\n=== Training LightGBM Regressor ===")
    model = LGBMRegressor(**params)
    
    # Save original stderr
    old_stderr = sys.stderr
    # Redirect stderr to devnull
    sys.stderr = open(os.devnull, 'w')
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X_train, y_train)
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = old_stderr

    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)

    # Calculate metrics
    test_r2 = r2_score(y_test, pred_test)
    train_r2 = r2_score(y_train, pred_train)
    test_mse = mean_squared_error(y_test, pred_test)
    test_mae = mean_absolute_error(y_test, pred_test)
    
    print(f"✓ Model training complete")
    print(f"\nRegression Performance (log scale):")
    print(f"  Test Set:  R² = {test_r2:.3f}, MSE = {test_mse:.3f}, MAE = {test_mae:.3f}")
    print(f"  Train Set: R² = {train_r2:.3f}")

    y_test_exp = np.expm1(y_test)
    pred_test_exp = np.expm1(pred_test)
    pred_test_exp[pred_test_exp < 0] = 0
    mae_days = mean_absolute_error(y_test_exp, pred_test_exp)
    print(f"  Days scale: MAE = {mae_days:.1f} days")

    pred_actual_df = pd.DataFrame({
        'real_log_time': y_test,
        'predicted_log_time': pred_test,
        'real_time_days': y_test_exp,
        'predicted_time_days': pred_test_exp
    })

    return model, pred_actual_df

# --- Hash Group Split for Data Leakage Prevention ---

def hash_group_train_test_split(X, y, sequence_dict, isolate_ids, train_size=0.8, random_state=42):
    """
    Performs train/test split based on sequence hash groups to prevent data leakage.
    
    Identical amino acid sequences are grouped together and assigned to the same fold
    to ensure that identical sequences don't appear in both training and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector  
        sequence_dict: Dictionary mapping isolate IDs to sequences
        isolate_ids: List of isolate IDs corresponding to X and y rows
        train_size: Fraction of data for training (default 0.8)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_indices, test_indices)
    """
    print(f"\n=== Hash Group Split (train_size={train_size}) ===")
    
    # Reset random state for reproducibility
    np.random.seed(random_state)
    
    # Create hash groups for sequences
    sequence_hashes = {}
    hash_to_indices = {}
    
    for i, isolate_id in enumerate(isolate_ids):
        if isolate_id in sequence_dict:
            sequence = sequence_dict[isolate_id]
            # Create SHA-1 hash of the amino acid sequence
            seq_hash = hashlib.sha1(sequence.encode('utf-8')).hexdigest()
            sequence_hashes[i] = seq_hash
            
            if seq_hash not in hash_to_indices:
                hash_to_indices[seq_hash] = []
            hash_to_indices[seq_hash].append(i)
    
    # Get unique hash groups
    unique_hashes = list(hash_to_indices.keys())
    n_unique_sequences = len(unique_hashes)
    
    print(f"  Total samples: {len(isolate_ids)}")
    print(f"  Unique sequence patterns: {n_unique_sequences}")
    print(f"  Duplication ratio: {len(isolate_ids) / n_unique_sequences:.1f}x")
    
    # Shuffle hash groups and split
    np.random.shuffle(unique_hashes)
    n_train_groups = int(train_size * n_unique_sequences)
    
    train_hashes = unique_hashes[:n_train_groups]
    test_hashes = unique_hashes[n_train_groups:]
    
    # Get indices for train and test sets
    train_indices = []
    test_indices = []
    
    for hash_val in train_hashes:
        train_indices.extend(hash_to_indices[hash_val])
    
    for hash_val in test_hashes:
        test_indices.extend(hash_to_indices[hash_val])
    
    # Sort indices to maintain order
    train_indices.sort()
    test_indices.sort()
    
    # Create splits
    X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
    X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    print(f"  Hash group split: {len(train_indices)} train, {len(test_indices)} test")
    print(f"  Train groups: {len(train_hashes)}, Test groups: {len(test_hashes)}")
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices

# --- Correlation Analysis ---

def load_fubar_data(filepath):
    """Loads FUBAR dN/dS data from CSV file.
    
    Args:
        filepath: Path to FUBAR CSV output file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['original_site', 'dN_dS'] or None if error
    """
    try:
        print(f"\nLoading FUBAR data from {filepath}...")
        # Read the CSV - now has proper headers including discard columns
        df_fubar = pd.read_csv(filepath)
        
        # Just use the columns we need
        df_fubar = df_fubar[['site', 'alpha', 'beta']].copy()
        
        # Debug: Check what we loaded
        print(f"  Raw data shape: {df_fubar.shape}")
        print(f"  Site range: {df_fubar['site'].min()} - {df_fubar['site'].max()}")
        
        # Calculate dN/dS ratio (beta/alpha)
        # Handle division by zero
        df_fubar['dN_dS'] = np.where(df_fubar['alpha'] > 0, 
                                     df_fubar['beta'] / df_fubar['alpha'],
                                     np.nan)
        df_fubar['dN_dS'] = df_fubar['dN_dS'].replace([np.inf, -np.inf], np.nan)
        
        # The FUBAR file contains amino acid positions for the full gene (1-based, 1-566)
        # This matches the numbering used by entropy and LEISR
        df_fubar['original_site'] = df_fubar['site'].astype(int)
        
        # Select only needed columns
        df_fubar = df_fubar[['original_site', 'dN_dS']].dropna()
        
        print(f"✓ FUBAR data loaded: {len(df_fubar)} sites")
        print(f"  dN/dS range: {df_fubar['dN_dS'].min():.3f} - {df_fubar['dN_dS'].max():.3f}")
        print(f"  Mean dN/dS: {df_fubar['dN_dS'].mean():.3f}")
        print(f"  Median dN/dS: {df_fubar['dN_dS'].median():.3f}")
        print(f"  Sites with dN/dS > 1: {(df_fubar['dN_dS'] > 1).sum()}")
        
        return df_fubar
        
    except Exception as e:
        print(f"⚠ Error loading FUBAR data from {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_leisr_data(filepath):
    """Loads evolutionary rate data from LEISR analysis.
    
    Args:
        filepath (str): Path to LEISR CSV file
        
    Returns:
        pd.DataFrame or None: DataFrame with 'site' and 'erate' columns, or None if failed
    """
    print(f"Loading evolutionary rate data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Warning: LEISR file not found at {filepath}. Cannot calculate E-Rate correlations.")
        return None
    try:
        df_erates = pd.read_csv(filepath)
        rename_map = {}
        if 'Site' in df_erates.columns: rename_map['Site'] = 'site'
        if 'MLE' in df_erates.columns: rename_map['MLE'] = 'erate'
        df_erates.rename(columns=rename_map, inplace=True)

        if 'site' not in df_erates.columns or 'erate' not in df_erates.columns:
            raise KeyError("LEISR file missing required 'site' or 'erate' (MLE) column.")

        df_erates['site'] = pd.to_numeric(df_erates['site'])
        print(f"✓ Evolutionary rates loaded: {len(df_erates)} sites")
        return df_erates[['site', 'erate']]
    except (KeyError, ValueError, TypeError, pd.errors.ParserError, Exception) as e:
        print(f"⚠ Error loading evolutionary rates from {filepath}: {e}")
        return None

def prepare_entropy_data(entropies, n_sequence_sites):
    """Creates DataFrame of Shannon entropy values per sequence site.
    
    Args:
        entropies (list): List of entropy values
        n_sequence_sites (int): Expected number of sequence sites
        
    Returns:
        pd.DataFrame or None: DataFrame with 'original_site' and 'entropy' columns
    """
    if entropies is None or len(entropies) != n_sequence_sites:
        print(f"⚠ Invalid entropy data (expected {n_sequence_sites} sites, got {len(entropies) if entropies else 0})")
        return None
    df_entropy = pd.DataFrame({
        'original_site': range(1, n_sequence_sites + 1),
        'entropy': entropies
    })
    print(f"✓ Entropy data prepared: {len(df_entropy)} sites")
    return df_entropy

def prepare_shap_correlation_data(shap_df, shap_value_col, site_offset):
    """Aggregates SHAP values by protein site for correlation analysis.
    
    Args:
        shap_df (pd.DataFrame): DataFrame with SHAP values and protein site info
        shap_value_col (str): Column name containing SHAP values to aggregate
        site_offset (int): Offset for converting protein to original site numbers
        
    Returns:
        pd.DataFrame or None: DataFrame with 'original_site' and 'shap_sum' columns
    """
    protein_site_col = 'protein_site'
    if not isinstance(shap_df, pd.DataFrame) or not all(c in shap_df.columns for c in [protein_site_col, shap_value_col]):
        print(f"Error: Invalid input DataFrame for SHAP aggregation. Missing '{protein_site_col}' or '{shap_value_col}'.")
        return None
    if shap_df.empty:
        print("Warning: Input SHAP DataFrame is empty.")
        return None

    shap_df_agg = shap_df.copy()
    shap_df_agg[protein_site_col] = pd.to_numeric(shap_df_agg[protein_site_col], errors='coerce')
    shap_df_agg.dropna(subset=[protein_site_col], inplace=True)
    shap_df_agg = shap_df_agg[shap_df_agg[protein_site_col] >= 1]
    shap_df_agg[protein_site_col] = shap_df_agg[protein_site_col].astype(int)

    if shap_df_agg.empty:
        print("⚠ No valid protein sequence sites found in SHAP data")
        return None

    site_agg = shap_df_agg.groupby(protein_site_col)[shap_value_col].sum()

    corr_base_df = pd.DataFrame({'shap_sum': site_agg})
    corr_base_df.index.name = 'protein_site'
    corr_base_df.reset_index(inplace=True)

    corr_base_df['original_site'] = corr_base_df['protein_site'] + site_offset

    return corr_base_df[['original_site', 'shap_sum']]

def perform_correlation_analysis(shap_agg_df, df_entropy, df_erates, df_fubar, model_name, output_dir, figure_start_index):
    """Performs correlation analysis between SHAP values and evolutionary metrics.
    
    Args:
        shap_agg_df (pd.DataFrame): Aggregated SHAP values by site
        df_entropy (pd.DataFrame): Shannon entropy data
        df_erates (pd.DataFrame): Evolutionary rate data (LEISR)
        df_fubar (pd.DataFrame): FUBAR dN/dS data
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save plots
        figure_start_index (int): Starting figure number
        
    Returns:
        tuple: (int: Updated figure counter, dict: correlation results)
    """
    print(f"\n=== Correlation Analysis: {model_name} Model ===")
    if shap_agg_df is None or shap_agg_df.empty:
        print(f"⚠ Skipping {model_name} correlations - no SHAP data available")
        return figure_start_index, {}

    correlation_results = {}
    xlabel_text = "sum of SHAP values"

    # 1. SHAP vs Entropy
    if df_entropy is not None:
        corr_df = pd.merge(shap_agg_df, df_entropy, on='original_site', how='inner')
        if not corr_df.empty:
            print(f"  Analyzing SHAP vs Entropy ({len(corr_df)} sites)...")
            corr, p_val = plot_correlation_scatter(
                corr_df, x_col='shap_sum', y_col='entropy',
                xlabel=xlabel_text, ylabel='Shannon entropy',
                filename=os.path.join(output_dir, f'figure_{figure_start_index}_{model_name.lower()}_shap_vs_entropy.pdf'),
                use_ranks=True
            )
            if corr is not None:
                correlation_results['entropy'] = (corr, p_val)
            figure_start_index += 1

    # 2. SHAP vs Evolutionary Rate (LEISR)
    if df_erates is not None:
        corr_df = pd.merge(shap_agg_df, df_erates, left_on='original_site', right_on='site', how='inner')
        if not corr_df.empty:
            print(f"  Analyzing SHAP vs Evolutionary Rate (LEISR) ({len(corr_df)} sites)...")
            corr, p_val = plot_correlation_scatter(
                corr_df, x_col='shap_sum', y_col='erate',
                xlabel=xlabel_text, ylabel='evolutionary rate (LEISR)',
                filename=os.path.join(output_dir, f'figure_{figure_start_index}_{model_name.lower()}_shap_vs_erate.pdf'),
                use_ranks=True
            )
            if corr is not None:
                correlation_results['leisr'] = (corr, p_val)
            figure_start_index += 1

    # 3. SHAP vs FUBAR dN/dS
    if df_fubar is not None:
        # Debug merging
        print(f"  Debug: SHAP sites range: {shap_agg_df['original_site'].min()} - {shap_agg_df['original_site'].max()}")
        print(f"  Debug: FUBAR sites range: {df_fubar['original_site'].min()} - {df_fubar['original_site'].max()}")
        print(f"  Debug: SHAP unique sites: {shap_agg_df['original_site'].nunique()}")
        print(f"  Debug: FUBAR unique sites: {df_fubar['original_site'].nunique()}")
        
        corr_df = pd.merge(shap_agg_df, df_fubar, on='original_site', how='inner')
        
        if not corr_df.empty:
            print(f"  Analyzing SHAP vs dN/dS (FUBAR) ({len(corr_df)} sites)...")
            print(f"  Debug: Merged data - SHAP range: {corr_df['shap_sum'].min():.3f} - {corr_df['shap_sum'].max():.3f}")
            print(f"  Debug: Merged data - dN/dS range: {corr_df['dN_dS'].min():.3f} - {corr_df['dN_dS'].max():.3f}")
            
            # Check for overlap in site numbers
            overlap_sites = set(shap_agg_df['original_site']).intersection(set(df_fubar['original_site']))
            print(f"  Debug: Overlapping sites: {len(overlap_sites)}")
            if len(overlap_sites) < 10:
                print(f"    Overlap sites: {sorted(list(overlap_sites))[:10]}")
            
            corr, p_val = plot_correlation_scatter(
                corr_df, x_col='shap_sum', y_col='dN_dS',
                xlabel=xlabel_text, ylabel='dN/dS (FUBAR)',
                filename=os.path.join(output_dir, f'figure_{figure_start_index}_{model_name.lower()}_shap_vs_dnds.pdf'),
                use_ranks=True
            )
            if corr is not None:
                correlation_results['fubar'] = (corr, p_val)
            figure_start_index += 1
        else:
            print(f"  WARNING: No data after merging SHAP and FUBAR!")

    return figure_start_index, correlation_results

def create_correlation_heatmap(passage_corr_results, date_corr_results, df_entropy, df_erates, df_fubar,
                               passage_shap_data, date_shap_data, output_dir, figure_start_index):
    """Creates a Spearman correlation heatmap of all metrics.
    
    Args:
        passage_corr_results: Dict of correlation results from passage model
        date_corr_results: Dict of correlation results from date model
        df_entropy: Entropy data
        df_erates: LEISR evolutionary rate data
        df_fubar: FUBAR dN/dS data
        passage_shap_data: Aggregated SHAP values from passage model
        date_shap_data: Aggregated SHAP values from date model
        output_dir: Directory to save the heatmap
        figure_start_index: Current figure counter
        
    Returns:
        int: Updated figure counter
    """
    print("\n=== Creating Spearman Correlation Heatmap ===")
    
    # Prepare a combined dataframe with all metrics
    combined_df = pd.DataFrame()
    
    # Add entropy data
    if df_entropy is not None:
        combined_df = df_entropy.copy()
        combined_df = combined_df.rename(columns={'entropy': 'Entropy'})
    
    # Add LEISR data
    if df_erates is not None:
        if combined_df.empty:
            combined_df = df_erates[['site', 'erate']].copy()
            combined_df = combined_df.rename(columns={'site': 'original_site', 'erate': 'LEISR'})
        else:
            temp_df = df_erates[['site', 'erate']].copy()
            combined_df = pd.merge(combined_df, temp_df, left_on='original_site', right_on='site', how='outer')
            combined_df = combined_df.rename(columns={'erate': 'LEISR'})
            combined_df.drop(columns=['site'], inplace=True, errors='ignore')
    
    # Add FUBAR data
    if df_fubar is not None:
        print(f"  Adding FUBAR data: {len(df_fubar)} sites, {df_fubar['dN_dS'].notna().sum()} non-null values")
        if combined_df.empty:
            combined_df = df_fubar[['original_site', 'dN_dS']].copy()
            combined_df = combined_df.rename(columns={'dN_dS': 'dN/dS'})
        else:
            temp_df = df_fubar[['original_site', 'dN_dS']].copy()
            combined_df = pd.merge(combined_df, temp_df, on='original_site', how='outer')
            combined_df = combined_df.rename(columns={'dN_dS': 'dN/dS'})
        print(f"  Combined dataframe now has {len(combined_df)} sites")
    
    # Add SHAP values from passage model
    if passage_shap_data is not None:
        temp_df = passage_shap_data[['original_site', 'shap_sum']].copy()
        combined_df = pd.merge(combined_df, temp_df, on='original_site', how='outer')
        combined_df = combined_df.rename(columns={'shap_sum': 'SHAP Passage'})
    
    # Add SHAP values from date model
    if date_shap_data is not None:
        temp_df = date_shap_data[['original_site', 'shap_sum']].copy()
        combined_df = pd.merge(combined_df, temp_df, on='original_site', how='outer')
        combined_df = combined_df.rename(columns={'shap_sum': 'SHAP Date'})
    
    # Select only the metric columns (exclude original_site)
    metric_cols = [col for col in combined_df.columns if col != 'original_site']
    
    print(f"  Metric columns for heatmap: {metric_cols}")
    print(f"  Combined dataframe shape: {combined_df.shape}")
    
    if len(metric_cols) < 2:
        print("⚠ Insufficient metrics for correlation heatmap")
        return figure_start_index
    
    # Calculate Spearman correlation matrix
    from scipy.stats import spearmanr
    corr_matrix = pd.DataFrame(index=metric_cols, columns=metric_cols, dtype=float)
    pval_matrix = pd.DataFrame(index=metric_cols, columns=metric_cols, dtype=float)
    
    for i, col1 in enumerate(metric_cols):
        for j, col2 in enumerate(metric_cols):
            if i == j:
                corr_matrix.loc[col1, col2] = 1.0
                pval_matrix.loc[col1, col2] = 0.0
            else:
                # Get non-null pairs
                valid_pairs = combined_df[[col1, col2]].dropna()
                if len(valid_pairs) > 2:
                    corr, pval = spearmanr(valid_pairs[col1], valid_pairs[col2])
                    corr_matrix.loc[col1, col2] = corr
                    pval_matrix.loc[col1, col2] = pval
                else:
                    corr_matrix.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
    
    # Convert to numeric type
    corr_matrix = corr_matrix.astype(float)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Use Blues colormap to match the confusion matrix style
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Create heatmap with annotations - only show positive correlations
    # Clip values to be between 0 and 1
    corr_matrix_display = corr_matrix.clip(lower=0)
    
    # Increase annotation font size to match confusion matrix (size 12)
    sns.heatmap(corr_matrix_display, annot=True, fmt='.2f', cmap=cmap,
                annot_kws={"size": 12},
                square=True, linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Spearman ρ"},
                vmin=0, vmax=1, mask=mask)
    
    # Remove title
    # plt.title('Spearman Correlation Matrix of Per-Site Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.xticks(rotation=90, ha='center', fontsize=11)  # Vertical x-labels with larger font
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'figure_{figure_start_index}_spearman_correlation_heatmap.pdf')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation heatmap: {filename}")
    
    # Print correlation summary
    print("\nSpearman Correlation Summary:")
    for i, col1 in enumerate(metric_cols):
        for j, col2 in enumerate(metric_cols):
            if i < j and not pd.isna(corr_matrix.loc[col1, col2]):
                corr_val = corr_matrix.loc[col1, col2]
                pval = pval_matrix.loc[col1, col2]
                print(f"  {col1} vs {col2}: ρ = {corr_val:.3f} (p = {pval:.3g})")
    
    return figure_start_index + 1

# --- Main Execution ---
def main():
    """Executes the complete influenza sequence analysis pipeline.
    
    This function orchestrates the entire analysis including:
    - Data loading and preprocessing
    - Passage history prediction modeling
    - Date prediction modeling
    - SHAP explainability analysis
    - Correlation analysis with evolutionary metrics
    - Visualization generation
    """
    # Comprehensive warning suppression
    warnings.simplefilter('ignore') # Suppress all warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Specifically suppress LightGBM warnings about splits
    warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
    warnings.filterwarnings('ignore', message='.*min_data_in_leaf.*')
    warnings.filterwarnings('ignore', message='.*min_child_samples.*')
    
    # Suppress all logging except critical errors
    # Disable flupan's file handler and all output
    logging.getLogger().handlers = []
    logging.getLogger().setLevel(logging.CRITICAL)
    # Also specifically suppress various package loggers
    logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
    logging.getLogger('LightGBM').setLevel(logging.CRITICAL)
    logging.getLogger('flupan').setLevel(logging.CRITICAL)
    logging.getLogger('root').setLevel(logging.CRITICAL)
    # Prevent any new handlers from being added
    logging.getLogger().propagate = False
    
    # Set publication-quality plotting style
    set_publication_style()
    
    create_output_dir(OUTPUT_DIR)
    figure_counter = 1 # Initialize figure counter

    # --- 1. Data Loading ---
    metadata_raw, d_seqs = import_data(METADATA_FILE, SEQUENCE_FILE)

    # --- 2. Passage History Parsing (with Caching) ---
    if PARSE_PASSAGES_FRESH or not os.path.exists(PASSAGES_CACHE):
        print("Parsing passage history (this may take a while)...")
        passages_df = parse_history(metadata_raw)
        write_cache(passages_df, PASSAGES_CACHE)
        print("✓ Passage history parsed and cached")
    else:
        print(f"Loading passage history from cache: {PASSAGES_CACHE}")
        passages_df = read_cache(PASSAGES_CACHE)
        if passages_df is None or not isinstance(passages_df, pd.DataFrame):
             print("Error: Could not load valid passages from cache. Rerun with PARSE_PASSAGES_FRESH=True.")
             return # Exit if essential data is missing
        print(f"✓ Passage history loaded from cache (skipping flupan parsing)")
        print(f"  Cache contains {len(passages_df)} records")
        if isinstance(passages_df, pd.DataFrame) and not passages_df.empty:
            print(f"  Sample passage values: {passages_df.iloc[0] if len(passages_df) > 0 else 'N/A'}")

    # --- 3. Consolidate Metadata ---
    md_consolidated, first_collection_date = consolidate_data(metadata_raw, passages_df)

    # --- 4. Filter and Clean Data ---
    md_final = get_clean_multipass(md_consolidated, d_seqs, N_SEQUENCE_SITES)
    print(f"\n✓ Final dataset ready: {md_final.shape[0]:,} sequences for modeling")
    if md_final.empty:
        print("✗ Error: No sequences remain after filtering - check data quality")
        return

    # --- 4a. Plot Collection Date Histogram ---
    plot_collection_date_histogram(md_final, os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_collection_date_histogram.pdf'))
    figure_counter += 1

    # --- 5. Geocoding (with Caching) ---
    if GEOCODE_FRESH or not all(os.path.exists(f) for f in [LOCATIONS_CACHE, COUNTRIES_CACHE]):
        latlong_map, country_lookup = pull_locs_data_online(md_final, LOCATIONS_CACHE, COUNTRIES_CACHE, GOOGLE_API_KEY)
    else:
        latlong_map, country_lookup = import_locs_data(LOCATIONS_CACHE, COUNTRIES_CACHE)

    if latlong_map is None or country_lookup is None:
         print("Warning: Geocoding failed or cache missing. Proceeding without Lat/Lon for affected entries.")
         if 'latitude' not in md_final.columns: md_final['latitude'] = np.nan
         if 'longitude' not in md_final.columns: md_final['longitude'] = np.nan
    else:
        md_final = append_latlong(md_final, latlong_map, country_lookup)

    # --- 6. Sequence Variability Analysis & Plotting ---
    print("\nAnalyzing sequence variability...")
    temp_sequences = np.array([d_seqs[iso_id] for iso_id in md_final['Isolate_Id']])
    temp_sequence_df = pd.DataFrame([list(seq) for seq in temp_sequences],
                                    columns=[f's{i+1}' for i in range(N_SEQUENCE_SITES)])
    entropies = [Shannon_entropy(temp_sequence_df[col]) for col in temp_sequence_df.columns]
    distinct_counts = [temp_sequence_df[col].nunique() for col in temp_sequence_df.columns]
    plot_combined_variability(entropies, distinct_counts, os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_sequence_variability.pdf'))
    figure_counter += 1
    del temp_sequences, temp_sequence_df # Clean up temporary data

    # --- 7. Prepare Features for Passage Prediction ---
    X_passage, passage_ohe = prepare_features(
        md_final, d_seqs, N_SEQUENCE_SITES, include_time=True, fit_encoder=True
    )
    y_passage = md_final['pass1'].astype(str)

    # --- 8. Train/Test Split for Passage Prediction ---
    # Check for classes with fewer than 2 samples before stratified split
    min_samples_required = 2 # For train_test_split stratification
    value_counts = y_passage.value_counts()
    rare_classes = value_counts[value_counts < min_samples_required].index.tolist()

    if rare_classes:
        print(f"\nWarning: The following classes have fewer than {min_samples_required} samples and will be removed before splitting:")
        for cls, count in value_counts[value_counts < min_samples_required].items():
            print(f"  - Class '{cls}': {count} sample(s)")

        # Filter out rare classes from features (X_passage) and target (y_passage)
        keep_indices = y_passage[~y_passage.isin(rare_classes)].index
        print(f"Removing {len(y_passage) - len(keep_indices)} samples belonging to rare classes.")
        X_passage = X_passage.loc[keep_indices]
        y_passage = y_passage.loc[keep_indices]

        if X_passage.empty or y_passage.empty:
            print("ERROR: No data remaining after removing rare classes. Cannot proceed with passage model.")
            return # Exit if no data left

    # Proceed with the split using the potentially filtered data
    print(f"\nPreparing train/test split:")
    print(f"  Total samples: {len(X_passage)}")
    print(f"  Unique classes in y_passage: {y_passage.nunique()}")
    print(f"  Class distribution: {y_passage.value_counts().to_dict()}")
    
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_passage, y_passage, train_size=0.8, random_state=RANDOM_STATE, stratify=y_passage
    )
    
    print(f"  Training set size: {len(X_train_p)}")
    print(f"  Test set size: {len(X_test_p)}")
    print(f"  Training classes: {y_train_p.nunique()}")
    print(f"\nPassage model dataset: {X_train_p.shape[0]:,} train, {X_test_p.shape[0]:,} test")
    print("\nTrain set passage distribution:")
    train_dist = y_train_p.value_counts(normalize=True).round(3)
    for passage_type, freq in train_dist.items():
        print(f"  {passage_type}: {freq:.1%}")
    print("\nTest set passage distribution:")
    test_dist = y_test_p.value_counts(normalize=True).round(3)
    for passage_type, freq in test_dist.items():
        print(f"  {passage_type}: {freq:.1%}")

    # --- 9. Passage History Prediction Model (LGBMClassifier) ---
    best_params_passage = None
    if RUN_BAYES_OPT_PASSAGE:
        best_params_passage = run_bayes_opt(X_train_p, y_train_p, BAYES_OPT_LOG_PASSAGE, BAYES_OPT_CACHE_PASSAGE)
    elif LOAD_BAYES_OPT_PASSAGE:
        best_params_passage = load_best_bayes_params(BAYES_OPT_CACHE_PASSAGE)

    if best_params_passage is None:
        print("Using pre-optimized parameters for passage classification model")
        best_params_passage = { # Parameters from original script's optimized run
             'reg_lambda': 522.0, 'reg_alpha': 0.0, 'learning_rate': 0.1,
             'objective': 'multiclass', 'num_leaves': 503, 'n_estimators': 1318,
             'metric': 'multi_logloss', 'min_child_samples': 5, 'max_depth': 8,
             'colsample_bytree': 1.0, 'boosting_type': 'gbdt', 'class_weight': 'balanced',
             'n_jobs': -1, 'random_state': RANDOM_STATE, 'verbosity': -1, 'force_row_wise': True
         }
    # Standardize final chosen parameters
    passage_model_params = standardize_lgbm_params(best_params_passage)

    # Train and evaluate
    lgbm_passage, passage_classes = train_evaluate_classifier(
        X_train_p, y_train_p, X_test_p, y_test_p, passage_model_params
    )

    # Plot Confusion Matrices
    plot_confusion_matrix(y_test_p, lgbm_passage.predict(X_test_p), passage_classes, normalize=False,
                          filename=os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_passage_confusion_matrix_raw.pdf'),
                          title_suffix="Passage Raw")
    figure_counter += 1
    plot_confusion_matrix(y_test_p, lgbm_passage.predict(X_test_p), passage_classes, normalize=True,
                           filename=os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_passage_confusion_matrix_normalized.pdf'),
                           title_suffix="Passage Normalized")
    figure_counter += 1

    # --- 10. SHAP Analysis for Passage Model ---
    print("\nCalculating SHAP values for passage prediction model...")
    explainer_passage = shap.TreeExplainer(lgbm_passage)
    shap_values_passage = explainer_passage.shap_values(X_test_p)

    # Process SHAP values - uses SITE_OFFSET to calculate protein site numbers
    shap_df_passage, site_agg_shap_passage_total, site_agg_shap_passage_per_class = process_shap_values_classifier(
        shap_values_passage, X_test_p, passage_ohe.categories_, passage_classes, SITE_OFFSET)

    # Filter out 'N/A' labeled features before plotting stacked importance
    print("Filtering 'N/A' labels from passage SHAP results for plotting stacked importance...")
    valid_labels_p5b = site_agg_shap_passage_total.index != 'N/A'
    filtered_site_agg_shap_passage_total = site_agg_shap_passage_total[valid_labels_p5b]
    # Ensure the per-class data aligns with the filtered total data
    filtered_site_agg_shap_passage_per_class = site_agg_shap_passage_per_class.loc[filtered_site_agg_shap_passage_total.index]

    # Plot stacked feature importance (uses filtered data)
    plot_diverging_stacked_importance(
        site_agg_shap_per_class=filtered_site_agg_shap_passage_per_class,
        site_agg_shap_total=filtered_site_agg_shap_passage_total,
        class_names=passage_classes,
        filename=os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_passage_feature_importance_stacked.pdf'),
        top_n=20
    )
    figure_counter += 1

    # --- 10a. Contingency Table Validation for Top SHAP Feature ---
    # This new section validates the top feature from the SHAP plot by checking for
    # statistical association between amino acids at that site and passage history.
    print("\n--- Validating top SHAP feature with contingency table analysis ---")

    # Find the top predictive feature that is a sequence site
    top_site_label = None
    for label in filtered_site_agg_shap_passage_total.index:
        # Check if the label contains digits (is a site) and is not purely alphabetic (like 'times')
        if any(char.isdigit() for char in label) and not label.isalpha():
            top_site_label = label
            break

    if top_site_label:
        try:
            # Extract the numeric part of the label to get the protein site number
            top_protein_site = int("".join(filter(str.isdigit, top_site_label)))
            generate_contingency_table(
                protein_site_num=top_protein_site,
                metadata_df=md_final, # Use the final, filtered metadata
                sequence_dict=d_seqs,
                site_offset=SITE_OFFSET
            )
        except (ValueError, TypeError):
            print(f"Could not parse a valid protein site number from the top SHAP feature label: '{top_site_label}'")
    else:
        print("Could not find a sequence-based feature in the top SHAP results to analyze.")

    # --- 10b. Save Passage Model SHAP Values to CSV ---
    print("\n--- Saving Passage Model SHAP Values to CSV ---")
    # Load evolutionary data for CSV export
    df_erates = load_leisr_data(LEISR_UNPASSAGED_FILE)
    df_fubar = load_fubar_data(FUBAR_FILE)
    save_shap_values_classifier_to_csv(shap_df_passage, entropies, df_erates, "results", "Passage", SITE_OFFSET)

    # --- 11. Date Prediction Model (LGBMRegressor on Unpassaged Data) ---
    print("\n--- Preparing Data for Date Prediction (Unpassaged Sequences) ---")
    md_unpassaged = md_final[md_final['pass1'] == 'UNPASSAGED'].copy()
    print(f"Number of UNPASSAGED sequences: {md_unpassaged.shape[0]}")

    if md_unpassaged.shape[0] < 50:
        print("Warning: Too few unpassaged sequences (<50). Skipping date prediction model and related analyses.")
    else:

        X_date, _ = prepare_features(
            md_unpassaged, d_seqs, N_SEQUENCE_SITES, include_time=False,
            fit_encoder=False, encoder=passage_ohe
        )
        y_date = np.log1p(md_unpassaged['times'].astype(float))

        # Train/Test Split
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_date, y_date, train_size=0.8, random_state=RANDOM_STATE
        )
        print(f"Date Model Split: Train={X_train_d.shape[0]}, Test={X_test_d.shape[0]}")

        # ==============================================================
        # QUICK COMPARISON: sequence‑only vs geo‑only for date prediction
        # ==============================================================

        geo_cols = ["latitude", "longitude"]
        seq_cols = [c for c in X_train_d.columns if c not in geo_cols]

        model_seq_only = lgb.LGBMRegressor(**{   # identical params later used
            'reg_lambda': 400.0, 'reg_alpha': 0.0, 'learning_rate': 0.1,
            'num_leaves': 215, 'n_estimators': 1200, 'min_child_samples': 15,
            'max_depth': 8, 'colsample_bytree': 1.0, 'boosting_type': 'gbdt',
            'objective': 'regression_l2', 'metric': 'rmse',
            'n_jobs': -1, 'random_state': RANDOM_STATE
        })
        model_geo_only = lgb.LGBMRegressor(**model_seq_only.get_params())

        # Suppress warnings during fit - redirect stderr completely
        import sys
        
        # Save original stderr
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model_seq_only.fit(X_train_d[seq_cols], y_train_d)
                model_geo_only.fit(X_train_d[geo_cols], y_train_d)
        finally:
            # Restore stderr
            sys.stderr.close()
            sys.stderr = old_stderr

        r2_seq = r2_score(y_test_d, model_seq_only.predict(X_test_d[seq_cols]))
        r2_geo = r2_score(y_test_d, model_geo_only.predict(X_test_d[geo_cols]))

        print("\n--- Feature subset performance (Tip‑date model) ---")
        print(f"  Sequence‑only R² : {r2_seq:.3f}")
        print(f"  Geo‑only      R² : {r2_geo:.3f}")
        print("\n\n")

        # Define, Train, Evaluate Regressor
        print("Using hardcoded parameters for LGBMRegressor (Date Model).")
        params_date_hardcoded = {
            'reg_lambda': 400.0, 'reg_alpha': 0.0, 'learning_rate': 0.1,
            'num_leaves': 215, 'n_estimators': 1200, 'min_child_samples': 15,
            'max_depth': 8, 'colsample_bytree': 1.0, 'boosting_type': 'gbdt',
            'objective': 'regression_l2', 'metric': 'rmse',
            'n_jobs': -1, 'random_state': RANDOM_STATE
        }
        date_model_params = standardize_lgbm_params(params_date_hardcoded)

        lgbm_date, pred_actual_df_date = train_evaluate_regressor(
            X_train_d, y_train_d, X_test_d, y_test_d, date_model_params
        )

        # --- 11b. Hash Group Split Date Model (for Data Leakage Assessment) ---
        print("\n--- Hash Group Split Date Model (No Data Leakage) ---")
        print("Training model with identical sequences kept in same fold...")
        
        # Get isolate IDs in the same order as prepare_features used
        md_unpassaged_reset = md_unpassaged.reset_index(drop=True)
        isolate_ids = md_unpassaged_reset['Isolate_Id'].values
        
        # Perform hash group split
        X_train_d_hash, X_test_d_hash, y_train_d_hash, y_test_d_hash, train_idx_hash, test_idx_hash = hash_group_train_test_split(
            X_date, y_date, d_seqs, isolate_ids, train_size=0.8, random_state=RANDOM_STATE
        )
        
        # Train model with hash group split
        lgbm_date_hash, pred_actual_df_date_hash = train_evaluate_regressor(
            X_train_d_hash, y_train_d_hash, X_test_d_hash, y_test_d_hash, date_model_params
        )
        
        # The train_evaluate_regressor function already returns predictions in days scale
        # No need to convert - use the 'real_time_days' and 'predicted_time_days' columns directly
        r2_hash_original = r2_score(pred_actual_df_date_hash['real_time_days'], 
                                   pred_actual_df_date_hash['predicted_time_days'])
        mae_hash_original = mean_absolute_error(pred_actual_df_date_hash['real_time_days'], 
                                               pred_actual_df_date_hash['predicted_time_days'])
        
        print(f"\n=== Hash Group Split Results (Original Scale) ===")
        print(f"  R² = {r2_hash_original:.3f}")
        print(f"  MAE = {mae_hash_original:.1f} days")
        print(f"  For manuscript: 'tip-date R² = {r2_hash_original:.2f}, MAE = {mae_hash_original:.0f} d'")
        
        # md_unpassaged_reset already created above for hash split

        # ------------------------------------------------------------------
        # Root‑to‑tip vs LightGBM‑predicted collection date  (protein tree)
        # ------------------------------------------------------------------
        print(f"\n--- Comparing Root-to-Tip Distance with Predicted Dates ---")
        if not os.path.exists(TREE_FILE):
            print(f"Warning: Tree file not found at {TREE_FILE}. Skipping root-to-tip analysis.")
        else:
            try:
                tree = Phylo.read(TREE_FILE, "newick")
                if not tree.rooted:
                    tree.root_at_midpoint()
                root_to_tip = {t.name: tree.distance(tree.root, t) for t in tree.get_terminals()}

                pred_actual_df_date["Accession"] = md_unpassaged_reset.iloc[
                    X_test_d.index
                ]["Isolate_Id"].values

                def clean(s): return str(s).split()[0]
                root_to_tip_clean = {clean(k): v for k, v in root_to_tip.items()}

                clock_df = pred_actual_df_date.copy()
                clock_df["predicted_time_years"] = clock_df["predicted_time_days"] / 365.25
                clock_df["root_to_tip"] = clock_df["Accession"].apply(
                    lambda x: root_to_tip_clean.get(clean(x))
                )
                clock_df = clock_df.dropna(subset=["root_to_tip"])

                print(f"Matched tips for root-to-tip analysis: {len(clock_df)} / {len(pred_actual_df_date)}")
                if len(clock_df) >= 2:
                    r, p = pearsonr(clock_df["predicted_time_years"], clock_df["root_to_tip"])
                    print(f"Root-to-tip vs predicted date: r = {r:.3f}, p = {p:.2e}")

                    # Create figure with improved styling
                    fig, ax = plt.subplots(figsize=(7, 6))
                    
                    # Plot scatter with improved styling
                    sns.scatterplot(data=clock_df, x="predicted_time_years", y="root_to_tip",
                                    s=40, alpha=0.7, color='#4477AA', 
                                    edgecolor='#2B5F8E', linewidth=0.5, ax=ax)
                    
                    # Add regression line
                    sns.regplot(data=clock_df, x="predicted_time_years", y="root_to_tip",
                                scatter=False, ci=None, line_kws={'color': '#EE6677', 'lw': 1.5},
                                ax=ax)
                    
                    # Labels and styling
                    ax.set_xlabel("LightGBM-predicted collection date (years)", fontsize=12)
                    ax.set_ylabel("root-to-tip distance (AA subs/site)", fontsize=12)
                    
                    # Grid and background
                    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                    ax.set_facecolor('white')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR,
                                f"figure_{figure_counter}_root_to_tip_vs_predicted.pdf"))
                    figure_counter += 1
                    plt.close()
                else:
                    print("Warning: Fewer than two matched tips after cleaning/merging. Check accession strings in tree and metadata.")
            except Exception as e:
                print(f"Error during root-to-tip analysis: {e}")


        # Plotting for Date Model
        plot_predicted_vs_actual(pred_actual_df_date,
                                 os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_date_prediction_vs_actual.pdf'),
                                 first_collection_date)
        figure_counter += 1
        plot_residuals(pred_actual_df_date, os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_date'))
        figure_counter += 2

        # --- 12. SHAP Analysis for Date Model ---
        print("\nCalculating SHAP values for date prediction model...")
        explainer_date = shap.TreeExplainer(lgbm_date)
        shap_values_date = explainer_date.shap_values(X_test_d) # Single array for regression

        # Process SHAP values - this now returns three items
        shap_df_date, site_agg_shap_date, site_agg_shap_magnitude = process_shap_values_regressor(
             shap_values_date, X_test_d, passage_ohe.categories_, SITE_OFFSET
        )

        # Plot SHAP Feature Importance (AA level)
        if shap_df_date is not None and not shap_df_date.empty:
            # Create series for AA-level plot (figure_10)
            plot_series_p10 = pd.Series(
                shap_df_date['mean_shap'].values, index=shap_df_date['feature_label']
            )
            plot_series_p10_filtered = plot_series_p10[plot_series_p10.index != 'N/A']

            # This plot (figure_10) remains a simple diverging bar chart
            plot_feature_importance_bar(
                plot_series_p10_filtered,
                filename=os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_date_feature_importance_aa.pdf'),
                top_n=20,
                xlabel="SHAP value (impact on date prediction)"
            )
            figure_counter += 1
        else:
            print("Skipping AA importance plot: SHAP DataFrame for date model not available.")

        # Plot SHAP Feature Importance (Site level, stacked)
        if site_agg_shap_magnitude is not None and not site_agg_shap_magnitude.empty:
            # This plot (figure_11) is the new stacked diverging bar chart
            plot_diverging_stacked_importance_regressor(
                shap_df=shap_df_date,
                ranking_series=site_agg_shap_magnitude,
                filename=os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_date_feature_importance_site.pdf'),
                top_n=20
            )
            figure_counter += 1
        else:
            print("Skipping site importance plot: Aggregated site SHAP data for date model not available.")

        # --- 12a. Save Date Model SHAP Values to CSV ---
        print("\n--- Saving Date Model SHAP Values to CSV ---")
        df_erates = load_leisr_data(LEISR_UNPASSAGED_FILE)
        df_fubar = load_fubar_data(FUBAR_FILE)
        save_shap_values_regressor_to_csv(shap_df_date, entropies, df_erates, "results", "Date", SITE_OFFSET)

        # --- 13. Correlation Analysis (SHAP vs Entropy/E-Rates) ---
        print("\n--- Preparing Data for Correlation Analysis ---")
        # df_erates already loaded above for CSV export
        df_entropy = prepare_entropy_data(entropies, N_SEQUENCE_SITES) # Uses original 1-566 site numbers

        # Prepare aggregated SHAP data for correlation
        # These functions return 'original_site' (1-566) for merging
        # Uses the full (unfiltered by 'N/A') SHAP dataframes as input
        passage_shap_corr_data = prepare_shap_correlation_data(shap_df_passage, 'total_shap_magnitude', SITE_OFFSET)
        date_shap_corr_data = prepare_shap_correlation_data(shap_df_date, 'mean_abs_shap', SITE_OFFSET)

        # Load FUBAR dN/dS data
        df_fubar = load_fubar_data(FUBAR_FILE)
        
        # Perform correlations for Passage Model - Merges happen on 'original_site'
        figure_counter, passage_corr_results = perform_correlation_analysis(passage_shap_corr_data, df_entropy, df_erates, df_fubar, "Passage", OUTPUT_DIR, figure_counter)

        # Perform correlations for Date Model - Merges happen on 'original_site'
        figure_counter, date_corr_results = perform_correlation_analysis(date_shap_corr_data, df_entropy, df_erates, df_fubar, "Date", OUTPUT_DIR, figure_counter)
        
        # Create correlation heatmap
        figure_counter = create_correlation_heatmap(passage_corr_results, date_corr_results, df_entropy, df_erates, df_fubar, 
                                                   passage_shap_corr_data, date_shap_corr_data, OUTPUT_DIR, figure_counter)

        # --- 14. Frequency Plots for Top Date Model Sites ---
        print("\n--- Generating  Plots for Selected Sites ---")

        # Use site_agg_shap_date which contains aggregated importance per protein site
        if 'site_agg_shap_date' in locals() and site_agg_shap_date is not None and not site_agg_shap_date.empty:
            # Get top N protein site numbers from date model SHAP results
            top_protein_sites = [3, 121, 131, 142, 144, 193, 198, 225, 261, 311] # MANUALLY SPECIFIED SITES
            print(f"Generating frequency plots for manually specified protein sites: {top_protein_sites}")

            if top_protein_sites:
                # Prepare DataFrame with unpassaged sequences and dates
                adf_u_plot = md_unpassaged[['Collection_Date', 'Isolate_Id']].copy()
                seq_list_u_plot = [d_seqs.get(iso_id) for iso_id in adf_u_plot['Isolate_Id']]
                valid_indices = [i for i, seq in enumerate(seq_list_u_plot) if seq is not None and len(seq) == N_SEQUENCE_SITES]
                if len(valid_indices) < len(adf_u_plot):
                    print(f"Warning: Filtering {len(adf_u_plot) - len(valid_indices)} entries with missing/invalid sequences for frequency plots.")
                    adf_u_plot = adf_u_plot.iloc[valid_indices].reset_index(drop=True)
                    seq_list_u_plot = [seq_list_u_plot[i] for i in valid_indices]

                seq_cols_df = pd.DataFrame(
                    [list(seq) for seq in seq_list_u_plot],
                    columns=[f's{i+1}' for i in range(N_SEQUENCE_SITES)] # Columns s1 to s566
                )
                adf_u_plot = pd.concat([adf_u_plot.reset_index(drop=True), seq_cols_df.reset_index(drop=True)], axis=1) # Ensure indices align
                adf_u_plot['Collection_Date'] = pd.to_datetime(adf_u_plot['Collection_Date'])

                # Define time bins
                min_dt = adf_u_plot['Collection_Date'].min()
                max_dt = adf_u_plot['Collection_Date'].max()
                if pd.notna(min_dt) and pd.notna(max_dt) and max_dt > min_dt:
                    # Ensure the range includes the max date fully for binning
                    date_bins = pd.date_range(start=min_dt, end=max_dt + pd.Timedelta(days=1), freq='60D')
                    if len(date_bins) > 1:
                        # Use the start date of the bins as labels for clarity on the x-axis
                        bin_labels = date_bins[:-1]
                        adf_u_plot['time_bin'] = pd.cut(adf_u_plot['Collection_Date'], bins=date_bins, right=False, labels=bin_labels)

                        # Loop through protein site numbers
                        for protein_site_num in top_protein_sites:
                            # Calculate the ORIGINAL FASTA position to access the correct data column
                            original_fasta_position = protein_site_num + SITE_OFFSET
                            site_col_name = f's{original_fasta_position}' # e.g., s19 for protein site 3

                            if site_col_name in adf_u_plot.columns and 'time_bin' in adf_u_plot.columns:

                                # Use protein site number for title and filename
                                plot_title = f'Amino Acid Frequency Over Time - Site {protein_site_num}'
                                plot_filename = os.path.join(OUTPUT_DIR, f'figure_{figure_counter}_frequency_site_{protein_site_num}.pdf')

                                # Call the modified plot_frequency_over_time function
                                # It now internally uses calculate_frequency_user_style
                                plot_frequency_over_time(
                                    data_df=adf_u_plot,
                                    site_col=site_col_name, # Use column name with FASTA position
                                    time_bin_col='time_bin',
                                    aa_categories=passage_ohe.categories_,
                                    filename=plot_filename, # Use protein site number
                                    title=plot_title # Use protein site number
                                )
                                figure_counter += 1 # Increment counter for each frequency plot generated
                            else:
                                print(f"Warning: Column {site_col_name} (for original FASTA position {original_fasta_position}) not found for frequency plot of protein site {protein_site_num}.")
                    else:
                        print("Warning: Not enough date range to create multiple time bins for frequency plots.")
                else:
                    print("Warning: Could not determine valid date range or bins for frequency plots.")
            else:
                print("No valid top protein sites identified from SHAP results to plot frequencies.")
        else:
            print("Skipping frequency plots: Aggregated SHAP data for date model not available.")

        # --- End of Date Model Section ---

    print("\n--- Analysis Script Finished ---")

if __name__ == "__main__":
    main()