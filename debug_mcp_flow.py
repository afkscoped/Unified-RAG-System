import os
import sys
from pathlib import Path
import pandas as pd
import io

# Add project root to path
sys.path.append(os.getcwd())

from src.meta_analysis.synthetic.utils.mcp_adapter import SyntheticMCPAdapter
from src.meta_analysis.mcp_servers.csv_experiment_server import CSVExperimentMCP
from src.meta_analysis.synthetic.utils.schema_mapper import SchemaMapper
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_flow():
    print("--- 1. Checking Data Directory ---")
    mcp = CSVExperimentMCP()
    data_dir = Path(mcp.data_dir)
    print(f"Data directory: {data_dir.absolute()}")
    if not data_dir.exists():
        print("CREATING data directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- 2. Creating Synthetic Data ---")
    # Create sample synthetic dataframe exactly like generator produces
    df = pd.DataFrame({
        'experiment_id': ['exp_1', 'exp_2'],
        'experiment_name': ['Test 1', 'Test 2'],
        'control_visitors': [1000, 2000],
        'control_conversions': [100, 200],
        'variant_visitors': [1000, 2000],
        'variant_conversions': [110, 220],
        'platform': ['web', 'web'],
        'domain': ['ecommerce', 'ecommerce']
    })
    print(f"Created DF with columns: {df.columns.tolist()}")
    
    print("\n--- 3. Standardizing Data ---")
    standardized_df = SchemaMapper.to_standardized_study(df)
    print(f"Standardized DF columns: {standardized_df.columns.tolist()}")
    
    if 'control_total' not in standardized_df.columns:
        print("ERROR: control_total missing from standardized DF!")
    else:
        print("SUCCESS: control_total present")

    print("\n--- 4. Uploading to MCP ---")
    csv_buffer = io.StringIO()
    standardized_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    filename = "debug_test_file.csv"
    saved_path = mcp.upload_file(csv_bytes, filename)
    print(f"Saved to: {saved_path}")
    
    if os.path.exists(saved_path):
        print(f"File exists on disk, size: {os.path.getsize(saved_path)} bytes")
    else:
        print("ERROR: File not found on disk after upload!")
        return

    print("\n--- 5. Parsing MCP Data ---")
    studies = mcp.parse_ab_test_csv(saved_path)
    print(f"Parsed {len(studies)} studies")
    
    if len(studies) == 0:
        print("ERROR: Parsed 0 studies!")
        # Inspect file content
        print("File content preview:")
        with open(saved_path, 'r') as f:
            print(f.read())
    else:
        print(f"SUCCESS: Found {len(studies)} studies")
        print(f"First study: {studies[0]}")

if __name__ == "__main__":
    try:
        test_flow()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
