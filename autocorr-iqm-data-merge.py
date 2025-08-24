import json
import os
from pathlib import Path

def merge_json_files(input_directory, output_file):
    """
    Merge multiple JSON files containing arrays into a single JSON file.
    
    Args:
        input_directory (str): Directory containing JSON files to merge
        output_file (str): Path for the merged output file
    """
    merged_data = []
    
    # Get all JSON files from the input directory
    input_path = Path(input_directory)
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_directory}")
        return
    
    print(f"Found {len(json_files)} JSON files to merge:")
    
    # Process each JSON file
    for json_file in sorted(json_files):
        print(f"Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Filter and count completed jobs
            completed_jobs = []
            total_jobs = 0
            
            if isinstance(data, list):
                total_jobs = len(data)
                # Check each job for completion status
                for job in data:
                    if isinstance(job, dict) and 'completed' in job and job['completed'] is not None:
                        completed_jobs.append(job)
                        
                merged_data.extend(completed_jobs)
                print(f"  Added {len(completed_jobs)}/{total_jobs} completed jobs from {json_file.name}")
                
                if len(completed_jobs) < total_jobs:
                    print(f"  Skipped {total_jobs - len(completed_jobs)} incomplete jobs")
                    
            else:
                # If it's a single object, check if it's completed
                total_jobs = 1
                if isinstance(data, dict) and 'completed' in data and data['completed'] is not None:
                    merged_data.append(data)
                    completed_jobs.append(data)
                    print(f"  Added 1/1 completed job from {json_file.name}")
                else:
                    print(f"  Skipped 1 incomplete job from {json_file.name}")
                
        except json.JSONDecodeError as e:
            print(f"  Error reading {json_file.name}: {e}")
        except Exception as e:
            print(f"  Unexpected error with {json_file.name}: {e}")
    
    # Write merged data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMerge completed successfully!")
        print(f"Total items merged: {len(merged_data)}")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

def main():
    # Define paths
    # input_dir = "autocorr-iqm-data"
    # output_file = "autocorr-iqm-data-merged.json"
    
    input_dir = "autocorr-iqm-echo-data"
    output_file = "autocorr-iqm-echo-data-merged.json"
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    input_path = script_dir / input_dir
    output_path = script_dir / output_file
    
    print("JSON File Merger")
    print("=" * 50)
    print(f"Input directory: {input_path}")
    print(f"Output file: {output_path}")
    print()
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        return
    
    # Perform the merge
    merge_json_files(input_path, output_path)
    
    # Show some statistics about the merged file
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"Output file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
