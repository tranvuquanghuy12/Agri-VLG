import os
import re
import pandas as pd
import argparse

def extract_results(output_dir):
    results = []
    print(f"🔍 Scanning {output_dir} for results...")
    
    for root, dirs, files in os.walk(output_dir):
        if "main.log" in files:
            log_path = os.path.join(root, "main.log")
            with open(log_path, "r") as f:
                content = f.read()
                # Find the CSV Summary line
                match = re.search(r"CSV Summary: (.*)", content)
                if match:
                    summary_data = match.group(1).split(",")
                    results.append(summary_data)

    if not results:
        print("❌ No results found!")
        return

    # Define columns based on our standard logging format
    columns = ["Dataset", "Method", "DataSource", "Prompt", "Shots", "Seed", "RetrievalSplit", "ZeroShot", "Stage1", "WiSE-FT", "LinearProbe", "FSFT"]
    df = pd.DataFrame(results, columns=columns)
    
    # Sort and display
    df["Shots"] = pd.to_numeric(df["Shots"])
    df = df.sort_values(by=["Dataset", "Shots", "Method"])
    
    print("\n🏆 --- AGRI-VLG CONSOLIDATED RESULTS --- 🏆")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = "consolidated_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="output", help="Directory to scan for main.log files")
    args = parser.parse_args()
    
    extract_results(args.dir)
