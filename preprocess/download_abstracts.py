import pandas as pd
import os
from config import gsheets, interimdir, indir
import util as ut


def download_abstracts_from_csv(key: str, always_download: bool = False) -> None:
    """Download abstracts for a given key from its TalkID or SessionID CSV."""
    suffix = "sessionid" if key == "special_session_submissions" else "talkid"
    csv_path = os.path.join(interimdir, f"{key}_{suffix}.csv")
    
    # Validate CSV file existence
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist. Skipping {key}.")
        return
    
    df = pd.read_csv(csv_path)
    url_col = (
        "FirstNameLastNameSession.tex file with Session Title and Description"
        if key == "special_session_submissions"
        else "FirstNameLastNameAbstract.tex file with Talk Title and Abstract"
    )
    id_col = "SessionID" if key == "special_session_submissions" else "TalkID"
    
    # Validate required columns
    if not {url_col, id_col}.issubset(df.columns):
        print(f"Required columns '{url_col}' or '{id_col}' not found in {csv_path}. Skipping {key}.")
        return
    
    abstracts_dir = os.path.join(indir, "abstracts")
    os.makedirs(abstracts_dir, exist_ok=True)
    
    for item_id, url in df[[id_col, url_col]].dropna().values:
        if not isinstance(url, str) or not url.strip():
            print(f"ERROR: Invalid URL for {item_id}. Skipping...")
            continue

        try:
            direct_url = ut.gdrive_direct_download(url)
            local_filename = os.path.join(abstracts_dir, f"{item_id}.tex")
            
            # Download file if necessary
            if not os.path.exists(local_filename) or always_download:
                ut.download_file(direct_url, abstracts_dir, item_id)
                print(f"Downloaded {item_id}.tex to {local_filename}")
        except Exception as e:
            print(f"ERROR: Failed to download {item_id}. URL: {url}. Error: {e}")


if __name__ == "__main__":
    # Process all gsheets keys except 'schedule'
    for key in gsheets:
        if key != "schedule":
            download_abstracts_from_csv(key)