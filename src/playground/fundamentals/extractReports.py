import time
import pandas as pd
import requests
from pathlib import Path

# Pfade für Input und Output
INPUT_CSV = "src/fundamentals/usefulURLs.csv"
OUTPUT_DIR = Path("localdata/reportsUseful")
ERROR_LOG = "localdata/failed_urlsUseful.csv"

# Ordner erstellen, falls er nicht existiert
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/" # Täuscht vor, du hättest den Link über Google gefunden
}

def download_reports():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Fehler: Die Datei {INPUT_CSV} wurde nicht gefunden.")
        return

    # Eindeutige Berichte extrahieren
    unique_reports = df[['report_name', 'url']].drop_duplicates()
    
    failed_downloads = []

    for _, row in unique_reports.iterrows():
        file_name = row['report_name']
        url = row['url']
        save_path = OUTPUT_DIR / file_name

        # Skip, wenn die Datei bereits existiert
        if save_path.exists():
            continue

        time.sleep(1) # Server-Schutz
        try:
            print(f"Lade herunter: {file_name}...")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
                
        except Exception as e:
            print(f"Fehler bei {file_name}: {e}")
            failed_downloads.append({
                "report_name": file_name,
                "url": url,
                "error": str(e)
            })

    # Fehler am Ende abspeichern
    if failed_downloads:
        error_df = pd.DataFrame(failed_downloads)
        error_df.to_csv(ERROR_LOG, index=False)
        print(f"\nFehlgeschlagene Downloads wurden in '{ERROR_LOG}' gesichert.")

if __name__ == "__main__":
    download_reports()