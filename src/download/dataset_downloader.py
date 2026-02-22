from __future__ import annotations

import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

def download_file(url: str, output: Path, timeout: int = 120) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    # Special handling for SCWAD zip file
    if output.name == "CDS_test.csv" and url.endswith(".zip"):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "archive.zip"
            
            # Download the zip file
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with zip_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find and copy CDS_test.csv
            csv_files = list(Path(temp_dir).rglob("CDS_test.csv"))
            if not csv_files:
                raise FileNotFoundError("CDS_test.csv not found in the extracted archive")
            
            import shutil
            shutil.copy2(csv_files[0], output)
    else:
        # Regular file download
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with output.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)

