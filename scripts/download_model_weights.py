import dropbox
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Dropbox access
ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# Base path relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(BASE_DIR, "../model_weights/pytorch_model_weights.pth")
dropbox_path = "/pytorch_model_weights.pth"

# Check if file already exists
if os.path.exists(local_path):
    logging.info(f"File already exists at {local_path}, skipping download.")
else:
    try:
        logging.info(f"Downloading from Dropbox: {dropbox_path}")
        metadata, res = dbx.files_download(path=dropbox_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(res.content)

        logging.info(f"Download complete. File saved to {local_path}")
    except Exception as e:
        logging.error(f"Failed to download file from Dropbox: {e}")
