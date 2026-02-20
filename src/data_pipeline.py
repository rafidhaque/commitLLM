
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = 'bigcode/commitpackft'
DATASET_CONFIG = 'python'
DATASET_SPLIT = 'train'
OUTPUT_DIR = Path('data')
OUTPUT_FILE = OUTPUT_DIR / 'processed_commits.csv'

# Filtering criteria
MIN_MSG_LENGTH = 15
MAX_MSG_LENGTH = 150
MIN_DIFF_LENGTH = 20
MAX_DIFF_LENGTH = 2000
FORBIDDEN_SUBSTRINGS = ['Merge pull request', 'bot']

# Target number of samples
TARGET_SAMPLE_COUNT = 10000

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def is_valid_sample(sample):
    """
    Checks if a given sample from the dataset meets the defined criteria.
    """
    msg_len = len(sample.get('message', ''))
    diff_len = len(sample.get('diff', ''))

    if not (MIN_MSG_LENGTH <= msg_len <= MAX_MSG_LENGTH):
        return False
    
    if not (MIN_DIFF_LENGTH <= diff_len <= MAX_DIFF_LENGTH):
        return False

    message_lower = sample['message'].lower()
    if any(sub in message_lower for sub in FORBIDDEN_SUBSTRINGS):
        return False
        
    return True

def main():
    """
    Main function to download, filter, and save the dataset.
    """
    logging.info(f"Starting data processing pipeline...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    logging.info(f"Loading dataset '{DATASET_NAME}' ({DATASET_CONFIG} config) in streaming mode.")
    try:
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            streaming=True
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    valid_samples = []
    processed_count = 0

    logging.info(f"Starting to filter stream for {TARGET_SAMPLE_COUNT} valid samples...")

    for sample in dataset:
        processed_count += 1
        
        if is_valid_sample(sample):
            valid_samples.append({
                'input': sample['diff'],
                'target': sample['message']
            })
            
            if len(valid_samples) % 100 == 0:
                 logging.info(f"Collected {len(valid_samples)} / {TARGET_SAMPLE_COUNT} valid samples...")

        if processed_count % 10000 == 0:
            logging.info(f"Processed {processed_count} raw samples...")

        if len(valid_samples) >= TARGET_SAMPLE_COUNT:
            logging.info(f"Target sample count of {TARGET_SAMPLE_COUNT} reached.")
            break
            
    if not valid_samples:
        logging.warning("No valid samples were found. Exiting.")
        return

    logging.info(f"Creating DataFrame with {len(valid_samples)} samples.")
    df = pd.DataFrame(valid_samples)

    logging.info(f"Saving DataFrame to '{OUTPUT_FILE}'")
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Successfully saved data to '{OUTPUT_FILE}'.")
    except IOError as e:
        logging.error(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
