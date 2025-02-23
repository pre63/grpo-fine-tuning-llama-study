import glob
import json

# Set up logging
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_timestamp(filename):
  """Extract timestamp from filename and convert to datetime object."""
  try:
    # Extract the timestamp part (e.g., "2025-02-22-20-35") from the filename
    timestamp_str = filename.split("_")[-1].replace(".json", "")
    return datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M")
  except ValueError as e:
    logger.error(f"Failed to parse timestamp from {filename}: {e}")
    return datetime.min  # Default to earliest possible time if parsing fails


def merge_json_files(input_dir=".predictions", output_file="merged_predictions.json"):
  """
    Merge all JSON files in the input directory into a single file, prioritizing most recent for key collisions.

    Args:
        input_dir (str): Directory containing JSON files.
        output_file (str): Path to save the merged JSON file.
    """
  # Get all JSON files in the directory
  json_files = glob.glob(os.path.join(input_dir, "*.json"))
  if not json_files:
    logger.error(f"No JSON files found in {input_dir}")
    return

  # Sort files by timestamp (most recent last)
  json_files.sort(key=parse_timestamp)
  logger.info(f"Found {len(json_files)} JSON files to merge: {json_files}")

  # Merge dictionaries with priority to most recent
  merged_data = {}
  for json_file in json_files:
    try:
      with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
          logger.error(f"File {json_file} does not contain a dictionary: {type(data)}")
          continue
        merged_data.update(data)  # Later files overwrite earlier ones
        logger.info(f"Processed {json_file} with {len(data)} entries")
    except json.JSONDecodeError as e:
      logger.error(f"Failed to decode JSON in {json_file}: {e}")
    except Exception as e:
      logger.error(f"Error reading {json_file}: {e}")

  # Save merged data to output file
  try:
    with open(output_file, "w", encoding="utf-8") as f:
      json.dump(merged_data, f, indent=4)
    logger.info(f"Merged data saved to {output_file} with {len(merged_data)} entries")
  except Exception as e:
    logger.error(f"Failed to write merged data to {output_file}: {e}")


if __name__ == "__main__":
  merge_json_files(input_dir=".predictions", output_file="merged_predictions.json")
