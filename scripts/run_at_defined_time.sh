#!/bin/bash

# Ask user for target time
echo -n "Enter the target time to run the command (e.g., 04:00 or 15:30): "
read -r target_time

# Validate target_time is not empty
if [ -z "$target_time" ]; then
    echo "Error: No target time provided."
    exit 1
fi

current_epoch=$(date +%s)

# Attempt to parse the target time for today
target_epoch=$(date -d "$target_time" +%s 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error: Invalid time format. Please use HH:MM or HH:MM:SS (e.g., 14:30)."
    exit 1
fi

# If target time today has already passed, target the same time tomorrow
if [ "$current_epoch" -ge "$target_epoch" ]; then
    target_epoch=$(date -d "tomorrow $target_time" +%s)
fi

sleep_seconds=$((target_epoch - current_epoch))

echo "Current time: $(date)"
echo "Sleeping for $sleep_seconds seconds until $(date -d "@$target_epoch")"

sleep "$sleep_seconds"

echo "Starting evaluation at $(date)..."
cd "$(dirname "$0")/.."
# Activate virtual environment
source venv/bin/activate
python3 glass/cli.py run @configs/aa_lcr_gemini_full.yaml
