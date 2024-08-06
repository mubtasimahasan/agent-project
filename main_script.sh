#!/bin/bash

echo "Running pipeline evaluation script..."

# Check if a pipeline choice is provided
if [ $# -gt 0 ]; then
    # If a choice is provided, assign it to the variable
    pipeline_choice=$1
    echo "Pipeline choice provided: $pipeline_choice"

    # Validate the pipeline choice
    if [[ "$pipeline_choice" != "static" && "$pipeline_choice" != "dynamic" && "$pipeline_choice" != "no_plan" && "$pipeline_choice" != "debug" ]]; then
        echo "Invalid pipeline choice: $pipeline_choice"
        echo "Valid options are: static, dynamic, no_plan, or debug."
        exit 1
    fi

    # If 'debug' is selected and a second argument is present, pass it
    if [[ "$pipeline_choice" == "debug" && $# -gt 1 ]]; then
        debug_value=$2
        python_args="$pipeline_choice $debug_value"
    else
        python_args="$pipeline_choice"
    fi
else
    # Default behavior if no choice is provided
    echo "No pipeline choice provided. Evaluating on all pipelines."
    pipeline_choice="all"
    python_args="$pipeline_choice"
fi

# Get date time and construct the log file name
date_time=$(date +"%m%d-%H%M")
log_file="./logs/outputs_${date_time}.log"

# Execute the Python script with the chosen arguments
echo "Executing: python main.py $python_args"
python main.py $python_args >> "$log_file"

echo "Pipeline evaluation complete. Logs saved to ./logs directory."
