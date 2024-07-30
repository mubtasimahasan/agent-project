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
else
    # Default behavior if no choice is provided
    echo "No pipeline choice provided. Evaluting on all pipelines."
    pipeline_choice="all"
fi

# Execute the Python script with the chosen pipeline
echo "Executing: python main.py $pipeline_choice"
python main.py "$pipeline_choice" >> ./logs/output.log

echo "Pipeline evaluation complete. Results appended to ./logs/output.log."
