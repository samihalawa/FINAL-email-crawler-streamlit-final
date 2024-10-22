#!/bin/bash

iteration=0  # Initialize iteration counter
max_iterations=30  # Set the maximum number of iterations
input_file="$1"  # Check if an input file is provided as a command-line argument

# Function to perform enhancement with Aider
run_aider() {
    local file="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    echo "[$timestamp] Analyzing with Aider on file: \"$file\" (Iteration: $iteration)..."

    # Run Aider for enhancement
    aider "$file" --map-refresh="always" --yes --model="gpt-4o-mini" --message="DEBUG without talking (EDIT ONLY WRONG FATAL THINGS IF ANY) Analyze the current logic and UI, and identify and fix any critical errors or bugs that prevent the application from functioning correctly. Avoid making any changes that are not strictly necessary to address these issues."

    if [[ $? -ne 0 ]]; then
        echo "[$timestamp] Enhancement process encountered an issue. Retrying in 1 hour... (Iteration: $iteration)"
        return 1
    else
        echo "[$timestamp] Enhancement process completed successfully. Waiting for the next iteration... (Iteration: $iteration)"
        return 0
    fi
}

# Prompt for input file if not provided as an argument
if [[ -z "$input_file" ]]; then
    while true; do
        read -p "Enter the file path to analyze (or press ENTER to run without a file): " input_file
        if [[ -f "$input_file" ]] || [[ -z "$input_file" ]]; then
            break
        else
            echo "Invalid input. Please enter a valid file path or press ENTER to proceed without a file."
        fi
    done
fi

# Check if the input file is valid or not required
if [[ ! -f "$input_file" ]] && [[ -n "$input_file" ]]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

# Main loop for continuous analysis
while (( iteration < max_iterations )); do
    iteration=$((iteration + 1))  # Increment iteration counter

    # Run Aider with the provided input file
    if ! run_aider "$input_file"; then
        sleep 3600  # Wait for an hour before retrying in case of failure
        continue
    fi

    # Clear Aider messages after every 4 iterations to prevent clutter
    if (( iteration % 4 == 0 )); then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Clearing Aider messages..."
        aider --message="/clear" >/dev/null 2>&1
    fi

    # Lint the input file every 5 iterations if a file is provided
    if (( iteration % 5 == 0 )) && [[ -f "$input_file" ]]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Linting the input file with Aider..."
        if ! aider "$input_file" --lint; then
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] Linting encountered issues."
        else
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] Input file linted successfully."
        fi
    fi

    sleep 3  # Wait for an hour before the next iteration
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Reached maximum iterations ($max_iterations). Exiting..."
