#!/bin/bash

# Initialize variables
script="neurons/validator.py"
generate_script="runnable/generate_request_outputs.py"
autoRunLoc=$(readlink -f "$0")
proc_name="vanta"
generate_proc_name="generate"
args=()
generate_args=() # Assuming no specific arguments to the generate script
version_location="meta/meta.json"
version=".subnet_version"
start_generate=false

old_args=$@

echo "$old_args"

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Define your function for version comparison and other utilities here

# Checks if $1 is smaller than $2
version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Checks if $1 is smaller than $2
version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    # Extract the version numbers from the tags
    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    # Split the version numbers into an array
    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    # Calculate the numerical difference
    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        # Compare the numbers and update the difference
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)

    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"
    local branch="$4"

    local url="https://api.github.com/repos/$repo/contents/$file_path?ref=$branch"

    local response=$(timeout 30 curl -s "$url" 2>/dev/null)
    local curl_exit_code=$?

    # Check if curl timed out or failed
    if [ $curl_exit_code -ne 0 ]; then
        echo "Error: Failed to retrieve file contents from GitHub (timeout or network error)."
        return 1
    fi

    # Check if the response contains an error message
    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    # Extract the base64 content and decode it
    json_content=$(echo "$response" | jq -r '.content' | base64 --decode 2>/dev/null)

    # Check if jq/base64 failed
    if [ $? -ne 0 ]; then
        echo "Error: Failed to decode GitHub response."
        return 1
    fi

    # Extract the "subnet_version" value using jq
    subnet_version=$(echo "$json_content" | jq -r '.subnet_version' 2>/dev/null)

    # Check if jq failed or returned null
    if [ $? -ne 0 ] || [ "$subnet_version" = "null" ]; then
        echo "Error: Failed to extract subnet_version from JSON."
        return 1
    fi

    # Print the value
    echo "$subnet_version"
}

strip_quotes() {
    local input="$1"

    # Remove leading and trailing quotes using parameter expansion
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

read_version_value() {
    jq -r $version "$version_location"
}

check_package_installed "jq"
if [ "$?" -ne 1 ]; then
    echo "Missing 'jq'. Please install it first."
    exit 1
fi

if [ ! -d "./.git" ]; then
    echo "This installation does not seem to be a Git repository. Please install from source."
    exit 1
fi

# Loop through all command line arguments
# Similar logic to handle script arguments; adjust as necessary

while [[ $# -gt 0 ]]; do
  arg="$1"

  if [[ "$arg" == -* ]]; then
    if [[ $# -gt 1 && "$2" != -* ]]; then
      if [[ "$arg" == "--script" ]]; then
        script="$2";
        shift 2
      else
        args+=("$arg")
        args+=("$2")
        shift 2
      fi
    else
        args+=("$arg")
      shift
    fi
  else
    args+=("$arg")
    shift
  fi
done

branch=$(git branch --show-current)
echo "Watching branch: $branch"
echo "PM2 process names: $proc_name"

current_version=$(read_version_value)

# Function to check and restart pm2 processes
check_and_restart_pm2() {
    local proc_name=$1
    local script_path=$2
    local -n proc_args_ref=$3

    if pm2 status | grep -q $proc_name; then
        echo "The script $script_path is already running with pm2 under the name $proc_name. Stopping and restarting..."
        pm2 delete $proc_name
    fi

    echo "Running $script_path with the following pm2 config:"

    joined_args=$(printf "'%s'," "${proc_args_ref[@]}")
    joined_args=${joined_args%,}

    echo "module.exports = {
      apps : [{
        name   : '$proc_name',
        script : '$script_path',
        interpreter: 'python3',
        min_uptime: '5m',
        max_restarts: '5',
        args: [$joined_args]
      }]
    }" > $proc_name.app.config.js

    cat $proc_name.app.config.js
    pm2 start $proc_name.app.config.js
}

# Initial call to start both processes before entering the update loop
pip install -e .
# Fixed: Proper array passing
check_and_restart_pm2 "$proc_name" "$script" args
if [ "$start_generate" = true ]; then
    check_and_restart_pm2 "$generate_proc_name" "$generate_script" generate_args
fi

backoff=1
max_backoff=60
max_retries=5

# Continuous checking and updating logic
while true; do
    # Check if current minute is divisible by 30
    current_minute=$(date +'%M')
    if [[ "$current_minute" != "07" && "$current_minute" != "37" ]]; then
        sleep 1 # Sleep for one second and check again
        continue
    fi

    retry_count=0
    latest_version=""
    current_backoff=$backoff

    echo "Starting version check at $(date)"

    while [ $retry_count -lt $max_retries ] && [ -z "$latest_version" ]; do
        retry_count=$((retry_count + 1))
        echo "Checking for latest version... (attempt $retry_count/$max_retries)"

        latest_version=$(check_variable_value_on_github "taoshidev/vanta-network" "$version_location" "$version" "$branch")

        # Check if we got a valid version (not an error message)
        if [ -n "$latest_version" ] && ! echo "$latest_version" | grep -q "^Error:"; then
            echo "Successfully retrieved latest version: $latest_version"
            break
        else
            latest_version=""  # Clear it if it was an error
            if [ $retry_count -lt $max_retries ]; then
                echo "Failed to get version. Retrying in $current_backoff seconds..."
                sleep $current_backoff
                current_backoff=$(( current_backoff * 2 ))
                if [ $current_backoff -gt $max_backoff ]; then
                    current_backoff=$max_backoff
                fi
            fi
        fi
    done

    # Check if we failed to get version after all retries
    if [ -z "$latest_version" ]; then
        echo "Failed to retrieve latest version after $max_retries attempts. Skipping this check cycle."
        sleep 300
        continue
    fi

    echo "Latest version: $latest_version"
    latest_version="${latest_version#"${latest_version%%[![:space:]]*}"}"
    current_version="${current_version#"${current_version%%[![:space:]]*}"}"

    if [ -n "$latest_version" ] && ! echo "$latest_version" | grep -q "Error" && version_less_than "$current_version" "$latest_version"; then
        echo "Updating due to version mismatch. Current: $current_version, Latest: $latest_version"
        if git pull origin "$branch"; then
            echo "New version published. Updating the local copy."
            if pip install -e .; then
                echo "Package installation successful."
                # Fixed: Proper array passing
                check_and_restart_pm2 "$proc_name" "$script" args
                if [ "$start_generate" = true ]; then
                    check_and_restart_pm2 "$generate_proc_name" "$generate_script" generate_args
                fi
                current_version=$(read_version_value)
                echo "Update completed. Continuing monitoring..."
            else
                echo "Error: Package installation failed. Please check the logs."
            fi
        else
            echo "Error: Git pull failed. Please stash your changes using git stash."
        fi
    else
        echo "You are up-to-date with the latest version."
    fi

    echo "Sleeping for 300 seconds until next check..."
    sleep 300
done
