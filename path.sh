# export FAIRSEQ_ROOT=/home/andybi7676/Desktop/fairseq # <-- change to your own fairseq path

# parse REBORN_WORK_DIR from the current path to reborn-uasr
REBORN_WORK_DIR=$(pwd)
REBORN_WORK_DIR=${REBORN_WORK_DIR%/reborn-uasr*}/reborn-uasr

# Set FAIRSEQ_ROOT to the path of fairseq
export FAIRSEQ_ROOT=${REBORN_WORK_DIR}/fairseq # <-- change to your own fairseq path

# Check if PYTHONPATH is not empty and if FAIRSEQ_ROOT is not in PYTHONPATH
if [[ -n "$PYTHONPATH" && ! ":$PYTHONPATH:" == *":$FAIRSEQ_ROOT:"* ]]; then
    # If PYTHONPATH exists but does not contain FAIRSEQ_ROOT, append FAIRSEQ_ROOT to it
    export PYTHONPATH="$PYTHONPATH:$FAIRSEQ_ROOT"
    echo "Appended $FAIRSEQ_ROOT to PYTHONPATH"
elif [[ -z "$PYTHONPATH" ]]; then
    # If PYTHONPATH does not exist, create it with FAIRSEQ_ROOT
    export PYTHONPATH="$FAIRSEQ_ROOT"
    echo "Added $FAIRSEQ_ROOT to PYTHONPATH"
else
    # If PYTHONPATH contains FAIRSEQ_ROOT, print out the message
    echo "$FAIRSEQ_ROOT is already in PYTHONPATH"
fi

# We currently have not packed REBORN into a package. To run the script correctly, we need to add the path of REBORN to PYTHONPATH
# REBORN_WORK_DIR=$(pwd) # Please make sure you run the script in the root directory of REBORN (:/path/to/reborn-uasr$ source path.sh)

if [[ ! ":$PYTHONPATH:" == *":$REBORN_WORK_DIR:"* ]]; then
    # If PYTHONPATH exists but does not contain FAIRSEQ_ROOT, append FAIRSEQ_ROOT to it
    export PYTHONPATH="$PYTHONPATH:$REBORN_WORK_DIR"
    echo "Appended $REBORN_WORK_DIR to PYTHONPATH"
else
    # If PYTHONPATH contains REBORN_WORK_DIR, print out the message
    echo "$REBORN_WORK_DIR is already in PYTHONPATH"
fi

# Finished env setup. Print out the env variables for debugging
echo "======================================================================================="
echo "FAIRSEQ_ROOT: $FAIRSEQ_ROOT"
echo "REBORN_WORK_DIR: $REBORN_WORK_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "Please make sure that FAIRSEQ_ROOT and REBORN_WORK_DIR are in PYTHONPATH"
echo "During each runtime, please make sure to run \`source path.sh\` to set up the environment."
echo "======================================================================================="
# Test the environment
python ${REBORN_WORK_DIR}/rl/utils/test_path.py # make sure that you are under REBORN_WORK_DIR
# If the output is "SUCCESS", then the environment is set up correctly in the current runtime