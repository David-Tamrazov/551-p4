#!/bin/bash

# Print help message
printHelp() {
    cat<<EOF
Prepare output directories
Usage: 
    ${0} <output-dir>
EOF
}

# Starting function
main() {
    if [[ ${#@} != 1 ]]; then
        printHelp
    else
        for dir in 0 1 2 3 4 5 6 7 8 9
        do
            mkdir -p "$1/images/$dir"
        done
    fi
}

# Start
main $@
