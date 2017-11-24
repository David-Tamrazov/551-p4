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
        for dir in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 24 25 27 28 30 32 35 36 40 42 45 48 49 54 56 63 64 72 81 777
        do
            mkdir -p "$1/images/$dir"
        done
    fi
}

# Start
main $@
