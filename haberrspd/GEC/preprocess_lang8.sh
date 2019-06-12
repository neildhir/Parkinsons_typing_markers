#!/bin/bash

# This function takes as input a .train file from the
# Lang-8 dataset.

# From: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Bash
function levenshtein {
    if (( $# != 2 )); then
        echo "Usage: $0 word1 word2" >&2
    elif (( ${#1} < ${#2} )); then
        levenshtein "$2" "$1"
    else
        local str1len=${#1}
        local str2len=${#2}
        local d

        for i in $( seq 0 $(( (str1len+1)*(str2len+1) )) ); do
            d[i]=0
        done

        for i in $( seq 0 $str1len );	do
            d[i+0*str1len]=$i
        done

        for j in $( seq 0 $str2len );	do
            d[0+j*(str1len+1)]=$j
        done

        for j in $( seq 1 $str2len ); do
            for i in $( seq 1 $str1len ); do
                [ "${1:i-1:1}" = "${2:j-1:1}" ] && local cost=0 || local cost=1
                del=$(( d[(i-1)+str1len*j]+1 ))
                ins=$(( d[i+str1len*(j-1)]+1 ))
                alt=$(( d[(i-1)+str1len*(j-1)]+cost ))
                d[i+str1len*j]=$( echo -e "$del\n$ins\n$alt" | sort -n | head -1 )
            done
        done
        echo ${d[str1len+str1len*(str2len)]}
    fi
}

# Check that input ends with "train"

### ----- file input

printf "You passed the file [ensure it ends with .train]: $1\n"

# Some checks to ensure that the input are sensible.
if [ $# -eq 0 ]
then
    echo "No input file supplied. Try again."
    exit
fi

### ----- size of training set determined

printf "How many sentence-pairs do you want to extract [enter a number between 0 and 1136635]?\n"
read N

if [ -z "$N" ]
then
    echo "No argument supplied. Try again."
    exit
fi

if [[ -n ${N//[0-9]/} ]]; then
    echo "You must provide an integer, no strings. Try again."
    exit
fi

### ----- selection of the $N rows chosen by the user

# There are a number of operations that take place.
# 1) To start, select only rows which have a correction
# 2) Filter rows if they have had a correction (if first col > 0)
# 3) 'shuf' ensures that our selection is random and picks N from these
user_selected_rows=$(awk -F '\t' '{if ($1>0) print $5, $6}' $1 | shuf -n $N)

### ----- filter rows by their Levenshtein distance

$user_selected_rows | {
    # 1) Calculate the LD between all filtered sentences
    xargs levenshtein
} | {
    # 2) Orders the rows by LD
    sort
} | {
    # 3) Calculates some summary statistics
    awk '{if(min==""){min=max=$1}; if($1>max) {max=$1}; if($1<min) {min=$1}; total+=$1; count+=1} END {print total/count, max, min}'
} | {
    # 4) Selects those rows which are below the average LD
}
