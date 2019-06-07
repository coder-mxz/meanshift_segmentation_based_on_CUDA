#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <test_path>" >&2
  exit 1
fi
if ! [ -e "$1" ]; then
  echo "$1 not found" >&2
  exit 1
fi
if ! [ -d "$1" ]; then
  echo "$1 not a directory" >&2
  exit 1
fi

test_path=$1

echo ">> Generating reference results"
for file in data/union_find/*.png; do
    file=`basename $file`
    filename="${file%%.*}"
    if [[ -e "data/union_find/$filename.bin" ]]; then
        echo ">>>> Skipping file $filename.jpg"
        continue
    fi
    echo ">>>> Processing file $filename.png"
    ./build/bin/union_find_gen data/union_find/$filename.png \
                               data/union_find/$filename.bin
done

echo ">> Running result comparison"
for file in data/union_find/*.bin; do
    file=`basename $file`
    filename="${file%%.*}"
    if [[ -e "$test_path/$filename.bin" ]]; then
        echo ">>>> Testing $filename.bin"
        echo ">>>> Result:"
        ./build/bin/union_find_cmp $test_path/$filename.bin \
                                   data/union_find/$filename.bin
    fi
done