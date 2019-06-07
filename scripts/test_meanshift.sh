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

spatial_radius=13
color_radius=100
test_path=$1

echo ">> Generating reference results"
for file in data/meanshift_filter/*.jpg; do
    file=`basename $file`
    filename="${file%%.*}"
    if [[ -e "data/meanshift_filter/$filename.bin" ]]; then
        echo ">>>> Skipping file $filename.jpg"
        continue
    fi
    echo ">>>> Processing file $filename.jpg"
    ./build/bin/meanshift_filter_gen data/meanshift_filter/$filename.jpg \
                                     data/meanshift_filter/$filename.bin \
                                     $spatial_radius $color_radius
done

echo ">> Running result comparison"
for file in data/meanshift_filter/*.bin; do
    file=`basename $file`
    filename="${file%%.*}"
    if [[ -e "$test_path/$filename.bin" ]]; then
        echo ">>>> Testing $filename.bin"
        echo ">>>> Result:"
        ./build/bin/meanshift_filter_cmp $test_path/$filename.bin \
                                         data/meanshift_filter/$filename.bin
        echo ""
    fi
done