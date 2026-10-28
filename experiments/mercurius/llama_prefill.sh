#!/bin/bash
# llama-mercurius prompt processing speed (pp) at 1K / 8K / 32K, NF4 GGUF. Under the GPU lock.
BIN=/mnt/ext/opendecider/llama-tq/tom/build/bin
cd $BIN
LD_LIBRARY_PATH=$BIN ./llama-bench -m /mnt/ext/opendecider/gguf/mercurius-1-4b-NF4.gguf -ngl 99 -fa on -p 1024,8192,32768 -n 0 -r 1 -ub 512 -b 2048 -mmp 0 -o json
echo LLAMA_PP_DONE
