#!/bin/bash
# llama-mercurius fork (alexhergomz/llama-mercurius, branch mercurius) decode speed at context depths, NF4 GGUF (the
# trained NF4 codes bit for bit) and IQ4_NL. Run under the GPU lock.
BIN=/mnt/ext/opendecider/llama-tq/tom/build/bin
G=/mnt/ext/opendecider/gguf
cd $BIN
for m in mercurius-1-4b-NF4.gguf mercurius-1-4b-IQ4_NL.gguf; do
  echo "== $m"
  LD_LIBRARY_PATH=$BIN ./llama-bench -m $G/$m -ngl 99 -fa on -p 0 -n 128 -d 1024,8192,32768 -r 2 -ub 512 -b 2048 -mmp 0 -o json
done
echo LLAMA_BENCH_DONE
