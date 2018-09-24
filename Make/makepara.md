## make 输入参数

<!-- toc -->

```sh
ifeq ($(arch), KNL)
        ifdef compiler
                CC=$(compiler)
                CFLAGS=-xMIC-AVX512
        else
                CC=gcc
                CFLAGS=-mfma -mavx512f -m512pf -m1512er -mavx512cd
        endif
else
        ifdef compiler
                CC=$(compiler)
                CFLAGS=-xcore-AVX512
        else
                CC=gcc
                CFLAGS=-mfma -mavx512f -mavx512cd
        endif
endif

all: main

main: main.cc
        $(CC) $(CFLAGS) main.cc
```

不同的参数会使编译器和编译选项不同


```sh

make
gcc -mfma -mavx512f -mavx512cd main.cc

make arch=KNL
gcc -mfma -mavx512f -m512pf -m1512er -mavx512cd main.cc

make compiler=icc
icc -xcore-AVX512 main.cc

make arch=KNL compiler=icc
icc -xMIC-AVX512 main.cc

```
