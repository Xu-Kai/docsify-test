# CUDA

<!-- toc -->

## atomicAdd()

```C
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                           unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                        unsigned long long int val);
```
## byte_perm

```C
__device__ ​ unsigned int __byte_perm ( unsigned int  x, unsigned int  y, unsigned int  s )
```

- Return selected bytes from two 32 bit unsigned integers.

- Returns

The returned value r is computed to be: result[n] := input[selector[n]] where result[n] is the nth byte of r.

- Description

byte_perm(x,y,s) returns a 32-bit integer consisting of four bytes from eight input bytes provided in the two input integers x and y, as specified by a selector, s.

The input bytes are indexed as follows: input[0] = x<7:0> input[1] = x<15:8> input[2] = x<23:16> input[3] = x<31:24> input[4] = y<7:0> input[5] = y<15:8> input[6] = y<23:16> input[7] = y<31:24> The selector indices are as follows (the upper 16-bits of the selector are not used): selector[0] = s<2:0> selector[1] = s<6:4> selector[2] = s<10:8> selector[3] = s<14:12>

## inline assembly

```C
uint32_t bexor;
uint32_t bucketId;
asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketId) : "r"(bexor));
```

bfe的含义由下面给出

```
bfe.type  d, a, b, c;

.type = { .u32, .u64,
          .s32, .s64 };

.u32, .u64:
	zero
.s32, .s64:
	msb of input a if the extracted field extends 
	beyond the msb of a msb of extracted field, otherwise

asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketId) : "r"(bexor));
msb = (.type==.u32 || .type==.s32) ? 31 : 63;
pos = b & 0xff;  // pos restricted to 0..255 range
len = c & 0xff;  // len restricted to 0..255 range

if (.type==.u32 || .type==.u64 || len==0)
    sbit = 0;
else
    sbit = a[min(pos+len-1,msb)];

d = 0;
for (i=0; i<=msb; i++) {
    d[i] = (i<len && pos+i<=msb) ? a[pos+i] : sbit;
}

```

