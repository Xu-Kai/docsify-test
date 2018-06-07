# Constant memory

<!-- toc -->

## 从host考数据到device

```C
__constant__ Sphere s[SPHERES];
Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
for (int i=0; i<SPHERES; i++) {
	temp_s[i].r = rnd( 1.0f );
	temp_s[i].g = rnd( 1.0f );
	temp_s[i].b = rnd( 1.0f );
	temp_s[i].x = rnd( 1000.0f ) - 500;
	temp_s[i].y = rnd( 1000.0f ) - 500;
	temp_s[i].z = rnd( 1000.0f ) - 500;
	temp_s[i].radius = rnd( 100.0f ) + 20;
}
cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES );
free( temp_s );
```
## constant memory 特点

constant memory 是只读的， 在同个warp内的线程读取constant memory 的同一个数据时，由半个warp也就是16个线程中的一个读取然后在广播到其他相同的半个warp中的15个线程。如果从constant Memory读取大量数据，相比于global memory 只有1/16的数据传输。

因为数据是只读的，因此硬件可以使用贪婪策略来对constant 数据进行cache。所以当从constant memory第一次读取数据时，另一半warp 16个线程，可以直接从constant cache中读取，不产生额外的访存。而且当其他warp访问同一个constant memory时，不会产生新的访存。但是如果半个warp中不同的线程访问不同的constant memory地址，16次访问constant memory会被串行化。