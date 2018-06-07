# Time event

<!-- toc -->

event 是异步执行的，GPU在执行任务时，CPU端继续执行，当调用cudaEventRecord()时，GPU会将这个event放在任务队列中。当event放在队列中时，event会在把排在任务队列前面的任务执行完成，然后再继续执行event。因此，如果我们直接读取stop的时间时，cudaEventRecord()可能没有执行，所以我们在后面加上cudaEventSynchronize()来确定在读取时间之前cudaEventRecord( stop, 0 )已经执行完成。 如果没有最后的同步，读取的时间可能是不对的。
```C
int main( void ) {
	// capture the start time
	cudaEvent_t start, stop;
 	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	// do some work on the GPU

	// get stop time, and display the timing results
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime,
		start, stop );
	printf( "Time to generate: %3.1f ms\n", elapsedTime );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}
```