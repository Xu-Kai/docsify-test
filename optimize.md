# How to optimize the application?

## reducing communication costs

### Reduce overhead of communication to sender/receiver

 Send fewer messages, make messages larger (amortize overhead)

 Coalesce many small messages into large ones

### Reduce latency of communication

 Application writer: restructure code to exploit locality

 Hardware implementor: improve communication architecture

### Reduce contention

 Replicate contended resources (e.g., local copies, fine-grained locks)

 Stagger access to contended resources

### Increase communication/computation overlap

 Application writer: use asynchronous communication (e.g., async messages)

 HW implementor: pipelining, multi-threading, pre-fetching, out-of-order exec
 
 Requires additional concurrency in application (more concurrency than number
of execution units)