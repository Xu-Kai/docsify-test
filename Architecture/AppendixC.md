# Appendix C : Pipelining
	
## Introduction

Pipelining is an implementation technique whereby multiple instructions are overlapped
in execution; it takes advantage of parallelism that exists among the actions
needed to execute an instruction.


### The Basics of the RISC V Instruciton Set

Every instruction in this RISC subset can be implemented in, at most, 5 clock
cycles. The 5 clock cycles are as follows

 1. Instruction fetch cycle (IF)

 2. Instruction decode/register fetch cycle (ID)

 3. Execution/effective address cycle (EX)

	-  Memory reference—The ALU adds the base register and the offset to form
the effective address.

	- Register-Register ALU instruction—The ALU performs the operation specified
by the ALU opcode on the values read from the register file.

	-  Register-Immediate ALU instruction—The ALU performs the operation specified 
by the ALU opcode on the first value read from the register file
and the sign-extended immediate.

	- Conditional branch—Determine whether the condition is true.

 4. Memory access (MEM)

 5. Write-back cycle (WB) 

## The Major Hurdle of Pipelining—Pipeline Hazards

1. Structural hazards arise from resource conflicts when the hardware cannot support
all possible combinations of instructions simultaneously in overlapped execution.
In modern processors, structural hazards occur primarily in special
purpose functional units that are less frequently used (such as floating point
divide or other complex long running instructions). They are not a major performance
factor, assuming programmers and compiler writers are aware of the
lower throughput of these instructions. Instead of spending more time on this
infrequent case, we focus on the two other hazards that are much more frequent.

2. Data hazards arise when an instruction depends on the results of a previous
instruction in a way that is exposed by the overlapping of instructions in the
pipeline.

3. Control hazards arise from the pipelining of branches and other instructions
that change the PC.

### Data Hazards

 1. Read After Write (RAW) hazard: the most common, these occur when a
read of register x by instruction j occurs before the write of register x by instruction
i. If this hazard were not prevented instruction j would use the wrong value
of x.

2. Write After Read (WAR) hazard: this hazard occurs when read of register x by
instruction i occurs after a write of register x by instruction j. In this case,
instruction i would use the wrong value of x. WAR hazards are impossible
in the simple five stage, integrer pipeline, but they occur when instructions
are reordered.

3. Write After Write (WAW) hazard: this hazard occurs when write of register x by
instruction i occurs after a write of register x by instruction j. When this occurs,
register x will have the wrong value going forward. WAR hazards are also
impossible in the simple five stage, integrer pipeline, but they occur when
instructions are reordered or when running times vary.

#### Solve
- Forwarding 
- Stalls 

### Branch Hazards

1. freeze or flush the pipeline, holding
or deleting any instructions after the branch until the branch destination is known

2. treat every branch as not taken, simply allowing the hardware to continue as if the
branch were not executed.

3. treat every branch as taken

4. delayed branch

#### Prediction

* Static Branch Prediction

A key way to improve compile-time branch prediction is to use profile information
collected from earlier runs.

* Dynamic Branch Prediction and Branch-Prediction Buffers

The simplest dynamic branch-prediction scheme is a branch-prediction buffer or
branch history table. A branch-prediction buffer is a small memory indexed by the
lower portion of the address of the branch instruction. The memory contains a bit
that says whether the branch was recently taken or not.

This simple 1-bit prediction scheme has a performance shortcoming. To remedy this weakness, 2-bit prediction schemes are often used.

## Exceptions

### Types of Exceptions and Requirements

I/O device request

Invoking an operating system service from a user program

Tracing instruction execution

Breakpoint (programmer-requested interrupt)

Integer arithmetic overflow

FP arithmetic anomaly

Page fault (not in main memory)

Misaligned memory accesses (if alignment is required)

Memory protection violation

Using an undefined or unimplemented instruction

Hardware malfunctions

Power failure

### Stopping and Restarting Execution

As in unpipelined implementations, the most difficult exceptions have two properties:
(1) they occur within instructions (that is, in the middle of the instruction
execution corresponding to EX or MEM pipe stages), and (2) they must be restartable.

If the pipeline can be stopped so that the instructions
just before the faulting instruction are completed and those after it can be restarted
from scratch, the pipeline is said to have precise exceptions

### Instruction Set Complications

Some processors have instructions that change the state in the
middle of the instruction execution, before the instruction and its predecessors are
guaranteed to complete.

## Handle Multicycle Operations

### Hazards and Forwarding in Longer Latency Pipelines

### Maintaining Precise Exceptions