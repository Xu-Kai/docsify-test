# GDB 调试整理
<!-- toc -->

## 断点设置

break ... if value==9

watch n > 9

判断寄存器的值

break \*addr

在代码内存地址addr设置断点(加 \*)

break line

在源代码文件里面第line行设置断点(不加 \*)

delete

用法：delete [breakpoints num] [range...]

## 显示代码

layout asm

显示汇编

set disassembly-flavor intel

显示汇编时格式是Intel格式，默认是AT&T格式

layout src 

显示c源代码

layout regs

显示寄存器的值

ctrl + x, a 

退出layout模式

## 打印值

(gdb) p /x $rbx

$9 = 0x7fffed05c200

打印寄存器的值

## 反汇编

使用objdump反汇编时

使用 -M intel 显示intel格式的汇编