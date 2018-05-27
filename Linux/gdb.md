# GDB 调试整理
<!-- toc -->

## 断点设置
```sh
break ... if value==9

watch n > 9
```
判断寄存器的值
```sh
break \*addr
```
在代码内存地址addr设置断点(加 \*)
```sh
break line
```
在源代码文件里面第line行设置断点(不加 \*)
```sh
delete
```
用法：delete [breakpoints num] [range...]

## 显示代码
```sh
layout asm
```
显示汇编
```sh
set disassembly-flavor intel
```
显示汇编时格式是Intel格式，默认是AT&T格式
```sh
layout src 
```
显示c源代码
```sh
layout regs
```
显示寄存器的值
```sh
ctrl + x, a 
```
退出layout模式

```sh
set disassembly-falvor intel
```

可以在$HOME/.gdbinit文件设置，不用每次在gdb调试过程中设置
## 打印值
```sh
(gdb) p /x $rbx

$9 = 0x7fffed05c200
```
打印寄存器的值

## 反汇编

使用objdump反汇编时

使用 -M intel 显示intel格式的汇编

