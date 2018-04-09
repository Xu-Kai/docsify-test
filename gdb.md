# GDB 调试整理

判断寄存器的值

break ... if value==9

watch n > 9

set disassembly-flavor intel

delete

用法：delete [breakpoints num] [range...]

layout asm

显示汇编


(gdb) p /x $rbx

$9 = 0x7fffed05c200

打印寄存器的值

使用objdump反汇编时

使用 -M intel 显示intel格式的汇编