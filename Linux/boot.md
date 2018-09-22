# Linux 引导 Windows 修复
<!-- toc -->


```sh
menuentry "Windows 10"{
	insmod part_msdos
	insmod nfts
	set root='(hd0,msdos1)'
	chainloader +1
}
```
