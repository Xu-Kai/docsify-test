## aotuconf

<!-- toc -->

```sh
//根据configure.in和Makefile.am生成makefile的步骤，基于UBUNTU 12.04
1.autoscan （可选）
2.aclocal
3.autoconf
4.autoheader（可选）
5.libtoolize --automake --copy --debug --force（可选）
6.automake --add-missing
7.autoreconf –f –i –Wall,no–obsolete（可选）
8../configure
```
[参考](http://www.51cos.com/?p=1649)