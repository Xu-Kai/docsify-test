## aotuconf

<!-- toc -->

```sh
//根据configure.in和Makefile.am生成makefile的步骤，基于UBUNTU 12.04
autoscan （可选）
aclocal
autoconf
autoheader（可选）
libtoolize --automake --copy --debug --force（可选）
automake --add-missing
autoreconf –f –i –Wall,no–obsolete（可选）
./configure
```
[参考](http://www.51cos.com/?p=1649)