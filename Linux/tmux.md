<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

# tmux 安装

tmux depends on libevent 2.x. Download it from:

	http://libevent.org

It also depends on ncurses, available from:

	http://invisible-island.net/ncurses/




```sh
./configure --prefix=/usr/local \
	--without-cxx --without-cxx-binding --without-ada --without-progs --without-curses-h \
	--with-shared --without-debug \
	--enable-widec --enable-const --enable-ext-colors --enable-sigwinch --enable-wgetch-events
```

To build and install tmux from a release tarball, use:

	$ ./configure && make
	$ sudo make install
	
vim ESC 延迟问题是tmux的转义延迟.下面的tmux设置应该更正：

set -s escape-time 0	