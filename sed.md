# Shell 整理

sed -i "s/few/asd/g" hello.txt

/g stands for "global", meaning to do this for the whole line. If you leave off the /g (with s/few/asd/, there always needs to be three slashes no matter what) and few appears twice on the same line, only the first few is changed to asd
