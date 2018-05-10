# matplotlib 画图

<!-- toc -->

## 柱状图

```python
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
from matplotlib import ticker

# print(plt.style.available)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
plt.rcParams["font.weight"] = "heavy"

plt.subplots_adjust(left=0.1, right=0.95, top=0.87, bottom=0.1)



def timeCompare():
    master = [8.08, 8.84, 8.49, 8.42, 9.58, 5.82,  8.17, 5.92]
    slave = [8.08, 8.84, 8.49, 8.42, 9.58, 5.82,  8.17, 5.92]

    master = [0.12, 0.11, 0.12, 0.12, 0.10, 0.17,  0.12,  0.17 ]
    slave = [1.44, 1.12, 1.22, 1.44, 0.87, 1.55, 1.02, 1.90]


    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x1 = [1, 2, 3, 4, 5, 6, 7, 8]

    xname = ["advect_u","advect_v","advect_w","advect_scalar","advance_uv","advance_w","advance_mut","micro_driver"]

    axis_font = {'fontname':'Times New Roman', 'size':'14'}

    plt.subplots_adjust(left=0.13, right=0.95, top=0.87, bottom=-0.65)
    plt.ylim(0,2.2)
    total_width, n = 0.8, 3
    width = total_width / n
    plt.bar(x, master, label="MPE", width=width)

    for a,b in zip(x, master):
        plt.text(a-0.08, b + 0.05, '%.2f'%b, ha='center', va='bottom', fontsize=11)

    for i in range(len(x)):
        x[i] += width
        x1[i] += width

    for a,b in zip(x, slave):
        plt.text(a-0.08, b + 0.05, '%.2f'%b, ha='center', va='bottom', fontsize=11)
    plt.bar(x, slave, label="CPE", width=width)
    for i in range(len(x)):
        x[i] += width


    plt.legend(loc="upper center", fancybox=True,   bbox_to_anchor=(0.5, 1.24), ncol = 3, shadow=True,fontsize=14)

    #plt.xticks((1.2, 2.2), ("t=1", "t=2", "t=1", "t=2", "t=1", "t=2", "t=1", "t=2", "t=1", "t=2", "t=1", "t=2", "t=1", "t=2", "t=1", "t=2"), fontsize=18)
    plt.xticks(x1, xname, rotation=45, fontsize=12, horizontalalignment="right")

    plt.yticks(fontsize=14)

    plt.ylabel("Speedup", **axis_font)
    plt.xlabel("Kernels", **axis_font)

    plt.subplots_adjust(bottom=0.28)
    plt.savefig('out/timeCompare.eps', format='eps')

    plt.show()

timeCompare()
```

## 折线图
```python
def scalability():
    plt.subplots_adjust(left=0.1, right=0.95, top=0.87, bottom=0.13)
    plt.rcParams["font.size"] = "12"
    plt.xlim(96, 5200)
    plt.ylim(1, 48)
    x = [128, 256, 512, 1024, 2048, 4096]

    y = [1, 2, 4, 8, 16, 32]

    ideal=[1, 2, 4, 8, 16,32]
    speedup=[1,1.81,3.45,6.15,10.93,17.15]


    #plt.xscale("log")
    #plt.yscale("log")
    efficiency=[1,0.904,0.861,0.769,0.684,0.536]

    plt.plot(x, ideal, "go-", label="Ideal")
    plt.plot(x,  speedup, "r^-", label="WRF efficiency", linewidth=1)

    plt.plot(x, ideal, "go-", x, speedup,"r>-")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)

    # xlabs = [0, 10, 100, 1000]

    ax = plt.gca()
    #plt.xticks(x)
    #plt.yticks(y)
    #ax.set_yticklabels(y)
    ax.set_yscale('symlog', basey=2)
    ax.set_xscale('symlog', basex=2)

    ax.set_yticks(y)
    ax.set_xticks(x)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plt.xlabel("Process number")
    plt.ylabel("Speedup")
    plt.savefig('out/scalability.eps', format='eps')
    plt.show()
scalability()
```
## 子图
```python
def blockSize():
    plt.rcParams["font.size"] = "12"
    plt.xlim(16, 256)
    x = [32,64,96,128,160,192,224]
    #plt.xlim(x)
    advect_u = [6046,4805, 5381, 5402, 5681, 5514, 5507]
    advect_v = [7220,5955,7729,7804,8079,7736,7770]
    advect_w = [6950,5577,6445,7043, 8789, 7182, 7183]
    advect_scalar = [5919, 4590, 5147, 5175, 5425, 5244, 5258]
    advance_uv = [70007, 55628, 65486, 66031, 69099, 65663, 65697]
    advance_w = [60827, 43250, 44205, 48776,66482,53021,52211]
    advance_mut = [53223, 43658, 43711, 46579, 52613, 43737, 43675]


    plt.subplots_adjust(left=0.13, right=0.95, top=0.87, bottom=0.1)

    plt.figure(1)


    plt.subplot(211)
    plt.ylim(40000, 72000)
    plt.plot(x, advance_uv, "y>-", label="advance_uv", linewidth=1)
    plt.plot(x, advance_w, "cD-", label="advance_w")
    plt.plot(x, advance_mut, "bo-", label="advance_mut", linewidth=1)

    ax = plt.gca()
    ax.set_xticklabels([])
    
    plt.xticks(x)
    plt.ylabel("time(us)")
    #plt.plot(x, advect_u, "bo-", x, advect_v, "gv-", x, advect_w, "r^-", x, advect_scalar, "cD-", x, advance_uv, "y>-", x, advance_w, "y>-",x, advance_mut, "y>-" )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, fancybox=True, shadow=False,fontsize=10)


    plt.subplot(212)
    plt.ylim(4000, 9000)
    plt.plot(x, advect_u, "bo-", label="advect_u")
    plt.plot(x, advect_v, "r^-", label="advect_v", linewidth=1)
    plt.plot(x, advect_w, "gv-", label="advect_w", linewidth=1)
    plt.plot(x, advect_scalar, "cD-", label="advect_scalar")

    plt.xticks(x)
    plt.ylabel("time(us)")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=4, fancybox=True, shadow=False,fontsize=10)

    # xlabs = [0, 10, 100, 1000]

    # ax.set_yticks(xlabs)
    # ax.set_yticklabels(xlabs)
    # ax.set_yscale('symlog', basey=2)
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plt.xlabel("Block size")
    plt.ylabel("time(us)")
    plt.savefig('out/blockSize.eps', format='eps')
    plt.show()

 blockSize()
```