'''
Description: 绘图优化,报错可能是因为传入的参数没有指定类型,应该是matplotlib中的ax的类型
Author: Cyletix
Date: 2023-03-16 19:38:56
LastEditTime: 2023-03-21 01:43:22
FilePath: \AutoMakeosuFile\plot_animation.py
'''
import matplotlib.pyplot as plt
import numpy as np
import time

x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)

fig, axes = plt.subplots(nrows=6)

fig.show()

# We need to draw the canvas before we start animating...
fig.canvas.draw()

styles = ['r-', 'g-', 'y-', 'm-', 'k-', 'c-']
def plot(ax, style):
    return ax.plot(x, y, style, animated=True)[0]
lines = [plot(ax, style) for ax, style in zip(axes, styles)]

# Let's capture the background of the figure
backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]

tstart = time.time()
for i in range(1, 2000):
    items = enumerate(zip(lines, axes, backgrounds), start=1)
    for j, (line, ax, background) in items:
        fig.canvas.restore_region(background)
        line.set_ydata(np.sin(j*x + i/10.0))
        ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)

print ('FPS:' , 2000/(time.time()-tstart))






import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)

fig, axes = plt.subplots(nrows=6, sharex=True)

styles = ['r-', 'g-', 'y-', 'm-', 'k-', 'c-']
def plot(ax, style):
    return ax.plot(x, y, style, animated=True)[0]
lines = [plot(ax, style) for ax, style in zip(axes, styles)]

def animate(i):
    for j, line in enumerate(lines, start=1):
        line.set_ydata(np.sin(j*x + i/10.0))
    return lines

# We'd normally specify a reasonable "interval" here...
ani = animation.FuncAnimation(fig, animate, range(1, 200), 
                              interval=0, blit=True)
plt.show()