delta_t = .05
delta_y = .05

t_min = 0 
t_max = 1

y_min = 0 
y_max = 1


t_domain = np.arange(t_min, (t_max + delta_t) , delta_t)
y_domain = np.arange(y_min, (y_max + delta_y), delta_y)

X, Y  = np.meshgrid(t_domain, y_domain)

# z = np.exp(-X) + Y
dx, dy = np.gradient(z)


n = -2
color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

fig, ax = plt.subplots(figsize=(7,7))
ax.quiver(X,Y,dx,dy, color_array, alpha=0.8)

# fig, ax = plt.subplots(figsize=(7,7))
# ax.quiver(X,Y,dx,dy)

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_aspect('equal')

plt.show()
