




fig, axs = plt.subplots(2, 2, figsize=[7, 7], dpi=500)

for n in range(len(ximins)):
    ax = axs[n // 2, n % 2]
    i = ximins[n]
    for j in range(len(ys)):
        y = ys[j]
        ps = poisson[j]
        om = omegas[j]
        res = []
        for k in range(len(snapshots)):
            res.append(integ_mrate(3*mlim, m_reds[snapshots[k]], xi_min=dexis[i], xi_max=1, om0=om, sig8=s8))
        if j == 0 and n == 0:
            ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j], label=r'EC $\Omega_m$={:1.2f}'.format(om))
            ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[i],
                       label=r'$\xi>$ {:1.2f}, $\Omega_m$={:1.2f}'.format(dexis[i], om))
        elif j == 0:
            ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[j],
                       label=r'$\xi>$ {:1.2f}'.format(dexis[i]))
            ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j])
        elif n == 0:
            ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j], label=r'EC $\Omega_m$={:1.2f}'.format(om))
            ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[j],
                       label=r'$\Omega_m$={:1.2f}'.format(om))
        else:
            ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[j])
            ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j])

        ax.fill_between(1 + m_reds[snapshots], y[i] - ps[i], y[i] + ps[i], color=colors[j], alpha=0.2)

    ax.set_xlabel('1+z', size=15)
    ax.set_ylabel(r'dN(>$\xi$/dz [mergers/halo/dz]')

    ax.legend()
plt.show()