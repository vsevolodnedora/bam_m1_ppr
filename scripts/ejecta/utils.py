class Constants:

    ns_rho = 1.6191004634e-5
    time_constant = 1.#0.004925794970773136  # to to to ms
    energy_constant = 1.#1787.5521500932314
    #volume_constant = 1.#2048

class Labels:
    def __init__(self):
        pass

    @staticmethod
    def labels(v_n, mask=None):
        # solar

        if v_n == "nsims":
            return r"$N_{\rm{resolutions}}$"

        elif v_n == 'theta':
            return r"Angle from orbital plane"

        elif v_n == 'temp' or v_n == "temperature":
            return r"$T$ [GEO]"

        elif v_n == 'phi':
            return r"Azimuthal angle"

        elif v_n == 'mass':
            return r'normed $M_{\rm{ej}}$'

        elif v_n == 'diskmass':
            return r'$M_{\rm{disk}}$ $[M_{\odot}]$'

        elif v_n == 'Mdisk3Dmax':
            return r'$M_{\rm{disk;max}}$ $[M_{\odot}]$'

        elif v_n == 'ejmass' or v_n == "Mej_tot":

            if mask == "geo_entropy_above_10":
                return r'$M_{\rm{ej;s>10}}$ $[10^{-2}M_{\odot}]$'
            elif mask == "geo_entropy_below_10":
                return r'$M_{\rm{ej;s<10}}$ $[10^{-2}M_{\odot}]$'
            else:
                return r'$M_{\rm{ej}}$ $[10^{-2}M_{\odot}]$'

        elif v_n == "Mej_tot_scaled":
            if mask == "geo_entropy_above_10":
                return r'$M_{\rm{ej;s>10}}/M_{\rm{b;tot}}$ $[10^{-2}M_{\odot}]$'
            elif mask == "geo_entropy_below_10":
                return r'$M_{\rm{ej;s<10}}/M_{\rm{b;tot}}$ $[10^{-2}M_{\odot}]$'
            else:
                return r'$M_{\rm{ej}}/M_{\rm{b;tot}}$ $[10^{-2}M_{\odot}]$'
            # else:
            #     raise NameError("label for v_n:{} mask:{} is not found".format(v_n, mask))

        elif v_n == "Mej_tot_scaled2":

            if mask == "geo_entropy_above_10":
                return r'$M_{\rm{ej;s>10}}/(\eta M_{\rm{b;tot}})$ $[10^{-2}M_{\odot}]$'
            elif mask == "geo_entropy_below_10":
                return r'$M_{\rm{ej;s<10}}/(\eta M_{\rm{b;tot}})$ $[10^{-2}M_{\odot}]$'
            else:
                return r'$M_{\rm{ej}}/(\eta M_{\rm{b;tot}})$ $[10^{-2}M_{\odot}]$'
            # else:
            #     raise NameError("label for v_n:{} mask:{} is not found".format(v_n, mask))

        elif v_n == 'ejmass3':
            return r'$M_{\rm{ej}}$ $[10^{-3}M_{\odot}]$'

        elif v_n == 'ejmass4':
            return r'$M_{\rm{ej}}$ $[10^{-4}M_{\odot}]$'

        elif v_n == "vel_inf":
            return r"$\upsilon_{\infty}$ [c]"

        elif v_n == "vel_inf_ave":
            return r"$<\upsilon_{\infty}>$ [c]"

        elif v_n == "Y_e" or v_n == "ye" or v_n == "Ye":
            return r"$Y_e$"

        elif v_n == "Lambda":
            return r"$\tilde{\Lambda}$"

        elif v_n == "Ye_ave":
            return r"$<Y_e>$"

        elif v_n == 'flux':
            return r'$\dot{M}$'

        elif v_n == 'time':
            return r'$t$ [ms]'

        elif v_n == 't-tmerg':
            return r'$t-t_{\rm{merg}}$ [ms]'

        elif v_n == "Y_final":
            return r'Relative final abundances'

        elif v_n == "A":
            return r"Mass number, A"

        elif v_n == 'entropy' or v_n == 's':
            return r'$s$'

        elif v_n == 't_eff' or v_n == 'T_eff':
            return r'$\log($T$_{eff}/K)$'

        elif v_n == "Mb":
            return r'$M_{\rm{b;tot}}$'

        else:
            return str(v_n).replace('_', '\_')
            # raise NameError("No label found for v_n:{}"
            #                 .format(v_n))