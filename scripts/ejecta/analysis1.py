import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import os
import sys
from scipy.interpolate import RegularGridInterpolator

class CONST:

    ns_rho = 1.6191004634e-5
    time_constant = 0.004925794970773136  # to to to ms
    energy_constant = 1787.5521500932314
    volume_constant = 2048

    fourpi = 12.5663706144
    oneoverpi = 0.31830988618
    c = 2.99792458e10  # [cm/s]
    c2 = 8.98755178737e20  # [cm^2/s^2]
    Msun = 1.98855e33  # [g]
    sec2day = 1.157407407e-5  # [day/s]
    day2sec = 86400.  # [sec/day]
    sigma_SB = 5.6704e-5  # [erg/cm^2/s/K^4]
    fourpisigma_SB = 7.125634793e-4  # [erg/cm^2/s/K^4]
    h = 6.6260755e-27  # [erg*s]
    kB = 1.380658e-16  # [erg/K]
    pc2cm = 3.085678e+18  # [cm/pc]
    sec2hour = 2.777778e-4  # [hr/s]
    day2hour = 24.  # [hr/day]
    small = 1.e-10  # [-]
    huge = 1.e+30  # [-]

class FORMULAS:

    @staticmethod
    def vinf(eninf):
        return np.sqrt(2. * eninf)

    @staticmethod
    def vinf_bern(eninf, enthalpy):
        return np.sqrt(2.*(enthalpy*(eninf + 1.) - 1.))

    @staticmethod
    def vel(w_lorentz):
        return np.sqrt(1. - 1. / (w_lorentz**2))

    @staticmethod
    def get_tau(rho, vel, radius, lrho_b):

        rho_b = 10 ** lrho_b
        tau_0 = 0.5 * 2.71828182845904523536 * (radius / vel) * (0.004925794970773136) # in ms
        tau_b = tau_0 * ((rho/rho_b) ** (1.0 / 3.0))
        return tau_b # ms

    @staticmethod
    def enthalpy(eps, press, rho):
        return 1 + eps + (press / rho)

class LOAD_OUTFLOW_SURFACE_H5:

    def __init__(self, fname, radius = None):

        # LOAD_ITTIME.__init__(self, sim, pprdir=pprdir)

        self.fname = fname
        self._radius = radius

        # self.list_detectors = [0, 1]

        # self.list_v_ns = ["fluxdens", "w_lorentz", "eninf", "surface_element",
        #                   "alp", "rho", "vel[0]", "vel[1]", "vel[2]", "Y_e",
        #                   "press", "entropy", "temperature", "eps"]
        self.list_v_ns = ['-u_t-1', 'Ye', 'epsl', 'fD', 'phi', 'press', 'rho', 's', 'surface_element', 'theta', 'time']

        self.list_grid_v_ns = ["theta", "phi", 'times']

        self.list_v_ns += self.list_grid_v_ns

        self.grid_pars = ["radius", "ntheta", "nphi"]

        self.matrix_data = [np.empty(0,) for v in range(len(self.list_v_ns)+len(self.list_grid_v_ns))]


        self.matrix_grid_pars = {}

    # def update_v_n(self, new_v_n=None):
    #     if new_v_n != None:
    #         if not new_v_n in self.list_v_ns:
    #             self.list_v_ns.append(v_n)
    #
    #             self.matrix_data = [[np.empty(0, )
    #                                  for v in range(len(self.list_v_ns) + len(self.list_grid_v_ns))]
    #                                 for d in range(len(self.list_detectors))]


    def check_v_n(self, v_n):
        if not v_n in self.list_v_ns:
            raise NameError("v_n:{} not in the list of v_ns: {}"
                            .format(v_n, self.list_v_ns))

    def i_v_n(self, v_n):
        return int(self.list_v_ns.index(v_n))

    def load_h5_file(self):

        assert os.path.isfile(self.fname)
        print("\tLoading {}".format(self.fname))
        dfile = h5py.File(self.fname, "r")

        # attributes
        if (not "radius" in dfile.attrs.keys()):
            radius = self._radius # cm
        else:
            radius = float(dfile.attrs["radius"])

        if (not "ntheta" in dfile.attrs.keys()):
            ntheta = len(np.array(dfile["theta"]))
        else:
            ntheta = int(dfile.attrs["ntheta"])

        if (not "nphi" in dfile.attrs.keys()):
            nphi = len(np.array(dfile["phi"]))
        else:
            nphi = int(dfile.attrs["nphi"])


        self.matrix_grid_pars["radius"] = radius
        self.matrix_grid_pars["ntheta"] = ntheta
        self.matrix_grid_pars["nphi"] = nphi

        for v_n in dfile:
            self.check_v_n(v_n)
            arr = np.array(dfile[v_n])
            self.matrix_data[self.i_v_n(v_n)] = arr

    def is_file_loaded(self):
        data = self.matrix_data[self.i_v_n(self.list_v_ns[0])]
        if len(data) == 0:
            self.load_h5_file()
        data = self.matrix_data[self.i_v_n(self.list_v_ns[0])]
        if len(data) == 0:
            raise ValueError("Error in loading/extracing data. Emtpy array")

    def get_full_arr(self, v_n):
        self.check_v_n(v_n)
        self.is_file_loaded()
        return self.matrix_data[self.i_v_n(v_n)]

    def get_grid_par(self, v_n):
        self.is_file_loaded()
        return self.matrix_grid_pars[v_n]

class COMPUTE_OUTFLOW_SURFACE_H5(LOAD_OUTFLOW_SURFACE_H5):

    def __init__(self, fname, radius = None):

        LOAD_OUTFLOW_SURFACE_H5.__init__(self, fname=fname, radius=radius)

        self.list_comp_v_ns = ["enthalpy", "vel_inf", "vel_inf_bern", "vel", "logrho", "eninf"]

        # self.list_v_ns = self.list_v_ns + self.list_comp_v_ns

        self.matrix_comp_data = [np.empty(0,) for v in self.list_comp_v_ns]

    def check_comp_v_n(self, v_n):
        if not v_n in self.list_comp_v_ns:
            raise NameError("v_n: {} is not in the list v_ns: {}"
                            .format(v_n, self.list_comp_v_ns))

    def i_comp_v_n(self, v_n):
        return int(self.list_comp_v_ns.index(v_n))

    # ----------------------------------

    def compute_arr(self, v_n):

        if v_n == "enthalpy":
            arr = FORMULAS.enthalpy(self.get_full_arr("eps"),
                                    self.get_full_arr("press"),
                                    self.get_full_arr("rho"))
        elif v_n == "eninf":
            arr = self.get_full_arr("-u_t-1") # THIS IS JUST "u_t"
            # arr = -1. * arr - 1.0
        elif v_n == "vel_inf":
            arr = FORMULAS.vinf(self.get_full_arr("eninf"))
        elif v_n == "vel_inf_bern":
            # print("----------------------------------------")
            arr = FORMULAS.vinf_bern(self.get_full_arr("eninf"),
                                     self.get_full_arr("enthalpy"))
        elif v_n == "vel":
            # arr = FORMULAS.vel(self.get_full_arr("w_lorentz"))
            arr = FORMULAS.vinf(self.get_full_arr("eninf"))
        elif v_n == "logrho":
            arr = np.log10(self.get_full_arr("rho"))
        else:
            raise NameError("No computation method for v_n:{} is found"
                            .format(v_n))
        return arr

    # ---------------------------------------

    def is_arr_computed(self, v_n):
        arr = self.matrix_comp_data[self.i_comp_v_n(v_n)]
        if len(arr) == 0:
            arr = self.compute_arr(v_n)
            self.matrix_comp_data[self.i_comp_v_n(v_n)] = arr
        if len(arr) == 0:
            raise ValueError("Computation of v_n:{} has failed. Array is emtpy"
                             .format(v_n))

    def get_full_comp_arr(self, v_n):
        self.check_comp_v_n(v_n)
        self.is_arr_computed(v_n)
        arr = self.matrix_comp_data[self.i_comp_v_n(v_n)]
        return arr

    def get_full_arr(self, v_n):
        if v_n in self.list_comp_v_ns:
            self.check_comp_v_n(v_n)
            self.is_arr_computed(v_n)
            arr = self.matrix_comp_data[self.i_comp_v_n(v_n)]
            return arr
        else:
            self.check_v_n(v_n)
            self.is_file_loaded()
            arr = self.matrix_data[self.i_v_n(v_n)]
            return arr

class ADD_MASK(COMPUTE_OUTFLOW_SURFACE_H5):

    def __init__(self, fname, add_mask=None, radius=None):

        COMPUTE_OUTFLOW_SURFACE_H5.__init__(self, fname=fname, radius=radius)

        self.list_masks = ["geo", "geo_v06",
                           "bern", "bern_geoend", "Y_e04_geoend", "theta60_geoend",
                           "geo_entropy_above_10", "geo_entropy_below_10"]
        if add_mask != None and not add_mask in self.list_masks:
            self.list_masks.append(add_mask)


        # "Y_e04_geoend"
        self.mask_matrix = [np.zeros(0,) for i in range(len(self.list_masks))]

        self.set_min_eninf = 0.
        self.set_min_enthalpy = 1.0022

    def check_mask(self, mask):
        if not mask in self.list_masks:
            raise NameError("mask: {} is not in the list: {}"
                            .format(mask, self.list_masks))

    def i_mask(self, mask):
        return int(self.list_masks.index(mask))

    # ----------------------------------------------
    def __time_mask_end_geo(self, length=0.):

        fluxdens = self.get_full_arr("fD")#("fluxdens")
        da = self.get_full_arr("surface_element") # "surface_element"
        t = self.get_full_arr("time")
        dt = np.diff(t)
        dt = np.insert(dt, 0, 0)
        mask = self.get_mask("geo").astype(int)
        fluxdens = fluxdens * mask
        flux_arr = np.sum(np.sum(fluxdens * da, axis=1), axis=1)  # sum over theta and phi
        tot_mass = np.cumsum(flux_arr * dt)  # sum over time
        tot_flux = np.cumsum(flux_arr)  # sum over time
        # print("totmass:{}".format(tot_mass[-1]))
        fraction = 0.98
        i_t98mass = int(np.where(tot_mass >= fraction * tot_mass[-1])[0][0])
        # print(i_t98mass)
        # assert i_t98mass < len(t)

        if length > 0.:
            if length > t[-1]:
                raise ValueError("length:{} is > t[-1]:{} [ms]".format(length,#*Constants.time_constant,
                                                                     t[-1]#*Constants.time_constant
                                                                       ))
            if t[i_t98mass] + length > t[-1]:
                # because of bloody numerics it can > but just by a tiny bit. So I added this shit.
                if np.abs(t[i_t98mass] - length > t[-1]) < 10: # 10 is a rundomly chosen number
                    length = length - 10
                else:
                    raise ValueError("t[i_t98mass] + length > t[-1] : {} > {}"
                                     .format((t[i_t98mass] + length),
                                             t[-1]))

            i_mask = (t > t[i_t98mass]) & (t < t[i_t98mass] + length)
        else:
            i_mask = t > t[i_t98mass]
        # saving time at 98% mass for future use
        # fpath = Paths.ppr_sims + self.sim + '/outflow_{}/t98mass.dat'.format(det)
        # try: open(fpath, "w").write("{}\n".format(float(t[i_t98mass])))
        # except IOError: Printcolor.yellow("\tFailed to save t98mass.dat")
        # continuing with mask
        newmask = np.zeros(fluxdens.shape)
        for i in range(len(newmask[:, 0, 0])):
            newmask[i, :, :].fill(i_mask[i])
        return newmask.astype(bool)
    # ----------------------------------------------

    def compute_mask(self, mask):
        self.check_mask(mask)

        if mask == "geo":
            # 1 - if geodeisc is true
            einf = self.get_full_arr("eninf")
            res = (einf >= self.set_min_eninf)
            return res
        if mask == "geo_v06":
            # 1 - if geodeisc is true
            einf = self.get_full_arr("eninf")
            vinf = self.get_full_arr("vel_inf")
            res = (einf >= self.set_min_eninf) & (vinf >= 0.6)
            return res
        elif mask == "geo_entropy_below_10":
            einf = self.get_full_arr("eninf")
            res = (einf >= self.set_min_eninf)
            entropy = self.get_full_arr("entropy")
            mask_entropy = entropy < 10.
            return res & mask_entropy
        elif mask == "geo_entropy_above_10":
            einf = self.get_full_arr("eninf")
            res = (einf >= self.set_min_eninf)
            entropy = self.get_full_arr("entropy")
            mask_entropy = entropy > 10.
            return res & mask_entropy
        elif mask == "bern":
            # 1 - if Bernulli is true
            enthalpy = self.get_full_comp_arr("enthalpy")
            einf = self.get_full_arr("eninf")
            res = ((enthalpy * (einf + 1) - 1) > self.set_min_eninf) & (enthalpy >= self.set_min_enthalpy)
        elif mask == "bern_geoend":
            # 1 - data above 98% of GeoMass and if Bernoulli true and 0 if not
            mask2 = self.get_mask("bern")
            newmask = self.__time_mask_end_geo()

            res = newmask & mask2
        elif mask == "Y_e04_geoend":
            # 1 - data above Ye=0.4 and 0 - below
            ye = self.get_full_arr("Y_e")
            mask_ye = ye >= 0.4
            mask_bern = self.get_mask("bern")
            mask_geo_end = self.__time_mask_end_geo()
            return mask_ye & mask_bern & mask_geo_end
        elif mask == "theta60_geoend":
            # 1 - data above Ye=0.4 and 0 - below
            theta = self.get_full_arr("theta")
            # print((theta / np.pi * 180.).min(), (theta / np.pi * 180.).max())
            # exit(1)
            theta_ = 90 - (theta * 180 / np.pi)
            # print(theta_); #exit(1)
            theta_mask = (theta_ > 60.) | (theta_ < -60)
            # print(np.sum(theta_mask.astype(int)))
            # assert np.sum(theta_mask.astype(int)) > 0
            newmask = theta_mask[np.newaxis, : , :]

            # fluxdens = self.get_full_arr(det, "fluxdens")
            # newmask = np.zeros(fluxdens.shape)
            # for i in range(len(newmask[:, 0, 0])):
            #     newmask[i, :, :].fill(theta_mask)

            # print(newmask.shape)
            # exit(1)
            # mask_ye = ye >= 0.4
            mask_bern = self.get_mask("bern")
            # print(mask_bern.shape)
            # print(mask_bern.shape)
            mask_geo_end = self.__time_mask_end_geo()
            return newmask & mask_bern & mask_geo_end
        elif str(mask).__contains__("_tmax"):
            raise NameError(" mask with '_tmax' are not supported")
            #
            # # 1 - data below tmax and 0 - above
            # base_mask_name = str(str(mask).split("_tmax")[0])
            # base_mask = self.get_mask(det, base_mask_name)
            # #
            # tmax = float(str(mask).split("_tmax")[-1])
            # tmax = tmax / Constants.time_constant # Msun
            # # tmax loaded is postmerger tmax. Thus it need to be added to merger time
            # fpath = self.pprdir+"/waveforms/tmerger.dat"
            # try:
            #     tmerg = float(np.loadtxt(fpath, unpack=True)) # Msun
            #     Printcolor.yellow("\tWarning! using defauled M_Inf=2.748, R_GW=400.0 for retardet time")
            #     ret_time = PHYSICS.get_retarded_time(tmerg, M_Inf=2.748, R_GW=400.0)
            #     tmerg = ret_time
            #     # tmerg = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            #     # tmerg = tmerg / (Constants.time_constant *1e-3)
            # except IOError:
            #     raise IOError("For the {} mask, the tmerger.dat is needed at {}"
            #                   .format(mask, fpath))
            # except:
            #     raise ValueError("failed to extract tmerg for outflow tmax mask analysis")
            #
            # t = self.get_full_arr(det, "times") # Msun
            # # tmax = tmax + tmerg       # Now tmax is absolute time (from the begniing ofthe simulation
            # print("t[-1]:{} tmax:{} tmerg:{} -> {}".format(t[-1]*Constants.time_constant*1e-3,
            #                                 tmax*Constants.time_constant*1e-3,
            #                                 tmerg*Constants.time_constant*1e-3,
            #                                 (tmax+tmerg)*Constants.time_constant*1e-3))
            # tmax = tmax + tmerg
            # if tmax > t[-1]:
            #     raise ValueError("tmax:{} for the mask is > t[-1]:{}".format(tmax*Constants.time_constant*1e-3,
            #                                                                  t[-1]*Constants.time_constant*1e-3))
            # if tmax < t[0]:
            #     raise ValueError("tmax:{} for the mask is < t[0]:{}".format(tmax * Constants.time_constant * 1e-3,
            #                                                                  t[0] * Constants.time_constant * 1e-3))
            # fluxdens = self.get_full_arr(det, "fluxdens")
            # i_mask = t < t[UTILS.find_nearest_index(t, tmax)]
            # newmask = np.zeros(fluxdens.shape)
            # for i in range(len(newmask[:, 0, 0])):
            #     newmask[i, :, :].fill(i_mask[i])

            # print(base_mask.shape,newmask.shape)

            # return base_mask & newmask.astype(bool)
        elif str(mask).__contains__("_length"):
            base_mask_name = str(str(mask).split("_length")[0])
            base_mask = self.get_mask(base_mask_name)
            delta_t = float(str(mask).split("_length")[-1])
            delta_t = (delta_t / 1e5) / (1e-3) # Msun
            t = self.get_full_arr("times")  # Msun
            print("\t t[0]: {}\n\t t[-1]: {}\n\t delta_t: {}\n\t mask: {}"
                  .format(t[0] * 1e-3,
                          t[-1] * 1e-3,
                          delta_t *  1e-3,
                          mask))
            assert delta_t < t[-1]
            assert delta_t > t[0]
            mask2 = self.get_mask("bern")
            newmask = self.__time_mask_end_geo(length=delta_t)

            res = newmask & mask2

        else:
            raise NameError("No method found for computing mask:{}"
                            .format(mask))

        return res

    # ----------------------------------------------

    def is_mask_computed(self, mask):
        if len(self.mask_matrix[self.i_mask(mask)]) == 0:
            arr = self.compute_mask(mask)
            self.mask_matrix[self.i_mask(mask)] = arr

        if len(self.mask_matrix[self.i_mask(mask)]) == 0:
            raise ValueError("Failed to compute the mask: {}".format(mask))

    def get_mask(self, mask):
        self.check_mask(mask)
        self.is_mask_computed(mask)
        return self.mask_matrix[self.i_mask(mask)]

class EJECTA(ADD_MASK):

    def __init__(self, fname, skynetdir, radius=None, add_mask=None):

        ADD_MASK.__init__(self, fname=fname, add_mask=add_mask, radius=radius)

        self.list_hist_v_ns = ["Y_e", "theta", "phi", "vel_inf", "entropy", "temperature", "logrho"]

        self.list_corr_v_ns = ["Y_e theta", "vel_inf theta", "Y_e vel_inf",
                               "logrho vel_inf", "logrho theta", "logrho Y_e"]

        self.list_ejecta_v_ns = [
                                    "tot_mass", "tot_flux",  "weights", "corr3d Y_e entropy tau",
                                ] +\
                                ["timecorr {}".format(v_n) for v_n in self.list_hist_v_ns] +\
                                ["hist {}".format(v_n) for v_n in self.list_hist_v_ns] +\
                                ["corr2d {}".format(v_n) for v_n in self.list_corr_v_ns] +\
                                ["mass_ave {}".format(v_n) for v_n in self.list_v_ns]

        self.matrix_ejecta = [[np.zeros(0,)
                                for k in range(len(self.list_ejecta_v_ns))]
                                for j in range(len(self.list_masks))]

        self.set_skynet_densmap_fpath = skynetdir / "densmap.h5"
        self.set_skyent_grid_fpath = skynetdir / "grid.h5"

    # ---

    def check_ej_v_n(self, v_n):
        if not v_n in self.list_ejecta_v_ns:
            raise NameError("module_ejecta v_n: {} is not in the list of module_ejecta v_ns {}"
                            .format(v_n, self.list_ejecta_v_ns))

    def i_ejv_n(self, v_n):
        return int(self.list_ejecta_v_ns.index(v_n))

    # --- methods for EJECTA arrays ----

    def get_cumulative_ejected_mass(self, mask):
        fluxdens = self.get_full_arr("fD")  # "fluxdens"
        surface_element = self.get_full_arr("surface_element")
        theta = self.get_full_arr("theta")
        dtheta = np.diff(theta)
        phi = self.get_full_arr("phi")
        dphi = np.diff(phi)
        t = self.get_full_arr("time")
        dt = np.diff(t)

        mask = (self.get_full_arr("-u_t-1") > 0).astype(int)#self.get_mask(mask).astype(int)

        res_t_phi_theta = fluxdens*mask*surface_element
        res_t_theta = np.trapz(res_t_phi_theta, dx=dphi[0] ,axis=2)
        flux_arr = np.trapz(res_t_theta, dx=dtheta[0], axis=1)

        # flux_arr = np.sum(np.sum(fluxdens * surface_element * dtheta[0] * dphi[0], axis=1), axis=1)

        # flux_arr = res_t

        # tot_mass = np.cumsum(flux_arr * dt[0])
        tot_mass = np.trapz(flux_arr, dx=dt[0], axis=0)/CONST.Msun
        print(tot_mass)
        # fluxdens = self.get_full_arr("fD") # "fluxdens"
        # da = self.get_full_arr("surface_element")
        # t = self.get_full_arr("time") / 1.e3 # [s]
        # dt = np.diff(t)
        # dt = np.insert(dt, 0, 0)
        # mask = self.get_mask(mask).astype(int)
        # fluxdens = fluxdens * mask
        # flux_arr = np.sum(np.sum(fluxdens * da, axis=1), axis=1)  # sum over theta and phi
        # tot_mass = np.cumsum(flux_arr * dt)  # sum over time
        # tot_flux = np.cumsum(flux_arr)  # sum over time
        # print("totmass:{}".format(tot_mass[-1]))
        return (t, flux_arr, tot_mass) # time in [s] 0.004925794970773136

    def get_weights(self, mask):

        dt = np.diff(self.get_full_arr("times"))
        dt = np.insert(dt, 0, 0)
        mask_arr = self.get_mask(mask).astype(int)
        weights = mask_arr * \
                  self.get_full_arr("fluxdens") * \
                  self.get_full_arr("surface_element") * \
                  dt[:, np.newaxis, np.newaxis]
        #
        if np.sum(weights) == 0.:
            _, _, mass = self.get_cumulative_ejected_mass(mask)
            print("Error. sum(weights) = 0. For mask:{} there is not mass (Total ej.mass is {})".format(mask,mass[-1]))
            raise ValueError("sum(weights) = 0. For mask:{} there is not mass (Total ej.mass is {})".format(mask,mass[-1]))
        #
        return weights

    def get_hist(self, mask, v_n, edge):

        times = self.get_full_arr("times")
        weights = np.array(self.get_ejecta_arr(mask, "weights"))
        data = np.array(self.get_full_arr(v_n))
        # if v_n == "rho":
        #     data = np.log10(data)
        historgram = np.zeros(len(edge) - 1)
        # tmp2 = []
        # print(data.shape, weights.shape, edge.shape)
        for i in range(len(times)):
            if np.array(data).ndim == 3: data_ = data[i, :, :].flatten()
            else: data_ = data.flatten()
            # print(data.min(), data.max())
            tmp, _ = np.histogram(data_, bins=edge, weights=weights[i, :, :].flatten())
            historgram += tmp
        middles = 0.5 * (edge[1:] + edge[:-1])
        assert len(historgram) == len(middles)
        if np.sum(historgram) == 0.:
            print("Error. Histogram weights.sum() = 0 ")
            raise ValueError("Error. Histogram weights.sum() = 0 ")
        return middles, historgram

        # res = np.vstack((middles, historgram))
        # return res

    def get_timecorr(self, mask, v_n, edge):

        historgrams = np.zeros(len(edge) - 1)

        times = self.get_full_arr("times")
        timeedges = np.linspace(times.min(), times.max(), 55)

        indexes = []
        for i_e, t_e in enumerate(timeedges[:-1]):
            i_indx = []
            for i, t in enumerate(times):
                if (t >= timeedges[i_e]) and (t<timeedges[i_e+1]):
                    i_indx = np.append(i_indx, int(i))
            indexes.append(i_indx)
        assert len(indexes) > 0
        #
        weights = np.array(self.get_ejecta_arr(mask, "weights"))
        data = np.array(self.get_full_arr(v_n))

        for i_ind, ind_list in enumerate(indexes):
            # print("{} {}/{}".format(i_ind,len(indexes), len(ind_list)))
            historgram = np.zeros(len(edge) - 1)
            for i in np.array(ind_list, dtype=int):
                if np.array(data).ndim == 3: data_ = data[i, :, :].flatten()
                else: data_ = data.flatten()
                tmp, _ = np.histogram(data_, bins=edge, weights=weights[i, :, :].flatten())
                historgram += tmp
            historgrams = np.vstack((historgrams, historgram))

        # print("done") #1min15
        # exit(1)

        bins = 0.5 * (edge[1:] + edge[:-1])

        # print("hist", historgrams.shape)
        # print("times", timeedges.shape)
        # print("edges", bins.shape)

        return bins, timeedges, historgrams

    @staticmethod
    def combine(x, y, xy, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        # work around the fact that for corr we want to save 'edges' not bins that exceed the size of xy data by 1
        if ( len(x) == 1 + len(xy[0,:]) and len(y) == 1 + len(xy[:,0]) ):
            # print('ping')
            _xy = np.zeros((len(y),len(x)))
            print(y.shape, x.shape, xy.shape)
            for iy in range(len(y)-1):
                for jx in range(len(x)-1):
                    _xy[iy,jx] = xy[iy,jx]
            xy = _xy

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        if corner_val != None:
            res[0, 0] = corner_val

        return res



    @staticmethod
    def combine3d(x, y, z, xyz, corner_val=None):
        '''creates a 2d array  1st raw    [0, 1:] -- x -- density     (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''

        tmp = np.zeros((len(xyz[:, 0, 0])+1, len(xyz[0, :, 0])+1, len(xyz[0, 0, :])+1))
        tmp[1:, 1:, 1:] = xyz
        tmp[1:, 0, 0] = x
        tmp[0, 1:, 0] = y
        tmp[0, 0, 1:] = z
        return tmp

    def get_corr2d(self, mask, v_n1, v_n2, edge1, edge2):

        tuple_edges = tuple([edge1, edge2])
        correlation = np.zeros([len(edge) - 1 for edge in tuple_edges])
        times = self.get_full_arr("times")
        weights = self.get_ejecta_arr(mask, "weights")
        for i_t, t in enumerate(times):

            data1 = self.get_full_arr(v_n1)
            if data1.ndim == 3: data1_ = data1[i_t, :, :].flatten()
            else: data1_ = data1.flatten()

            data2 = self.get_full_arr(v_n2)
            if data2.ndim == 3: data2_ = data2[i_t, :, :].flatten()
            else: data2_ = data2.flatten()

            data = tuple([data1_, data2_])
            tmp, _ = np.histogramdd(data, bins=tuple_edges, weights=weights[i_t, :, :].flatten())
            correlation += tmp

        bins1 = 0.5 * (edge1[1:] + edge1[:-1])
        bins2 = 0.5 * (edge2[1:] + edge2[:-1])

        return (bins1, bins2, correlation.T)

    def get_corr3d(self, mask, v_n1, v_n2, v_n3, edge1, edge2, edge3):

        tuple_edges = tuple([edge1, edge2, edge3])
        correlation = np.zeros([len(edge) - 1 for edge in tuple_edges])
        times = self.get_full_arr("times")
        weights = self.get_ejecta_arr(mask, "weights")
        for i_t, t in enumerate(times):

            data1 = self.get_full_arr(v_n1)
            if data1.ndim == 3: data1_ = data1[i_t, :, :].flatten()
            else: data1_ = data1.flatten()
            #
            data2 = self.get_full_arr(v_n2)
            if data2.ndim == 3: data2_ = data2[i_t, :, :].flatten()
            else: data2_ = data2.flatten()
            #
            data3 = self.get_full_arr(v_n3)
            if data2.ndim == 3: data3_ = data3[i_t, :, :].flatten()
            else: data3_ = data3.flatten()
            #
            data = tuple([data1_, data2_, data3_])
            tmp, _ = np.histogramdd(data, bins=tuple_edges, weights=weights[i_t, :, :].flatten())
            correlation += tmp

        bins1 = 0.5 * (edge1[1:] + edge1[:-1])
        bins2 = 0.5 * (edge2[1:] + edge2[:-1])
        bins3 = 0.5 * (edge3[1:] + edge3[:-1])

        return bins1, bins2, bins3, correlation.T

    @staticmethod
    def get_edges_from_centers(bins):

        edges = np.array(0.5 * (bins[1:] + bins[:-1]))  # from edges to center of bins
        edges = np.insert(edges, 0, edges[0] - np.diff(edges)[0])
        edges = np.append(edges, edges[-1] + np.diff(edges)[-1])

        return edges

    def get_corr_ye_entr_tau(self, mask):

        densmap = h5py.File(self.set_skynet_densmap_fpath, "r")

        dmap_ye = np.array(densmap["Ye"])
        dmap_rho = np.log10(np.array(densmap["density"]))
        dmap_entr = np.array(densmap["entropy"])

        interpolator = RegularGridInterpolator((dmap_ye, dmap_entr), dmap_rho,
                                               method="linear", bounds_error=False)

        grid = h5py.File(self.set_skyent_grid_fpath,"r")
        grid_ye = np.array(grid["Ye"])
        grid_entr = np.array(grid["entropy"])
        grid_tau = np.array(grid["tau"])

        data_ye = self.get_full_arr("Ye") # Y_e
        data_entr = self.get_full_arr("s") # entropy
        data_rho = self.get_full_arr("rho") #* 6.173937319029555e+17 # CGS
        data_vel = self.get_full_comp_arr("vel")

        lrho_b = [[np.zeros(len(data_ye[:, 0, 0]))
                   for i in range(len(data_ye[0, :, 0]))]
                  for j in range(len(data_ye[0, 0, :]))]
        for i_theta in range(len(data_ye[0, :, 0])):
            for i_phi in range(len(data_ye[0, 0, :])):
                data_ye_i = data_ye[:, i_theta, i_phi].flatten()
                data_entr_i = data_entr[:, i_theta, i_phi].flatten()

                data_ye_i[data_ye_i > grid_ye.max()] = grid_ye.max()
                data_entr_i[data_entr_i > grid_entr.max()] = grid_entr.max()
                data_ye_i[data_ye_i < grid_ye.min()] = grid_ye.min()
                data_entr_i[data_entr_i < grid_entr.min()] = grid_entr.min()

                A = np.zeros((len(data_ye_i), 2))
                A[:, 0] = data_ye_i
                A[:, 1] = data_entr_i
                lrho_b_i = interpolator(A)

                lrho_b[i_phi][i_theta] = lrho_b_i
                sys.stdout.flush()

        # from d3analysis import FORMULAS
        lrho_b = np.array(lrho_b, dtype=np.float).T
        data_tau = FORMULAS.get_tau(data_rho, data_vel, self.get_grid_par("radius"), lrho_b)

        weights = self.get_ejecta_arr(mask, "weights")
        edges_ye = self.get_edges_from_centers(grid_ye)
        edges_tau = self.get_edges_from_centers(grid_tau)
        edges_entr = self.get_edges_from_centers(grid_entr)
        edges = tuple([edges_ye, edges_entr, edges_tau])

        correlation = np.zeros([len(edge) - 1 for edge in edges])

        for i in range(len(weights[:, 0, 0])):
            data_ye_i = data_ye[i, :, :]
            data_entr_i = data_entr[i, : ,:]
            data_tau_i = data_tau[i, :, :]
            data = tuple([data_ye_i.flatten(), data_entr_i.flatten(), data_tau_i.flatten()])
            tmp, _ = np.histogramdd(data, bins=edges, weights=weights[i, :, :].flatten())
            correlation += tmp

        bins_ye = 0.5 * (edges_ye[1:] + edges_ye[:-1])
        bins_entr = 0.5 * (edges_entr[1:] + edges_entr[:-1])
        bins_tau = 0.5 * (edges_tau[1:] + edges_tau[:-1])

        if not (np.sum(correlation) > 0):
            print("Error. np.sum(correlation) = 0")
            raise ValueError("np.sum(correlation) = 0")

        if not (np.sum(correlation) <= np.sum(weights)):
            print("Error np.sum(correlation) > np.sum(weights)")
            raise ValueError("np.sum(correlation) <= np.sum(weights)")

        assert correlation.shape == (17, 17, 17)

        return bins_ye, bins_entr, bins_tau, correlation

    def get_mass_averaged(self, mask, v_n):

        dt = np.diff(self.get_full_arr("times"))
        dt = np.insert(dt, 0, 0)
        dt = dt[:, np.newaxis, np.newaxis]
        data = self.get_full_arr(v_n)

        mask = self.get_mask(mask)
        flux = self.get_full_arr("fluxdens") * mask
        mass_averages = np.zeros((len(flux[0, :, 0]), len(flux[0, 0, :])))
        total_flux = np.zeros((len(flux[0, :, 0]), len(flux[0, 0, :])))

        for i in range(len(flux[:, 0, 0])):

            total_flux += flux[i, :, :] * dt[i, :, :]

            if v_n == "fluxdens":
                mass_averages += flux[i, :, :] * dt[i, :, :]
            else:
                mass_averages += data[i, :, :] * flux[i, :, :] * dt[i, :, :]

        return np.array(mass_averages / total_flux)

    # ---------------------------------

    def compute_ejecta_arr(self, mask, v_n):

        # as Bernoulli criteria uses different vel_inf defention
        if v_n.__contains__("vel_inf") and mask.__contains__("bern"):
            v_n = v_n.replace("vel_inf", "vel_inf_bern")


        # ----------------------------------------
        if v_n in ["tot_mass", "tot_flux"]:
            t, flux, mass = self.get_cumulative_ejected_mass(mask)
            arr = np.vstack((t, flux, mass)).T

        elif v_n == "weights":
            arr = self.get_weights(mask)

        elif v_n.__contains__("hist "):
            v_n = str(v_n.split("hist ")[-1])
            edge = get_hist_bins_ej(v_n)
            middles, historgram = self.get_hist(mask, v_n, edge)
            arr = np.vstack((middles, historgram)).T

        elif v_n.__contains__("corr2d "):
            v_n1 = str(v_n.split(" ")[1])
            v_n2 = str(v_n.split(" ")[2])
            edge1 = get_hist_bins_ej(v_n1)
            edge2 = get_hist_bins_ej(v_n2)
            bins1, bins2, weights = self.get_corr2d(mask, v_n1, v_n2, edge1, edge2)
            # arr = self.combine(bins1, bins2, weights) # y_arr [1:, 0] x_arr [0, 1:]
            arr = self.combine(edge1, edge2, weights) # y_arr [1:, 0] x_arr [0, 1:]

        elif v_n.__contains__("timecorr "):
            v_n = str(v_n.split(" ")[1])
            edge = get_hist_bins_ej(v_n)
            bins, binstime, weights = self.get_timecorr(mask, v_n, edge)
            return self.combine(binstime, bins, weights.T)

        elif v_n == "corr3d Y_e entropy tau":
            bins1, bins2, bins3, corr = self.get_corr_ye_entr_tau(mask)
            arr = self.combine3d(bins1, bins2, bins3, corr)

        elif v_n.__contains__("mass_ave "):
            v_n = str(v_n.split("mass_ave ")[-1])
            # print(v_n)
            arr = self.get_mass_averaged(mask, v_n)

        else:
            raise NameError("no method found for computing module_ejecta arr for mask:{} v_n:{}"
                            .format(mask, v_n))


        return arr

    # ---------------------------------

    def is_ejecta_arr_computed(self, mask, v_n):
        data = self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)]
        if len(data) == 0:
            arr = self.compute_ejecta_arr(mask, v_n)
            self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)] = arr

        data = self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)]
        if len(data) == 0:
            raise ValueError("Failed to compute module_ejecta array for "
                             "mask:{} v_n:{}"
                             .format(mask, v_n))

    def get_ejecta_arr(self, mask, v_n):
        self.check_mask(mask)
        self.check_ej_v_n(v_n)
        self.is_ejecta_arr_computed(mask, v_n)
        data = self.matrix_ejecta[self.i_mask(mask)][self.i_ejv_n(v_n)]
        return data

def main():

    # o_dfile = COMPUTE_OUTFLOW_SURFACE_H5(fname=Path("../../data/large/skynet_input.h5"), radius=450. * 1000 * 100)
    # print( len( o_dfile.get_full_comp_arr("einf")[o_dfile.get_full_comp_arr("einf") > 0] ) )

    o_ej = EJECTA(fname=Path("../../data/large/skynet_input.h5"), skynetdir=Path("../../data/skynet"), radius=450. * 1000 * 100)
    arr = o_ej.get_full_arr("eninf")
    print(arr.min(), arr.max())
    print(len(np.array(arr[arr>0]).flatten()))

    time, mass_flux, mass = o_ej.get_cumulative_ejected_mass("geo")
    print(mass[-1]/CONST.Msun)
    plt.semilogy(time, mass/CONST.Msun)
    plt.show()

    #
    # # find precompute SkyNet grid
    # skynet_path = Path("../../data/skynet")
    # densmap_dfile = h5py.File(skynet_path / "densmap.h5", "r")
    # dmap_ye = np.array(densmap_dfile["Ye"])
    # dmap_rho = np.log10(np.array(densmap_dfile["density"]))
    # dmap_entr = np.array(densmap_dfile["entropy"])
    # interpolator = RegularGridInterpolator((dmap_ye, dmap_entr), dmap_rho,
    #                                        method="linear", bounds_error=False)
    #
    #
    # grid_dfile = h5py.File(skynet_path / "grid.h5", "r")
    # grid_ye = np.array(grid_dfile["Ye"])
    # grid_entr = np.array(grid_dfile["entropy"])
    # grid_tau = np.array(grid_dfile["tau"])
    #
    # # find data
    # fpath = Path("../../data/large/skynet_input.h5")
    # radius = 500. * 1000. * 100. # cm
    # dfile = h5py.File(fpath)
    # print(dfile.keys())
    # data_ye = np.array(dfile["Ye"])
    # data_entr = np.array(dfile["s"])
    # data_rho = np.array(dfile["rho"])# * 6.173937319029555e+17  # CGS
    # data_vel = np.array(dfile["v"])
    # times = np.array(dfile["time"])
    # theta = np.array(dfile["theta"])
    # phi = np.array(dfile["phi"])
    # print("Format={} time={}, theta={} phi={}".format(data_ye.shape,times.shape, theta.shape, phi.shape))
    #
    # lrho_b = [[np.zeros(len(data_ye[:, 0, 0]))
    #            for i in range(len(data_ye[0, :, 0]))]
    #           for j in range(len(data_ye[0, 0, :]))]
    # for i_theta in range(len(data_ye[0, :, 0])):
    #     for i_phi in range(len(data_ye[0, 0, :])):
    #         data_ye_i = data_ye[:, i_theta, i_phi].flatten()
    #         data_entr_i = data_entr[:, i_theta, i_phi].flatten()
    #
    #         data_ye_i[data_ye_i > grid_ye.max()] = grid_ye.max()
    #         data_entr_i[data_entr_i > grid_entr.max()] = grid_entr.max()
    #         data_ye_i[data_ye_i < grid_ye.min()] = grid_ye.min()
    #         data_entr_i[data_entr_i < grid_entr.min()] = grid_entr.min()
    #
    #         A = np.zeros((len(data_ye_i), 2))
    #         A[:, 0] = data_ye_i
    #         A[:, 1] = data_entr_i
    #         lrho_b_i = interpolator(A)
    #
    #         lrho_b[i_phi][i_theta] = lrho_b_i
    #         sys.stdout.flush()
    #
    # # from d3analysis import FORMULAS
    # lrho_b = np.array(lrho_b, dtype=np.float).T
    # data_tau = FORMULAS.get_tau(data_rho, data_vel, self.get_grid_par("radius"), lrho_b)
    #
    # weights = self.get_ejecta_arr(mask, "weights")
    # edges_ye = self.get_edges_from_centers(grid_ye)
    # edges_tau = self.get_edges_from_centers(grid_tau)
    # edges_entr = self.get_edges_from_centers(grid_entr)
    # edges = tuple([edges_ye, edges_entr, edges_tau])
    #
    # correlation = np.zeros([len(edge) - 1 for edge in edges])
    #
    # for i in range(len(weights[:, 0, 0])):
    #     data_ye_i = data_ye[i, :, :]
    #     data_entr_i = data_entr[i, :, :]
    #     data_tau_i = data_tau[i, :, :]
    #     data = tuple([data_ye_i.flatten(), data_entr_i.flatten(), data_tau_i.flatten()])
    #     tmp, _ = np.histogramdd(data, bins=edges, weights=weights[i, :, :].flatten())
    #     correlation += tmp
    #
    # bins_ye = 0.5 * (edges_ye[1:] + edges_ye[:-1])
    # bins_entr = 0.5 * (edges_entr[1:] + edges_entr[:-1])
    # bins_tau = 0.5 * (edges_tau[1:] + edges_tau[:-1])

if __name__ == '__main__':
    main()