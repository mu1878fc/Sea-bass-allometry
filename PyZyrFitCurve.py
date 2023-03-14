import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as sst
from sklearn.metrics import r2_score, mean_squared_error

__author__ = "Luiz Claudio Navarro"
__version__ = "0.0.12"
__date__ = "2022.10.17"
libgroup = "Zyryus Projects Infrastructure Libraries."
libname = "Fitting curve functions."
libversion = "Version: " + __version__ + " Date: " + __date__ + "."
copyrightmsg = "Zyryus Consulting (c) 2019-2021."
__doc__ = libgroup + " " + libname + "\n" + libversion + " " + copyrightmsg + "\n"

##############################################################################################
# Constants and Global variables
##############################################################################################
C_N_ROUNDS = 10
C_N_PARTITIONS = 10
C_VALID_PART = [(0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (3, 4, 5),
                (4, 5, 6),
                (5, 6, 7),
                (6, 7, 8),
                (7, 8, 9),
                (8, 9, 0),
                (9, 0, 1)]

C_SAMPLE_FIT = 1
C_SAMPLE_VALID = 2

# maximum value of p
C_ALPHA_P = 0.05


##############################################################################################
# Utility Functions
##############################################################################################
def compute_resul_stats(yreal, ypred, coeffs, cstd, nvar, itcpt):
    nobsv = len(yreal)
    ncoeff = len(coeffs)
    pord = int((ncoeff - int(itcpt)) / nvar)
    assert nobsv == len(ypred), "Compute error: Invalid data arguments length!"
    assert ncoeff == len(cstd) and \
           (nvar * pord) <= ncoeff <= \
           ((nvar * pord) + 1), "Compute error: Invalid coeff arguments length!"

    # regression stats
    ynozer = np.copy(yreal)
    selzr = np.where(ynozer == 0.0, True, False)
    ynozer[selzr] = 1.0e-12
    r2 = r2_score(ynozer, ypred)
    mse = mean_squared_error(yreal, ypred, squared=True)
    rmse = mean_squared_error(yreal, ypred, squared=False)
    nrmse = rmse / np.mean(yreal)
    ymean = np.mean(yreal)
    ystd = np.std(yreal)
    ypredmean = np.mean(ypred)
    ypredstd = np.std(ypred)

    # ANOVA stats
    dftot = nobsv - 1
    dfregr = len(coeffs) - 1
    if dfregr == 0:
        # if one coefficient per variable and no intercept,
        # consider intercept 0 as coefficient
        dfregr = 1
    dfresd = dftot - dfregr
    if dfresd == 0:
        dfresd = 1
    regrcomp = ypred - ymean
    resdcomp = yreal - ypred
    totcomp = yreal - ymean
    sstot = np.sum(totcomp ** 2)
    ssregr = np.sum(regrcomp ** 2)
    ssresd = np.sum(resdcomp ** 2)
    msregr = ssregr / dfregr
    msresd = ssresd / dfresd
    adjr2 = 1.0 - (msresd / (ystd ** 2))
    if msresd != 0.0:
        fstat = msregr / msresd
        pval = 1.0 - sst.f.cdf(fstat, dfregr, dfresd, loc=0, scale=1)
    else:
        fstat = msregr / 1e-12
        pval = 0.0
    # Coefficient stats
    cstd[cstd == 0] = 1e-12
    tcoeff = coeffs / cstd
    pcoeff = 2.0 * sst.t.sf(np.abs(tcoeff), dfresd, loc=0, scale=1)
    if pval > C_ALPHA_P:
        neffcoeff = 0
    else:
        neffcoeff = np.count_nonzero(pcoeff <= C_ALPHA_P)

    resul_dict = {
        "coefficients": {
            "n_variables": nvar,
            "n_coefficients": ncoeff,
            "intercept": int(itcpt),
            "n_effect_coeff": neffcoeff
        },
        "real": {
            "mean": ymean,
            "std": ystd
        },
        "regression": {
            "mean": ypredmean,
            "std": ypredstd,
            "R2": r2,
            "adjusted_R2": adjr2,
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "observations": nobsv
        },
        "ANOVA": {
            "regression": {
                "df": dfregr,
                "SS": ssregr,
                "MS": msregr,
                "F": fstat,
                "p": pval
            },
            "residual": {
                "df": dfresd,
                "SS": ssresd,
                "MS": msresd,
            },
            "total": {
                "df": dftot,
                "SS": sstot,
            }
        }
    }
    k = 0
    for i in range(nvar):
        vardict = {}
        for j in range(pord):
            vardict["coeff_{:03d}".format(k)] = coeffs[k]
            vardict["coeff_std_{:03d}".format(k)] = cstd[k]
            vardict["coeff_tstat_{:03d}".format(k)] = tcoeff[k]
            vardict["coeff_p_{:03d}".format(k)] = pcoeff[k]
            k += 1
        resul_dict["coefficients"]["var_{:03d}".format(i)] = vardict
    if itcpt:  # if there is intercept coefficient in the function
        resul_dict["coefficients"]["var_{:03d}".format(nvar)] = {
            "coeff_{:03d}".format(k): coeffs[k],
            "coeff_std_{:03d}".format(k): cstd[k],
            "coeff_tstat_{:03d}".format(k): tcoeff[k],
            "coeff_p_{:03d}".format(k): pcoeff[k]
        }

    return resul_dict, pcoeff


def compute_ypred_stats(xreal, yreal, func, coeffs, cstd, itcpt, logdata):
    nvar, nobsv = xreal.shape
    assert nobsv == len(yreal), "Compute error: Invalid arguments length!"
    ypred = func(xreal, *coeffs)
    if logdata:
        yreal = 10.0 ** np.copy(yreal)
        ypred = 10.0 ** ypred
    resul_dict, pcoeff = compute_resul_stats(yreal, ypred, coeffs, cstd, nvar, itcpt)
    return ypred, resul_dict, pcoeff


def print_stats(resul_dict, prtstr=True):
    ctaster = 0
    lin = "-------------------------------------------------------------------------------------\n"
    strstt = "========================= Final Regression Analysis Results =========================\n"
    strstt += lin
    strstt += "                               Regression Statistics\n"
    strstt += lin
    regr = resul_dict["regression"]
    strstt += "R2 ............................... : {:10.5f}\n".format(regr["R2"])
    strstt += "Adjusted R2 ...................... : {:10.5f}\n".format(regr["adjusted_R2"])
    strstt += "Mean Squared Error (MSE) ......... : {:10.5f}\n".format(regr["MSE"])
    strstt += "Root Mean Squared Error (RMSE) ... : {:10.5f}\n".format(regr["RMSE"])
    strstt += "Normalized RMSE by mean .......... : {:9.4f}%\n". \
        format(regr["NRMSE"] * 100.0)
    strstt += "Observations ..................... : {:10d}\n".format(regr["observations"])
    strstt += lin
    strstt += "                                       ANOVA\n"
    strstt += lin
    aregr = resul_dict["ANOVA"]["regression"]
    aresd = resul_dict["ANOVA"]["residual"]
    atot = resul_dict["ANOVA"]["total"]
    strstt += "{:16s} {:^8s} {:^14s} {:^14s} {:^14s} {:^14s}\n". \
        format("------ANOVA-----", "---df---", "------SS------", "------MS------",
               "----F stat----", "-------p------")
    p = aregr["p"]
    if p >= C_ALPHA_P:
        strp = "(*)"
        ctaster += 1
    else:
        strp = "   "
    strstt += "{:16s} {:8d} {:14.5f} {:14.5f} {:14.5f} {:3s} {:10.7f}\n". \
        format("Regression", aregr["df"], aregr["SS"], aregr["MS"],
               aregr["F"], strp, p)
    strstt += "{:16s} {:8d} {:14.5f} {:14.5f}\n". \
        format("Residual", aresd["df"], aresd["SS"], aresd["MS"])
    strstt += "{:16s} {:8d} {:14.5f}\n".format("Total", atot["df"], atot["SS"])
    strstt += lin
    varsdict = resul_dict["coefficients"]
    nvars = varsdict["n_variables"]
    ncoeff = varsdict["n_coefficients"]
    itcpt = varsdict["intercept"]
    nc = int((ncoeff - int(itcpt)) / nvars)
    neffcf = varsdict["n_effect_coeff"]
    strstt += "                         Coefficients (n = {:d}, effective = {:d})\n". \
        format(ncoeff, neffcf)
    strstt += lin
    strstt += "{:20s} {:^4s} {:^14s} {:^14s} {:^14s} {:^14s}\n". \
        format("------Variable------", "CNum", " -Coefficient-", "Standard Error",
               " Coeff. T stat", " Coefficient p")
    for v_id in sorted(varsdict.keys()):
        if v_id[:4] != "var_":
            continue
        v = varsdict[v_id]
        vname = v["name"]
        if vname == "intercept":
            assert itcpt, "Inconsistent intercept variable name in dictionary"
            klist = [ncoeff - 1]
        else:
            vnum = int(v_id[-3:])
            klist = list(range(vnum * nc, (vnum * nc) + nc))
        for k in klist:
            clbl = "coeff_{:03d}".format(k)
            slbl = "coeff_std_{:03d}".format(k)
            tlbl = "coeff_tstat_{:03d}".format(k)
            plbl = "coeff_p_{:03d}".format(k)
            assert clbl in v.keys() and \
                   slbl in v.keys() and \
                   tlbl in v.keys() and \
                   plbl in v.keys(), "Invalid coefficient index!"
            p = v[plbl]
            if p >= C_ALPHA_P:
                strp = "(*)"
                ctaster += 1
            else:
                strp = "   "
            strstt += "{:20s} {:^4d} {:14.5f} {:14.5f} {:14.5f}  {:3s} {:9.7f}\n". \
                format(vname, k, v[clbl], v[slbl], v[tlbl], strp, p)
    strstt += lin
    if ctaster > 0:
        strstt += "(*) p > {:5.3f} indicates ".format(C_ALPHA_P) + \
                  "statistically non-significant result or coefficient."

    if prtstr:
        print(strstt)
    return strstt


##############################################################################################
# Base functions and corresponding fitting
##############################################################################################
def linear_func(x, *coeffs):
    ncoeffs = len(coeffs)
    if ncoeffs == 1:
        return coeffs[0] * x
    if ncoeffs == 2:
        return (coeffs[0] * x) + coeffs[1]
    assert False, "Invalid number of coefficients in linear function!"


def fit_linear(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(linear_func, x, y,
                           p0=inicoeffs, method="trf",
                           bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, linear_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


def multi_linear_func(x, *coeffs):
    ncoeffs = len(coeffs)
    if not hasattr(x, "__len__") or isinstance(x, list) or \
            (isinstance(x, np.ndarray) and x.ndim == 1):
        assert 1 <= ncoeffs <= 2, "Invalid coefficients length!"
        val = x * coeffs[0]
        if ncoeffs == 2:
            val += coeffs[-1]
    elif isinstance(x, np.ndarray) and x.ndim > 1:
        nvar = len(x)
        nval = len(x[0])
        val = np.zeros(nval)
        itcpt = len(coeffs) - nvar
        assert 0 <= itcpt <= 1, "Invalid coefficients length!"
        for i in range(nvar):
            val += x[i, :] * coeffs[i]
        if itcpt > 0:
            val += coeffs[-1]
    else:
        assert False, "Invalid variable type or dimensions"
    return val


def fit_multi_linear_func(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x[0, :]) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(multi_linear_func, x, y, p0=inicoeffs,
                           method="trf", bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, multi_linear_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


def log_func(x, a, b):
    return np.log10(b * np.power(x, a))


def fit_log(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(log_func, x, y,
                           p0=inicoeffs,
                           method="trf", bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, log_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


def poly2_func(x, a, b):
    return (a * np.square(x)) + (b * x)


def fit_poly2(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(poly2_func, x, y,
                           p0=inicoeffs,
                           method="trf", bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, poly2_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


def multi_poly2_func(x, *coeffs):
    ncoeffs = len(coeffs)
    if not hasattr(x, "__len__") or isinstance(x, list) or \
            (isinstance(x, np.ndarray) and x.ndim == 1):
        assert 2 <= ncoeffs <= 3, "Invalid coefficients length!"
        val = poly2_func(x, coeffs[0], coeffs[1])
        if ncoeffs == 3:
            val += coeffs[-1]
    elif isinstance(x, np.ndarray) and x.ndim > 1:
        nvar = len(x)
        nval = len(x[0])
        val = np.zeros(nval)
        itcpt = len(coeffs) - (2 * nvar)
        assert 0 <= itcpt <= 1, "Invalid coefficients length!"
        for i in range(nvar):
            val += (coeffs[2 * i] * np.square(x[i, :])) + (coeffs[2 * i + 1] * x[i, :])
        if itcpt > 0:
            val += coeffs[-1]
    else:
        assert False, "Invalid variable type or dimensions"
    return val


def fit_multi_poly2_func(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x[0, :]) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(multi_poly2_func, x, y, p0=inicoeffs,
                           method="trf", bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, multi_poly2_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


def power_func(x, a, b):
    return a * np.power(x, b)


def fit_power(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(power_func, x, y,
                           p0=inicoeffs,
                           method="trf", bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, power_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


def multi_power_func(x, *coeffs):
    ncoeffs = len(coeffs)
    if not hasattr(x, "__len__") or isinstance(x, list) or \
            (isinstance(x, np.ndarray) and x.ndim == 1):
        assert 2 <= ncoeffs <= 3, "Invalid coefficients length!"
        val = power_func(x, coeffs[0], coeffs[1])
        if ncoeffs == 3:
            val += coeffs[-1]
    elif isinstance(x, np.ndarray) and x.ndim > 1:
        nvar = len(x)
        nval = len(x[0])
        val = np.zeros(nval)
        itcpt = len(coeffs) - (2 * nvar)
        assert 0 <= itcpt <= 1, "Invalid coefficients length!"
        for i in range(nvar):
            val += power_func(x[i, :], coeffs[2 * i], coeffs[2 * i + 1])
        if itcpt > 0:
            val += coeffs[-1]
    else:
        assert False, "Invalid variable type or dimensions"
    return val


def fit_multi_power_func(x, y, inicoeffs, bounds, itcp, logdata):
    assert len(x[0, :]) == len(y), "Fitting arrays dimension does not match!"
    popt, pcov = curve_fit(multi_power_func, x, y, p0=inicoeffs,
                           method="trf", bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    ypred, resul_dict, ppval = compute_ypred_stats(x, y, multi_power_func,
                                                   popt, perr, itcp, logdata)
    return ypred, popt, perr, ppval, resul_dict


##############################################################################################
# Model multi_poly2 validation and fitting
##############################################################################################
def create_validation_plan(nsamples):
    np.random.seed(19562704 + (nsamples * 97))
    validplan = np.ones((nsamples, C_N_ROUNDS), dtype=int) * C_SAMPLE_FIT
    samppart = np.random.choice(C_N_PARTITIONS, nsamples, replace=True)
    for r in range(C_N_ROUNDS):
        for p in C_VALID_PART[r]:
            selval = np.where(samppart == p, True, False)
            validplan[selval, r] = C_SAMPLE_VALID
    return validplan


def model_fitting(x_vals, y_vals, fit_func, func, logdata, varnames,
                  ncoeffs, itcpt, inibound, validplan):
    nsamp, nround = validplan.shape
    nvars = len(varnames)
    assert nvars == len(x_vals), "Number of variables mismatch!"
    assert nsamp == len(x_vals[0]) and nsamp == len(y_vals), "Number of samples mismatch!"
    inicoeffs = np.ones(ncoeffs)
    if inibound == 0:
        inibound = np.inf
    bounds = (np.ones(ncoeffs) * (-inibound), np.ones(ncoeffs) * inibound)
    fitcoeff = np.zeros((nround, ncoeffs))
    fitr2 = np.zeros(nround)
    for r in range(nround):
        selfit = np.where(validplan[:, r] == C_SAMPLE_FIT, True, False)
        selvalid = np.where(validplan[:, r] == C_SAMPLE_VALID, True, False)
        _, vcoeff, vcstd, _, _ = fit_func(x_vals[:, selfit], y_vals[selfit],
                                          inicoeffs, bounds, itcpt, logdata)
        _, valid_dict, ppval = compute_ypred_stats(x_vals[:, selvalid],
                                                   y_vals[selvalid], func,
                                                   vcoeff, vcstd, itcpt, logdata)
        fitcoeff[r] = vcoeff
        fitr2[r] = valid_dict["regression"]["R2"]

    meacoeff = np.mean(fitcoeff, axis=0)
    stdcoeff = np.std(fitcoeff, axis=0)
    bounds1 = (meacoeff - stdcoeff, meacoeff + stdcoeff)
    validmear2 = np.mean(fitr2)
    validstdr2 = np.std(fitr2)

    ypred, coeff, cffstd, cffp, resul_dict = fit_func(x_vals, y_vals, meacoeff,
                                                      bounds1, itcpt, logdata)

    r2 = resul_dict["regression"]["R2"]

    for i in range(nvars):
        resul_dict["coefficients"]["var_{:03d}".format(i)]["name"] = varnames[i]
    if itcpt:  # if function has intercept coefficient
        resul_dict["coefficients"]["var_{:03d}".format(nvars)]["name"] = "intercept"

    return ypred, resul_dict, coeff, cffstd, cffp, r2, validmear2, validstdr2


##############################################################################################
# main:
##############################################################################################
if __name__ == "__main__":
    print(__doc__)
