import argparse
import os
import json
import csv
import itertools
import numpy as np
import shutil
from datetime import datetime
from matplotlib import pyplot as plt

import PyZyrFitCurve as Zfit

__author__ = "Luiz Claudio Navarro"
__version__ = "0.0.12"
__date__ = "2022.10.17"
libgroup = "Fish Allometry Project"
libname = "Fish Allometry Model Curve Determination"
libversion = "Version: " + __version__ + " Date: " + __date__ + "."
copyrightmsg = "Zyryus Consulting (c) 2019-2021."
__doc__ = libgroup + " " + libname + "\n" + libversion + " " + copyrightmsg + "\n"

##############################################################################################
# Constants and global variables
##############################################################################################
C_MAX_PLOT_N_MODELS = 10

C_ALPHA_P = 0.05


##############################################################################################
# Utility Function
##############################################################################################
def load_fish_allo_data(allofname):
    print("*** Loading fish allometry data from {:s} ***".format(allofname))
    assert os.path.isfile(allofname) and allofname.endswith(".csv"), \
        "Allometry data csv file not found!"

    idname = ""
    tgtname = ""
    varname = []
    alloid = []
    allotgt = []
    allodata = []
    count = 0
    with open(allofname, 'r', newline='') as allofile:
        allorows = csv.reader(allofile, delimiter=',')
        for row in allorows:
            if count == 0:
                idname = row[0]
                tgtname = row[1]
                varname = row[2:]
            else:
                assert len(row) == (len(varname) + 2), "Inconsistent records length!"
                try:
                    alloid.append(row[0])
                    allotgt.append(float(row[1]))
                    allodata.append(list(map(float, row[2:])))
                except ValueError:
                    assert False, "Invalid record data types!"
            count += 1
    print("    {:d} lines loaded from file {:s} .".format(count, os.path.basename(allofname)))

    return idname, tgtname, varname, alloid, np.array(allotgt), np.array(allodata)


def save_fish_allometry_model_metrics(mtrfname, functype, modelsid, varsidx,
                                      varnames, coeffvals, cfstdvals, cfpvals, itcpt,
                                      r2, mear2, stdr2, anovaf, anovap, effc_coeff):
    print("*** Saving fish allometry models metrics into {:s} ***".format(mtrfname))
    assert mtrfname.endswith(".csv"), "File name is not .csv!"
    nmdl = len(modelsid)
    assert nmdl == len(varsidx) == len(r2) == len(mear2) == len(stdr2) and \
           nmdl == len(coeffvals) == len(cfstdvals) == len(cfpvals) and \
           nmdl == len(anovaf) == len(anovap) and \
           nmdl == len(effc_coeff), "Models data dimensions mismatch!"
    cteffc = 0
    with open(mtrfname, 'w', newline='') as mtrfile:
        mtrwriter = csv.writer(mtrfile, delimiter=',')
        mtrwriter.writerow(["func_type", "model id", "nvars",
                            "r2", "valid_mean_r2", "valid_std_r2",
                            "anova_fstat", "anova_p", "n_coeff", "effective_coeff",
                            "<var_names_list>", "<coeff_list>", "<cfstd_list>", "<cfp_list>"])
        for i in range(nmdl):
            vnidx = list(varsidx[i])
            nvar = len(vnidx)
            ncoeff = len(coeffvals[i])
            if ncoeff == effc_coeff[i]:
                cteffc += 1
            outrow = [functype, modelsid[i], nvar, r2[i],
                      mear2[i], stdr2[i], anovaf[i], anovap[i], ncoeff, effc_coeff[i]]
            if r2[i] != 0.0 and len(coeffvals[i]) != 0:
                assert ncoeff == len(cfstdvals[i]), \
                    "Coefficients length does not match"
                if itcpt[i]:
                    outrow += list(map(lambda x: varnames[x], vnidx)) + ["intercept"] + \
                              coeffvals[i] + cfstdvals[i] + cfpvals[i]
                else:
                    outrow += list(map(lambda x: varnames[x], vnidx)) + \
                              coeffvals[i] + cfstdvals[i] + cfpvals[i]
            mtrwriter.writerow(outrow)
    print("    {:d} lines saved into {:s}.".format(nmdl, os.path.basename(mtrfname)))
    print("    {:d} models with n. coefficients equal to n. effective coefficients..".
          format(cteffc))
    return


def save_fish_allometry_model_predictions(predfname, fishids, modelsid, realval, predvals):
    print("*** Saving fish allometry predictions into {:s} ***".format(predfname))
    assert predfname.endswith(".csv"), "File name is not .csv!"
    nfish = len(fishids)
    assert nfish == len(realval) and nfish == len(predvals), "Dimensions mismatch!"

    with open(predfname, 'w', newline='') as predfile:
        predwriter = csv.writer(predfile, delimiter=',')
        predwriter.writerow(["fish id", "real_value"] + modelsid)
        for i in range(nfish):
            outrow = [fishids[i], realval[i]] + predvals[i].tolist()
            predwriter.writerow(outrow)
    print("    {:d} lines saved into {:s}.".format(nfish, os.path.basename(predfname)))
    return


def save_fish_allometry_model_stats(sttfname, mdlstats):
    print("*** Saving fish allometry statistics into {:s} ***".format(sttfname))
    assert sttfname.endswith(".json"), "File name is not .json!"
    with open(sttfname, "w") as sttfile:
        json.dump(mdlstats, sttfile)
    print("    {:d} models saved into {:s}.".
          format(len(mdlstats), os.path.basename(sttfname)))
    return


def var_index_combinations(n, k):
    idx = list(range(n))
    idx_comb_list = list(itertools.combinations(idx, k))
    return idx_comb_list


def save_text_report(mdls_r2, mdls_id, mdls_coeff, mdls_vars, mdls_itcp,
                     mdls_dict, varnames, funcfit, title, savefprfx):
    nmdls = len(mdls_id)
    assert nmdls == len(mdls_r2), "Dimensions mismatch!"
    if savefprfx:
        print("--- Reporting {:s} ---".format(title.replace('\n', ' ')))
        reptfname = savefprfx + "_Models_Best_Ranked_Report.txt"
        with open(reptfname, "w") as rptfile:
            rptstr = "Report: {:s}\n".format(title)
            rptfile.write(rptstr)
            for i in range(nmdls):
                rptstr = "\nRank: {:d} - ".format(i + 1)
                rptstr += "Model_id: {:s} - ".format(mdls_id[i])
                coeffs = mdls_coeff[i]
                idxvars = mdls_vars[i]
                if mdls_r2[i] != 0.0 and len(idxvars) != 0:
                    rptstr += "Fitted function: "
                    if funcfit == "loglin":
                        rptstr += "10 ** ("
                    for j in range(len(idxvars)):
                        if j > 0:
                            rptstr += " + "
                        vname = varnames[idxvars[j]]
                        if funcfit == "poly2":
                            rptstr += "(({:7.4f} * ({:s} ^ 2))".format(coeffs[2 * j], vname)
                            rptstr += "({:7.4f} * {:s}))".format(coeffs[(2 * j) + 1], vname)
                        elif funcfit == "pow":
                            rptstr += "({:7.4f} * ".format(coeffs[2 * j])
                            rptstr += "({:s} ^ {:7.4f}))".format(vname, coeffs[(2 * j) + 1])
                        elif funcfit == "lin":
                            rptstr += "{:s} * {:7.4f}".format(vname, coeffs[j])
                        elif funcfit == "loglin":
                            rptstr += "(log10({:s}) * {:7.4f})".format(vname, coeffs[j])
                        else:
                            assert False, "Invalid fitting function"
                    if mdls_itcp[i]:
                        rptstr += " + {:7.4f}".format(coeffs[-1])
                    if funcfit == "loglin":
                        rptstr += ")"
                    rptstr += "\n"
                    rptstr += Zfit.print_stats(mdls_dict[i], prtstr=False)
                    rptstr += "\n"
                else:
                    rptstr += "Fitting process has not converged!\n"
                rptfile.write(rptstr)
    return


def plot_report_ranked_models(mdls_r2, mdls_mear2, mdls_stdr2, mdls_id,
                              title, savefprfx):
    print("--- Ploting {:s} ---".format(title.replace('\n', ' ')))
    nmdls = len(mdls_id)
    assert nmdls == len(mdls_r2) and nmdls == len(mdls_mear2) and \
           nmdls == len(mdls_stdr2), "Dimensions mismatch!"
    xtk = []
    for i in range(nmdls):
        xtk.append("{:d}:{:s}".format(i + 1, mdls_id[i]))
    x = np.arange(nmdls)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, mdls_mear2,
                yerr=mdls_stdr2,
                label="Validation")
    ax.plot(x, mdls_r2, "*r", label="Final Model")
    ax.set_xlabel("model")
    ax.set_xticks(x)
    ax.set_xticklabels(xtk, rotation=90, fontsize=8)
    ax.set_ylabel("R2")
    ax.set_title(title)
    ax.legend(loc="lower left")
    plt.tight_layout()

    if savefprfx:
        savefname = savefprfx + "_Models_Best_Ranked_chart_300dpi.pdf"
        plt.savefig(savefname, dpi=300)

    plt.close('all')

    return


def plot_models_prediction(val_real, mdls_pred, mdlsid,
                           measure, title, savefprfx):
    print("--- Ploting {:s} ---".format(title.replace('\n', ' ')))
    nvals, nmdls = mdls_pred.shape
    assert nvals == len(val_real) and nmdls == len(mdlsid), "Dimensions mismatch!"
    lbl = []
    for i in range(nmdls):
        lbl.append("{:d}:{:s}".format(i + 1, mdlsid[i]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    assert nvals > 6, "Number of models not suported"
    fmtlist = ["ob", "*g", "+k", "dm", "sc", "oy"]
    for i in range(nmdls - 1, -1, -1):
        ax.plot(val_real, mdls_pred[:, i], fmtlist[i],
                label=lbl[i], alpha=0.5)
    ax.plot(val_real, val_real, ".:r", label="measured value")
    ax.set_xlabel("real value")
    ax.set_ylabel(measure)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc="lower right")
    plt.tight_layout()

    if savefprfx:
        savefname = savefprfx + "_Models_Prediction_chart_300dpi.pdf"
        plt.savefig(savefname, dpi=300)

    plt.close('all')

    return


##############################################################################################
# Model Determination
##############################################################################################
def fish_allom_model_det(datafn, dirout, funcfit, inibound):
    initime = datetime.now()
    dirout = dirout + "_" + funcfit
    print("### Begin fish allometry model determination ###")
    print("Input fish allometry data ..............: {:s}".format(datafn))
    print("Output folder ..........................: {:s}".format(dirout))
    print("Function to fit ........................: {:s}".format(funcfit))
    print("Initial bound value for coefficients ...: {:f}".format(inibound))

    if funcfit == "poly2":
        fitfunc = Zfit.fit_multi_poly2_func
        func = Zfit.multi_poly2_func
        logdata = False
    elif funcfit == "pow":
        fitfunc = Zfit.fit_multi_power_func
        func = Zfit.multi_power_func
        logdata = False
    elif funcfit == "lin" or funcfit == "loglin":
        fitfunc = Zfit.fit_multi_linear_func
        func = Zfit.multi_linear_func
        logdata = funcfit == "loglin"
    else:
        assert False, "Invalid fitting function"

    assert os.path.isfile(datafn) and datafn.endswith(".csv"), \
        "Input fish allometry csv file not found!"

    if not os.path.isdir(dirout):
        os.makedirs(dirout)
        print("Output folder not found, then created!")

    basefn = os.path.basename(datafn)
    print("*** Copy {:s} input file to output folder ***".format(basefn))
    shutil.copy2(datafn, os.path.join(dirout, basefn))

    fishidhdr, tgthdr, varhdr, alloid, allotgt, allovars = \
        load_fish_allo_data(datafn)

    # if input and target should be logged then do to increase performance
    realtgt = np.copy(allotgt)
    if logdata:
        allotgt = np.log10(allotgt)
        allovars = np.log10(allovars)

    nfish, nvars = allovars.shape
    assert nvars == len(varhdr) and nfish == len(alloid) and \
           nfish == len(allotgt), "Dimensions mismatch"

    valid_plan = Zfit.create_validation_plan(nfish)

    print("*** Determining variable combinations ***")
    comblist = []
    for k in range(1, nvars + 1):
        comblist += var_index_combinations(nvars, k)

    nmdls = 2 * len(comblist)  # with and without intercept
    mdlr2 = np.zeros(nmdls)
    mdlmr2 = np.zeros(nmdls)
    mdlsr2 = np.zeros(nmdls)
    mdlanvf = np.zeros(nmdls)
    mdlanvp = np.zeros(nmdls)
    mdlpred = np.zeros((nfish, nmdls))
    mdlncoef = np.zeros(nmdls)
    mdlnefcf = np.zeros(nmdls)
    mdlitcp = np.zeros(nmdls, dtype=bool)
    mdlvars, mdlcoeff, mdlcfstd, mdlcfpval = [], [], [], []
    mdlid = []
    mdldict = {}

    print("*** Fitting model for {:d} variable combinations ***")
    for i in range(len(comblist)):
        idxvars = comblist[i]
        mdlnvars = len(idxvars)
        x_vars = np.transpose(allovars[:, idxvars])
        name_vars = [varhdr[i] for i in idxvars]
        for itcpt in range(2):
            k = (2 * i) + itcpt
            mdlname = "{:s}_{:d}_{:d}vars_{:d}itcp". \
                format(funcfit, i + 1, mdlnvars, itcpt)
            print("--- #{:d} Fitting model {:s} ".format(i, mdlname) +
                  "for combination of {:d} variables ".format(mdlnvars) +
                  "with {:d} intercept ---".format(itcpt))
            if funcfit == "lin" or funcfit == "loglin":
                ncoeff = mdlnvars
            else:
                ncoeff = 2 * mdlnvars
            if itcpt:
                ncoeff += 1
            try:
                ypred, regr_dict, coeff, cfstd, cfpval, r2, mear2, stdr2 = \
                    Zfit.model_fitting(x_vars, allotgt, fitfunc, func, logdata,
                                       name_vars, ncoeff, itcpt, inibound, valid_plan)
                anvf = regr_dict["ANOVA"]["regression"]["F"]
                anvp = regr_dict["ANOVA"]["regression"]["p"]
                neffcf = regr_dict["coefficients"]["n_effect_coeff"]
            except (RuntimeError, ValueError) as err:
                print("!!!! Model could not converge! " + str(err))
                ypred, regr_dict = np.zeros(nfish), {}
                coeff, cfstd, cfpval = np.zeros(ncoeff), np.zeros(ncoeff), np.ones(ncoeff)
                cfpval = np.ones(ncoeff)
                r2, mear2, stdr2, anvf, anvp, neffcf = 0.0, 0.0, 0.0, 0.0, 1.0, 0
            mdlvars.append(idxvars)
            mdlcoeff.append(coeff.tolist())
            mdlcfstd.append(cfstd.tolist())
            mdlcfpval.append(cfpval.tolist())
            mdlncoef[k] = ncoeff
            mdlitcp[k] = itcpt
            mdlnefcf[k] = neffcf
            mdlpred[:, k] = ypred
            mdlr2[k] = r2
            mdlmr2[k] = mear2
            mdlsr2[k] = stdr2
            mdlanvf[k] = anvf
            mdlanvp[k] = anvp
            mdlid.append(mdlname)
            mdldict[mdlid[-1]] = regr_dict
            print("    #{:d} Combination of {:d} variables ".format(i, mdlnvars) +
                  "model {:s} fitted r2: {:6.4f}, ".format(mdlname, r2) +
                  "ANOVA F stat: {:6.4e}, p: {:6.4e}, ".format(anvf, anvp) +
                  "n. coefficientes {:d}".format(ncoeff) +
                  "n. effective coeff {:d}".format(neffcf))

    print("*** Saving models fitted results ***")
    mtrfname = os.path.join(dirout, basefn.replace(".csv", "_CombVar_Models_Metrics.csv"))
    save_fish_allometry_model_metrics(mtrfname, funcfit, mdlid, mdlvars, varhdr,
                                      mdlcoeff, mdlcfstd, mdlcfpval, mdlitcp,
                                      mdlr2, mdlmr2, mdlsr2,
                                      mdlanvf, mdlanvp, mdlnefcf)
    prdfname = os.path.join(dirout, basefn.replace(".csv", "_CombVar_Models_Predictions.csv"))
    save_fish_allometry_model_predictions(prdfname, alloid, mdlid, realtgt, mdlpred)

    jsonfname = os.path.join(dirout, basefn.replace(".csv", "_CombVar_Models_Statistics.json"))
    save_fish_allometry_model_stats(jsonfname, mdldict)

    print("*** Reporting ranked models fitted results ***")
    selmdls = np.where(mdlnefcf == mdlncoef, True, False)
    headsort = np.zeros(nmdls)
    nsigmdl = np.count_nonzero(selmdls)
    if nsigmdl > 0:
        headsort[selmdls] += 1.0
    idxsort = np.argsort(-(mdlmr2 - mdlsr2 + headsort))
    allmdlsr2 = mdlr2[idxsort]
    allmdlsmr2 = mdlmr2[idxsort]
    allmdlssr2 = mdlsr2[idxsort]
    allmdlsid = [mdlid[i] for i in idxsort]
    allmdlscoeff, allmdlsvars, allmdlitcp, allmdldict = [], [], [], []
    for i in idxsort:
        allmdlscoeff.append(mdlcoeff[i])
        allmdlsvars.append(mdlvars[i])
        allmdlitcp.append(mdlitcp[i])
        allmdldict.append(mdldict[mdlid[i]])

    filesprfx = os.path.join(dirout, basefn.replace(".csv", ""))
    save_text_report(allmdlsr2, allmdlsid, allmdlscoeff, allmdlsvars,
                     allmdlitcp, allmdldict, varhdr, funcfit,
                     "R2 metric of {:d} ranked allometry models".
                     format(nmdls), filesprfx)

    print("*** Plotting statistically significant models ***")
    nsigmdl = np.count_nonzero(selmdls)
    if nsigmdl > 0:
        print("    There are {:d} statistically significant models.".format(nsigmdl))
        if 0 < C_MAX_PLOT_N_MODELS < nsigmdl:
            nbest = C_MAX_PLOT_N_MODELS
        else:
            nbest = nsigmdl
        print("    Best {:d} statistically significant models are ranked.".format(nbest))
        bestmdlsr2 = allmdlsr2[:nbest]
        bestmdlsmr2 = allmdlsmr2[:nbest]
        bestmdlssr2 = allmdlssr2[:nbest]
        bestmdlsid = [allmdlsid[i] for i in range(nbest)]

        plot_report_ranked_models(bestmdlsr2, bestmdlsmr2, bestmdlssr2, bestmdlsid,
                                  "R2 metric of {:d} best significant ranked allometry models".
                                  format(nbest), filesprfx)

        ncomp = min(6, nbest)
        plot_models_prediction(realtgt, mdlpred[:, idxsort[:ncomp]],
                               bestmdlsid[:ncomp], tgthdr,
                               "Predictions comparison of {:d} best significant ranked models".
                               format(ncomp), filesprfx)
    else:
        print("    There is no statistically significant model (p model and p of all coefficients <= 0.05).")

    print("### End fish allometry model determination ###")
    endtime = datetime.now()
    elapsed = endtime - initime
    print("Begun at ......: {:s}".format(str(initime)))
    print("Finished at ...: {:s}".format(str(endtime)))
    print("Elapsed .......: {:s}".format(str(elapsed)))
    print("########## End Program ##########")

    return


##############################################################################################
# main:
##############################################################################################
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-i", "--infile", required=True,
                        help="input fish allometry data csv file")
    parser.add_argument("-o", "--outdir", required=True,
                        help="output folder")
    parser.add_argument("-f", "--function", required=False,
                        default="linear", help="function to fit: poly2, pow")
    parser.add_argument("-b", "--bound", required=False,
                        default=0.0, help="coefficients bound, if 0 then -inf, inf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(__doc__)
    fish_allom_model_det(args.infile, args.outdir, args.function, float(args.bound))
