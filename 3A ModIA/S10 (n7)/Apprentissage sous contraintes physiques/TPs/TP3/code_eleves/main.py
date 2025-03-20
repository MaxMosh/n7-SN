"""
Main script
Exemple:
 - launch experiment
 $ python main.py -save lin2d_exp.py -run
 $ python main.py -save lorenz_exp.py -run
"""
import sys
import os
import subprocess
#import pprint
import torch
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as tck
#from matplotlib import rc


def get_args(args, s):
    """get the list of elements of args between s and the next
    element strating with -
    """
    nargs, flag = [], False
    if s in args:
        i, flag = args.index(s) + 1, True
    while flag:
        if i == len(args):
            flag = False
        else:
            arg = args[i]
            if arg[0] == "-":
                flag = False
            else:
                nargs.append(arg)
        i += 1
    return nargs


def get_arg(args, s, fail=None, n=0):
    """
    returns the nth element after s in args if succeed
    otherwise returns fail
    """
    lst = get_args(args, s)
    if len(lst) > n:
        fail = lst[n]
    return fail


if __name__ == "__main__":
    args = sys.argv[1:]
    # The names of the experiments to work with
    nameexps = get_args(args, "-exps")
    # getting disdirs by loading kwargs associated with nameexps
    disdirs = []
    for nameexp in nameexps:
        disdirs.append(torch.load(
            nameexp + "/kwargs.pt",
            map_location=torch.device("cpu"))["directory"])

    if "-save" in args:
        """Experiment parameters are saved in a dict.  This dict is
        written on disk by a script.  The command launch such a
        script. This script writes the kwargs and returns the
        corresponding experiment name and directory which are appended
        to nameexps and disdirs. the save_dict
        """
        dict_script = get_arg(args, "-save")
        out = subprocess.check_output([
            "python3",
            dict_script
        ]).decode("utf-8")
        print("saved: " + out)
        for ab in out.split(" "):
            dir_saved, nameexp_saved = ab.split(",")
            nameexps.append(nameexp_saved)
            disdirs.append(dir_saved)

    if "-run" in args:
        """ Run the experiments in nameexps
        """
        job_script = get_arg(args, "-run")
        cmd = ""
        for nameexp, disdir in zip(nameexps, disdirs):
            if job_script is None:
                # Normal execution
                cmd += "python3 " +\
                    disdir + "manage_exp.py " +\
                    disdir + nameexp
            else:
                assert(0)
            os.system(cmd)

    if "-plot" in args:
        """ plot the experiments in nameexps
       """
        scores, kwargs = {"train": {}, "test": {}}, {}
        for nameexp in nameexps:
            kwargs[nameexp] = torch.load(nameexp+"/kwargs.pt")
            if os.path.exists(nameexp+"/scores.pt"):
                scores["train"][nameexp] = torch.load(
                    nameexp+"/scores.pt",
                    map_location=torch.device('cpu'))
            if os.path.exists(nameexp+"/test_scores.pt"):
                scores["test"][nameexp] = torch.load(
                    nameexp+"/test_scores.pt",
                    map_location=torch.device('cpu'))
        kscores = list(scores["train"][nameexps[-1]].keys())
        start, step, end = 0, 1, min(
            [len(scores["train"][s][kscores[0]]) for s in nameexps])
        start = int(get_arg(args, "-start", start))
        end = int(get_arg(args, "-end", end))
        step = int(get_arg(args, "-step", step))
        kscores = get_arg(args, "-kscores", kscores)

        # plot parameters
        rc('figure', figsize=(8, 10))
        rc('font', size=8.0)
        rc('figure.subplot',
           top=0.98,
           bottom=0.055,
           left=0.12,
           right=0.96,
           hspace=0.1,
           wspace=0.5)
        fig, axes = plt.subplots(len(kscores), 2)
        for k, mode in enumerate(["train", "test"]):
            for j, nameexp in enumerate(nameexps):
                print("### "+nameexp+" "+mode+" ###")
                print("  CYCLES")
                print("    last= "+str(start+step*(end-start)))
                print(scores[mode].keys())
                if scores[mode] != {}:
                    for i, kscore in enumerate(kscores):
                        out = scores[mode][nameexp][kscore]
                        print("  " + kscore)
                        if np.isnan(out).any():
                            print("Warning: Nan in "+nameexp)
                            out = np.ma.masked_invalid(out)
                        out = out[start:end:step]
                        axe = axes[i, k]
                        axe.plot(range(start+1, start+step*len(out)+1, step),
                                 out,
                                 label=nameexp)
                        #axe.set_yscale('log')
                        axe.yaxis.set_label_position("right")
                        axe.set_ylabel(kscore)
                        #axe.get_yaxis().set_major_locator(tck.LogLocator())
                        #axe.get_yaxis().set_major_formatter(
                        #    tck.LogFormatterSciNotation())
                        axe.grid()
                        print("    last= " + str(out[-1]))
                        print("    mean= " + str(np.mean(out)))
        axes[0, 0].legend()
        axes[0, 0].set_title("train")
        axes[0, 1].set_title("test")
        plt.savefig(nameexps[-1]+"/plot.pdf")
        plt.show()
