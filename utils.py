# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
# Author(s):       Vincent Rouvreau, Bertrand Michel
#
# Copyright (C) 2016 Inria

from os import path
from math import isfinite
import numpy as np
from functools import lru_cache
import warnings
import errno
import os

from gudhi.reader_utils import read_persistence_intervals_in_dimension
from gudhi.reader_utils import read_persistence_intervals_grouped_by_dimension

_gudhi_matplotlib_use_tex = True

def _array_handler(a):
    """
    :param a: if array, assumes it is a (n x 2) np.array and return a
                persistence-compatible list (padding with 0), so that the
                plot can be performed seamlessly.
    """
    if isinstance(a[0][1], (np.floating, float)):
        return [[0, x] for x in a]
    else:
        return a

@lru_cache(maxsize=1)
def _matplotlib_can_use_tex():
    """This function returns True if matplotlib can deal with LaTeX, False otherwise.
    The returned value is cached.
    """
    try:
        from matplotlib import checkdep_usetex

        return checkdep_usetex(True)
    except ImportError as import_error:
        warnings.warn(f"This function is not available.\nModuleNotFoundError: No module named '{import_error.name}'.")

def _limit_to_max_intervals(persistence, max_intervals, key):
    """This function returns truncated persistence if length is bigger than max_intervals.
    :param persistence: Persistence intervals values list. Can be grouped by dimension or not.
    :type persistence: an array of (dimension, array of (birth, death)) or an array of (birth, death).
    :param max_intervals: maximal number of intervals to display.
        Selected intervals are those with the longest life time. Set it
        to 0 to see all. Default value is 1000.
    :type max_intervals: int.
    :param key: key function for sort algorithm.
    :type key: function or lambda.
    """
    if max_intervals > 0 and max_intervals < len(persistence):
        warnings.warn(
            "There are %s intervals given as input, whereas max_intervals is set to %s."
            % (len(persistence), max_intervals)
        )
        # Sort by life time, then takes only the max_intervals elements
        return sorted(persistence, key=key, reverse=True)[:max_intervals]
    else:
        return persistence

def __min_birth_max_death(persistence, band=0.0):
    """This function returns (min_birth, max_death) from the persistence.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param band: band
    :type band: float.
    :returns: (float, float) -- (min_birth, max_death).
    """
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][1][0]
    for interval in reversed(persistence):
        if float(interval[1][1]) != float("inf"):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    if band > 0.0:
        max_death += band
    # can happen if only points at inf death
    if min_birth == max_death:
        max_death = max_death + 1.0
    return (min_birth, max_death)

def __plot_persistence_diagram_(
    persistence=[],
    MAX_DEATH=20,
    persistence_file="",
    alpha=0.6,
    band=0.0,
    max_intervals=1000000,
    inf_delta=0.1,
    legend=False,
    colormap=None,
    axes=None,
    fontsize=16,
    greyblock=True,
):
    """
    A customized plot function to set the axis scale manually so that all your diagrams have the same scall.
    IMPORTANT: ****THE MAX_DEATH NEED TO BE SET MANUALLY****. If you don't care the scale among different diagrams,
    use plotPD() INSTEAD.
    Below are initial documents.

    This function plots the persistence diagram from persistence values
    list, a np.array of shape (N x 2) representing a diagram in a single
    homology dimension, or from a `persistence diagram <fileformats.html#persistence-diagram>`_ file`.

    :param persistence: Persistence intervals values list. Can be grouped by dimension or not.
    :type persistence: an array of (dimension, array of (birth, death)) or an array of (birth, death).
    :param persistence_file: A `persistence diagram <fileformats.html#persistence-diagram>`_ file style name
        (reset persistence if both are set).
    :type persistence_file: string
    :param alpha: plot transparency value (0.0 transparent through 1.0
        opaque - default is 0.6).
    :type alpha: float.
    :param band: band (not displayed if :math:`\leq` 0. - default is 0.)
    :type band: float.
    :param max_intervals: maximal number of intervals to display.
        Selected intervals are those with the longest life time. Set it
        to 0 to see all. Default value is 1000000.
    :type max_intervals: int.
    :param inf_delta: Infinity is placed at :code:`((max_death - min_birth) x
        inf_delta)` above :code:`max_death` value. A reasonable value is
        between 0.05 and 0.5 - default is 0.1.
    :type inf_delta: float.
    :param legend: Display the dimension color legend (default is False).
    :type legend: boolean.
    :param colormap: A matplotlib-like qualitative colormaps. Default is None
        which means :code:`matplotlib.cm.Set1.colors`.
    :type colormap: tuple of colors (3-tuple of float between 0. and 1.).
    :param axes: A matplotlib-like subplot axes. If None, the plot is drawn on
        a new set of axes.
    :type axes: `matplotlib.axes.Axes`
    :param fontsize: Fontsize to use in axis.
    :type fontsize: int
    :param greyblock: if we want to plot a grey patch on the lower half plane for nicer rendering. Default True.
    :type greyblock: boolean
    :returns: (`matplotlib.axes.Axes`): The axes on which the plot was drawn.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib import rc

        if _gudhi_matplotlib_use_tex and _matplotlib_can_use_tex():
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
        else:
            plt.rc("text", usetex=False)
            plt.rc("font", family="DejaVu Sans")

        if persistence_file != "":
            if path.isfile(persistence_file):
                # Reset persistence
                persistence = []
                diag = read_persistence_intervals_grouped_by_dimension(persistence_file=persistence_file)
                for key in diag.keys():
                    for persistence_interval in diag[key]:
                        persistence.append((key, persistence_interval))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), persistence_file)

        try:
            persistence = _array_handler(persistence)
            persistence = _limit_to_max_intervals(
                persistence, max_intervals, key=lambda life_time: life_time[1][1] - life_time[1][0]
            )
            min_birth, max_death = __min_birth_max_death(persistence, band)
        except IndexError:
            min_birth, max_death = 0.0, 1.0
            pass

        max_death = MAX_DEATH
        delta = (max_death - min_birth) * inf_delta
        # Replace infinity values with max_death + delta for diagram to be more
        # readable
        infinity = max_death + delta
        axis_end = max_death + delta / 2
        axis_start = min_birth - delta

        if axes == None:
            _, axes = plt.subplots(1, 1)
        if colormap == None:
            colormap = plt.cm.Set1.colors
        # bootstrap band
        if band > 0.0:
            x = np.linspace(axis_start, infinity, 1000)
            axes.fill_between(x, x, x + band, alpha=alpha, facecolor="red")
        # lower diag patch
        if greyblock:
            axes.add_patch(
                mpatches.Polygon(
                    [[axis_start, axis_start], [axis_end, axis_start], [axis_end, axis_end]],
                    fill=True,
                    color="lightgrey",
                )
            )
        # line display of equation : birth = death
        axes.plot([axis_start, axis_end], [axis_start, axis_end], linewidth=1.0, color="k")

        x=[birth for (dim,(birth,death)) in persistence]
        y=[death if death != float("inf") else infinity for (dim,(birth,death)) in persistence]
        c=[colormap[dim] for (dim,(birth,death)) in persistence]

        axes.scatter(x,y,alpha=alpha,color=c)
        if float("inf") in (death for (dim,(birth,death)) in persistence):
            # infinity line and text
            axes.plot([axis_start, axis_end], [infinity, infinity], linewidth=1.0, color="k", alpha=alpha)
            # Infinity label
            yt = axes.get_yticks()
            yt = yt[np.where(yt < axis_end)]  # to avoid plotting ticklabel higher than infinity
            yt = np.append(yt, infinity)
            ytl = ["%.3f" % e for e in yt]  # to avoid float precision error
            ytl[-1] = r"$+\infty$"
            axes.set_yticks(yt)
            axes.set_yticklabels(ytl)

        if legend:
            dimensions = list(set(item[0] for item in persistence))
            axes.legend(handles=[mpatches.Patch(color=colormap[dim], label=str(dim)) for dim in dimensions])

        axes.set_xlabel("Birth", fontsize=fontsize)
        axes.set_ylabel("Death", fontsize=fontsize)
        axes.set_title("Persistence diagram", fontsize=fontsize)
        # Ends plot on infinity value and starts a little bit before min_birth
        axes.axis([axis_start, axis_end, axis_start, infinity + delta / 2])
        return axes

    except ImportError as import_error:
        warnings.warn(f"This function is not available.\nModuleNotFoundError: No module named '{import_error.name}'.")