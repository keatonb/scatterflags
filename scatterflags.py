#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains functions for producing scatter plots with markers
that indicate which binary flags are set on each data point and accompanying
legends.

The output of the main scatterflags() function can be used as the input to
the flagbar() function to produce a colorbar.

Example usage:
    import scatterflags as sf
    import numpy as np
    import matplotlib.pyplot as plt
    npts=50
    x = np.arange(npts)
    y = np.random.randn(npts)
    f,(ax0,ax1) = plt.subplots(2,1,figsize=(6,5),gridspec_kw={'height_ratios':[4,1]})
    kwargs = sf.scatterflags(x,y,np.round(np.random.randint(1,64,npts)),ax=ax0)
    ax0.scatter(x,y,c='0',s=1,zorder=10,marker='*')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    sf.flagbar(cax=ax1,flaglabels=['flag'+str(i) for i in range(6)],barlabel='flags',**kwargs)
    plt.tight_layout()
    plt.show()

@author: bell@mps.mpg.de
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def scatterflags(x,y,flags,r=5.,dr=5.,ax=None,nflags=None,colors=None,cmap=None,
                 minzorder=1,**kwargs):
    """Plot scatter markers that indicate associated set flags.
    
    Args:
        x, y: input data
        flags: per-point flags (bitwise binary strings or base10 equiv ints)
        r: radius (pixels) of smallest marker (default: 5)
        dr: radius increase per flag (default: 5)
        ax: axis for scatterplot (optional)
        nflags: number of possible flags (can be inferred from flags)
        colors: colors to plot for each flag
        cmap: colormap to use for automaticly picking colors from (no jet!)
        minzorder: zorder of last flag (each earlier flag has one higher zorder)
        **kwargs: other keywords to pass to scatter (e.g. marker)
        
    Returns:
        kwargs for flagbar() function (produces colorbar legend) 
    """
    #Where is this plot going?
    if ax is None:
        ax = plt.gca()
    
    #Convert all integer strings to bitwise flags
    for i in range(len(flags)):
        if isinstance(flags[i], (int, long)):
            flags[i] = bin(flags[i])[2:]
    
    #Ensure flags is a list and x,y are arrays
    flags = list(flags)
    x = np.array(x)
    y = np.array(y)
    
    #Set number of flags if not explicit
    if nflags is None:
        nflags = max([len(flag) for flag in flags])
    
    #Pad string flags to (at least) nflags
    flags = [flag.zfill(nflags) for flag in flags]
    
    #Each flag needs an associated color
    #These may have been explicitly included
    ncolors = nflags
    #If fewer colors were included, cycle through
    if colors is not None:
        colors = [colors[i % len(colors)] for i in range(nflags)]
    else: #colors not specified
        colors = sns.color_palette(cmap,nflags)
        
    #Define marker sizes
    ms = (r+dr*np.arange(nflags))**2.
    
    #Scatter plot for each flag
    #Smallest to largest (highest zorder to lowest = minzorder)
    for i in range(nflags):
        flagged = np.where([int(chars[-i-1]) for chars in flags])
        ax.scatter(x[flagged],y[flagged],s=ms[i],c=mpl.colors.to_hex(colors[i]),
                   lw=0,zorder=minzorder+nflags-i,**kwargs)
    
    #Return dict of kwargs for formatting the colorbar
    return dict({'r':r,'dr':dr,'nflags':nflags,'colors':colors},**kwargs)


def flagbar(cax=None,nflags=None,r=5.,dr=5.,colors=None,cmap=None,
            flaglabels=None,barlabel=None,**kwargs):
    """Plot colorbar with scatter marker shapes/sizes
    
    Args:
        cax: target axis for colorbar (short and wide ideally)
        nflags: number of different flags
        r: radius (pixels) of smallest marker (default: 5)
        dr: radius increase per flag (default: 5)
        colors: colors to plot for each flag
        cmap: colormap to use for automaticly picking colors from (no jet!)
        flaglabels: str labels associated with each flag
        barlabel: overall label for colorbar
        **kwargs: other keywords to pass to scatter (e.g. marker)
        
    Note: besides the cax, the returned dicts from scatterflags() will set
        the rest of these args appropriately.  Call as `flagbar(cax,**kwargs)`
        where `kwargs = scatterflags(...)`
    """
    #Determine number of flags to represent
    if nflags is None:
        if flaglabels is not None:
            nflags = len(flaglabels)
        elif colors is not None:
            nflags = len(colors)
    
    #Throw and error if number of flags not defined
    try:
        _ = int(nflags)
    except TypeError:
        raise ValueError("Must specify number of flags to flagbar, explicitly or implicitly.")
    
    #Each flag needs an associated color
    #These may have been explicitly included
    ncolors = nflags
    #If fewer colors were included, cycle through
    if colors is not None:
        colors = [colors[i % len(colors)] for i in range(nflags)]
    else: #colors not specified
        colors = sns.color_palette(cmap,nflags)
    
    #Define marker sizes
    ms = (r+dr*np.arange(nflags))**2.
    
    #Length of flaglabels must be nflags
    if flaglabels is None:
        flaglabels = []
    nflaglabels = len(flaglabels)
    if nflaglabels < nflags:
        flaglabels += [""]*(nflags-nflaglabels)
    else:
        flaglabels = flaglabels[:nflags]
    
    #Where to plot
    if cax is None: #Probably better to specify
        cax = plt.gca()
    
    #Plot colorbar
    colorbounds = np.linspace(0,1,nflags+1)
    colorcenters = (colorbounds[1:]+colorbounds[:-1])/2.
    cb = mpl.colorbar.ColorbarBase(cax,cmap=mpl.colors.ListedColormap(colors),
                              ticks=colorcenters,spacing='proportional',
                              orientation='horizontal')
    cb.ax.set_xticklabels(flaglabels,rotation=45) #label flags under colorbar
    if barlabel is not None: #label y axis side
        cax.set_ylabel(barlabel)
    cax.scatter(colorcenters,[.5]*nflags,edgecolor='0',s=ms,c='none',**kwargs)
