def get_pallette_schaefer400():
    
    from nilearn import datasets
    from nilearn.plotting import plot_roi
    import seaborn as sns
    
    # Getting a priori defined 7-network partition's label 
    schaefer400 = datasets.fetch_atlas_schaefer_2018(n_rois=400, 
                                               yeo_networks=7, 
                                               resolution_mm=1,
                                               data_dir=None, 
                                               base_url=None, 
                                               resume=True, 
                                               verbose=1)

    # Dictionary to create network's abbreviations
    schaefer_dict = {
        "Vis": "VIS",
        "SomMot": "SOM",
        "DorsAttn": "DAN",
        "Sal": "VAN",
        "Limbic": "LIM",
        "Cont": "FPN",
        "Default": "DMN"
    }

    networks = [val for key, val in schaefer_dict.items() for roi in schaefer400['labels'] if key in roi.decode('UTF-8')]

    # Creating color palette for netorks
    schaefer_colors = {'DAN':'#00ab2e', 
                       'DMN':'#dc6179', 
                       'FPN':'#e8c830',
                       'LIM':'#7eb3d4', 
                       'SOM':'#7e8dc1', 
                       'VAN':'#d182c6',
                       'VIS':'#ac00ad',}


    network_pal = (sns.color_palette(schaefer_colors.values()))
    sns.palplot(network_pal)
     
    return(networks, network_pal)   



def surface_plots_schaefer400(parcel_values, cmap='RdBu_r', vmin=None, vmax=None, title=None):
    from nilearn import surface
    from nilearn import plotting
    import numpy as np

    n_parcels = 400
    if len(parcel_values) != n_parcels:
        raise ValueError(
            f"The mumber of parcels values ({len(parcel_values)}) does't match the number of parcels in a parcellarion ({n_parcels})"
        )

    # Freesurfer annotation file
    schaefer400_atlas_lh = surface.load_surf_data('../../support/schaeffer_400_parcellation/surface/lh.Schaefer2018_400Parcels_7Networks_order.annot')
    schaefer400_atlas_rh = surface.load_surf_data('../../support/schaeffer_400_parcellation/surface/rh.Schaefer2018_400Parcels_7Networks_order.annot')


    # Surface mesh
    lh_pial = surface.load_surf_mesh('../../support/schaeffer_400_parcellation/surface/lh.pial')
    lh_curv = surface.load_surf_data('../../support/schaeffer_400_parcellation/surface/lh.curv')
    rh_pial = surface.load_surf_mesh('../../support/schaeffer_400_parcellation/surface/rh.pial')
    rh_curv = surface.load_surf_data('../../support/schaeffer_400_parcellation/surface/rh.curv')

    _, idx1 = np.unique(schaefer400_atlas_lh, return_inverse=True)
    _, idx2 = np.unique(schaefer400_atlas_rh, return_inverse=True)

    scheafer_val_left = parcel_values[:200]
    scheafer_val_right = parcel_values[200:]

    scheafer_val_left_ins = np.insert(scheafer_val_left, 0, 0, axis=0)
    scheafer_val_right_ins = np.insert(scheafer_val_right, 0, 0, axis=0)

    plots = {"Left lateral": [lh_pial, lh_curv, scheafer_val_left_ins, "left", "lateral", idx1],
             "Left medial": [lh_pial, lh_curv, scheafer_val_left_ins, "left", "medial", idx1],
             "Right lateral": [rh_pial, rh_curv, scheafer_val_right_ins, "right", "lateral", idx2],
             "Right medial": [rh_pial, rh_curv, scheafer_val_right_ins, "right", "medial", idx2]}

    for plot, parameter in plots.items():
        plotting.plot_surf(parameter[0], parameter[2][parameter[5]], bg_map=parameter[1], 
                     hemi=parameter[3], view=parameter[4], 
                     cmap=cmap, 
                     bg_on_data=True,
                     colorbar=True,      
                     vmin=vmin,
                     vmax=vmax,
                     title=f"{title}{plot}",
                     darkness=1)
        
        
def get_seaborn_slyle(font_size = 15):

    # Plot style setup
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-white')
    plt.rcParams['font.family'] = 'Helvetica'

    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', linewidth=2.2)
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('lines', linewidth=2.2, color='gray')
    
    #ax.tick_params(axis='both', color = 'black', length = 5, width = 2)