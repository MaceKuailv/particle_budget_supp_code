import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

dpi = 300
rerun = True
regen_talk = False

nep_extent = [110, 250, 20, 65]
projection = ccrs.Mercator(central_longitude=180.0, 
                           min_latitude=20.0, 
                           max_latitude=80.0,
                           latitude_true_scale=40.0)

balance = cmocean.cm.balance
tempo = cmocean.cm.tempo
depth_cmap = "Greys_r"
depth_norm = mpl.colors.Normalize(vmin=-5000, vmax=5000)

nep_time_cmap = plt.get_cmap('BuPu_r')
nep_theme_color = 'teal'
nep_idate = 8643

wau_time_cmap = plt.get_cmap('OrRd_r')
wau_theme_color = 'maroon'
wau_idate = 6999

mean_time_cmap = plt.get_cmap('YlGn_r')

s_cmap = plt.get_cmap('PuOr_r')
term_cmap = balance
term_cmap_r = cmocean.cm.balance_r

# a_palette5 = ["#df2935","#86ba90","#f5f3bb","#dfa06e","#412722"]
# a_palette5 = ["#1b4079","#4d7c8a","#7f9c96","#8fad88","#cbdf90"]
a_palette5 = ["#003049","#61988e","#f77f00","#7d1538","#8390fa"]
region_names =['gulf','labr','gdbk','nace','egrl']
region_longnames = ['Gulf Stream','Labrador Current','Grand Bank','NAC Extension','East Greenland Current']
region_longnames = dict(zip(region_names, region_longnames))
region_colors = dict(zip(region_names,a_palette5))

# rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
rhs_list = ['A','F','dif_v','dif_h','E','e_ua','I']
term_colors = ['#fc8d62','#66c2a5','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494']
color_dic = dict(zip(rhs_list,term_colors))

error_color = 'r'

term_dic = {
    'A': r"$-u'\cdot \nabla \bar s$",
    'F': "Evap/Prec",
    'E': r"$-(u'\nabla s'-\overline{u'\nabla s'})$",
    # 'E': r"$\overline{u'\cdot \nabla s'}$",
    'dif_v': "Vertical Diffusion",
    'dif_h': "Horizontal diffusion",
    'e_ua': "Subdaily Advection",
    'I': "Surface salt flux and salt plume"
}
case_term_dic = {
    'A': r"$-u'\cdot \nabla \bar \theta$",
    'F': "Dilution",
    # 'E': r"$(-u'\nabla s'-\overline{u'\nabla s'})$",
    'E': r"$\overline{u'\cdot \nabla \theta'}$",
    'dif_v': "Vertical Diffusion",
    'dif_h': "Horizontal diffusion",
    'e_ua': "Subdaily Advection",
    'I': "q'"
}

TOTAL_VOLUME_nep,NUMBER_OF_PARTICLE_nep,VOLUME_EACH_nep = (137447274971136.0, 9992, 13755731968.0)
TOTAL_VOLUME_wau,NUMBER_OF_PARTICLE_wau,VOLUME_EACH_wau = (63566417756160.0,10013, 6348388864.0)

fill_betweenx_kwarg = dict(
    color = 'grey',
    alpha = 0.5
)