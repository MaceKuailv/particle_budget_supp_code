format = '.tiff'

water2color = {
    'lw':'#FEC601',
    'iw':'#5B8C5A',
    'dw':'#EA7317',
    'bw':'#3D3B8E'
}

term2longname = {
    'c_dif_h':'Horizontal diffusion',
    'c_dif_v':'Vertical diffusion',
    'total_bio':'Total biological activity',
    'cerror':'Surface correction',
    'cf':'DIC concentration',
    'cl':'DIC concentration',
    'diccarb': 'Calcium carbonate cycle',
    'dicpflx': 'Accumulation of remineralized phosphorus',
    'dicrdop': 'Remineralization of dissolved organic phosphorus',
    'dictflx': 'DIC exchange with the atmosphere',
    'e_ua_c':'Unresolved advection',
    'e_ua_p':'Unresolved advection',
    'forc_c':'Dilution',
    'forc_p':'Dilution',
    'negbio':'Net community uptake of phosphorus',
    'p_dif_h':'Horizontal diffusion',
    'p_dif_v':'Vertical diffusion',
    'perror':'Surface correction',
    'pf':'Phosphate Concentration',
    'pl':'Phosphate Concentration',
}

longname2color = {
    'Net community uptake of phosphorus':'#44CF6C',
    'Phosphate Concentration':'#B3001B',
    'Accumulation of remineralized phosphorus':'#8B6220',
    'Vertical diffusion':'#7D84B2',
    'Total biological activity':'#717744',
    'Remineralization of dissolved organic phosphorus':'#A5BE00',
    'Dilution':'#2EC4B6',
    'Surface correction':'#DB5461',
    'Calcium carbonate cycle':'#CCAD8F',
    'Unresolved advection':'#C75000',
    'Horizontal diffusion':'#427AA1',
    'DIC exchange with the atmosphere':'#E7E247',
    'DIC concentration':'g',
}
term2color = {}
for term in term2longname.keys():
    term2color[term] = longname2color[term2longname[term]]

number = [1084,1898, 2864, 4350]
water = ['bw','dw','iw','lw']
water2number = dict(zip(water,number))

crhs_list = ['e_ua_c','c_dif_h','c_dif_v','diccarb','dictflx','forc_c','total_bio']
prhs_list = ['e_ua_p','p_dif_h','p_dif_v','dicrdop','dicpflx','forc_p','negbio']