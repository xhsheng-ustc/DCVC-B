def get_coding_structure_RA_test():
    coding_order_list = [0,           32,           16,          8,         4,          2,         1,         3,  ]+\
                        [6,           5,            7,           12,        10,         9,         11,        14, ]+\
                        [13,          15,           24,          20,        18,         17,        19,        22, ]+\
                        [21,          23,           28,          26,        25,         27,        30,        29, ]+\
                        [31,          64,           48,          40,        36,         34,        33,        35, ]+\
                        [38,          37,           39,          44,        42,         41,        43,        46, ]+\
                        [45,          47,           56,          52,        50,         49,        51,        54, ]+\
                        [53,          55,           60,          58,        57,         59,        62,        61, ]+\
                        [63,          80,           72,          68,        66,         65,        67,        70, ]+\
                        [69,          71,           76,          74,        73,         75,        78,        77, ]+\
                        [79,          88,           84,          82,        81,         83,        86,        85, ]+\
                        [87,          92,           90,          89,        91,         94,        93,        95  ]
    forward_ref_lists=  [[None],      [None],       [0],        [0],        [0],        [0],       [0],       [2], ]+\
                        [[4],         [4],          [6],        [8],        [8],        [8],       [10],      [12],]+\
                        [[12],        [14],         [16],       [16],       [16],       [16],      [18],      [20],]+\
                        [[20],        [22],         [24],       [24],       [24],       [26],      [28],      [28],]+\
                        [[30],        [None],       [32],       [32],       [32],       [32],      [32],      [34], ]+\
                        [[36],        [36],         [38],       [40],       [40],       [40],      [42],      [44], ]+\
                        [[44],        [46],         [48],       [48],       [48],       [48],      [50],      [52], ]+\
                        [[52],        [54],         [56],       [56],       [56],       [58],      [60],      [60], ]+\
                        [[62],        [64],         [64],       [64],       [64],       [64],      [66],      [68], ]+\
                        [[68],        [70],         [72],       [72],       [72],       [74],      [76],      [76], ]+\
                        [[78],        [80],         [80],       [80],       [80],       [82],      [84],      [84], ]+\
                        [[86],        [88],         [88],       [88],       [90],       [92],      [92],      [94]  ]

    backword_ref_lists= [[None],      [None],       [32],       [16],       [8],        [4],       [2],       [4],  ]+\
                        [[8],         [6],          [8],        [16],       [12],       [10],      [12],      [16], ]+\
                        [[14],        [16],         [32],       [24],       [20],       [18],      [20],      [24], ]+\
                        [[22],        [24],         [32],       [28],       [26],       [28],      [32],      [30], ]+\
                        [[32],        [None],       [64],       [48],       [40],       [36],      [34],      [36], ]+\
                        [[40],        [38],         [40],       [48],       [44],       [42],      [44],      [48], ]+\
                        [[46],        [48],         [64],       [56],       [52],       [50],      [52],      [56], ]+\
                        [[54],        [56],         [64],       [60],       [58],       [60],      [64],      [62], ]+\
                        [[64],        [64],         [80],       [72],       [68],       [66],      [68],      [72], ]+\
                        [[70],        [72],         [80],       [76],       [74],       [76],      [80],      [78], ]+\
                        [[80],        [64],         [88],       [84],       [82],       [84],      [88],      [86], ]+\
                        [[88],        [80],         [92],       [90],       [92],       [80],      [94],      [92]  ]

    layer_ids =         [0,           0,            1,          2,          3,          4,         5,         5, ]+\
                        [4,           5,            5,          3,          4,          5,         5,         4, ]+\
                        [5,           5,            2,          3,          4,          5,         5,         4, ]+\
                        [5,           5,            3,          4,          5,          5,         4,         5, ]+\
                        [5,           0,            1,          2,          3,          4,         5,         5, ]+\
                        [4,           5,            5,          3,          4,          5,         5,         4, ]+\
                        [5,           5,            2,          3,          4,          5,         5,         4, ]+\
                        [5,           5,            3,          4,          5,          5,         4,         5, ]+\
                        [5,           1,            2,          3,          4,          5,         5,         4, ]+\
                        [5,           5,            3,          4,          5,          5,         4,         5, ]+\
                        [5,           2,            3,          4,          5,          5,         4,         5, ]+\
                        [5,           3,            4,          5,          5,          4,         5,         5]
    return coding_order_list, forward_ref_lists, backword_ref_lists, layer_ids

def empty_memory(dpb_list, fa_idx):
    if fa_idx==6:
        for i in range(0,4):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==12:
        for i in range(4,8):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==14:
        for i in range(8,12):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==24:
        for i in range(12,16):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==22:
        for i in range(16,20):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==28:
        for i in range(20,24):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==30:
        for i in range(24,28):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==64:
        for i in range(28,32):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==35:
        for i in range(32,24):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==38:
        for i in range(34,36):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==39:
        for i in range(36,38):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==44:
        for i in range(38,40):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==43:
        for i in range(40,42):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==46:
        for i in range(42,44):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==47:
        for i in range(44,46):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==56:
        for i in range(46,48):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==51:
        for i in range(48,50):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==54:
        for i in range(50,52):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==55:
        for i in range(52,54):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==60:
        for i in range(54,56):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==59:
        for i in range(56,58):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==62:
        for i in range(58,60):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==63:
        for i in range(60,62):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==80:
        for i in range(62,64):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==67:
        for i in range(65,66):######keep 64
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==70:
        for i in range(66,68):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==71:
        for i in range(68,70):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==76:
        for i in range(70,72):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==75:
        for i in range(72,74):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==78:
        for i in range(74,76):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==79:
        for i in range(76,78):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==88:
        for i in range(78,80):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==83:
        for i in range(81,82):#####keep 80
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==85:
        for i in range(82,84):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==87:
        for i in range(84,86):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==92:
        for i in range(86,88):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==91:
        for i in range(88,90):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    elif fa_idx==93:
        for i in range(90,92):
            dpb_list[i]={
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
    return dpb_list