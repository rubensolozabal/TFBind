# Bases
COMPLEMENT_BASE = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} 
NATURAL_BASES={'A', 'C', 'T', 'G'}
NATURAL_PAIRS={'AT', 'TA', 'CG', 'GC'}

# Visualization
COLOR_NUC = {'T': 'red',
            'C': 'blue',
            'G': 'gold',
            'A': 'grey'}

COLOR_FG = {'A': (1.0, 0.0, 0.0),  # red
            'D': (0.0, 0.0, 1.0),  # blue
            'M': (1.0, 1.0, 0.0),  # yellow
            'n': (0.7, 0.7, 0.7),  # grey
            's': (1.0, 1.0, 1.0),  # white
            'x': (0.0, 0.0, 0.0)}  # black

COLOR_MODS = {
    "none": "#1f77b4",
    "I": "#17becf",
    "D": "#7f7f7f",
    "6mA": "#e377c2",
    "7dA": "#ff7f0e",
    "7dG": "#d62728",
    "dUPT": "#9467bd",
    "5mC": "#8c564b",
    "both": "#bcbd22",
}

# Encoding functional groups
# A- Acceptor, D- Donor, M- Methyl, n- non-polar, x- empty 
# Define the groups in Major Groove:
MG_funcGroups= {
    'A': ['A','D'],
    'T': ['M','A'],
    'C': ['n','D'],
    'G': ['A','A'],
    'U': ['n','A'], # U
    'X': ['M','D'], # 5mC
    'a': ['n','D'], # 7dA
    'g': ['n','A'], # 7dG
    'I': ['A','A'], # I
    'M': ['A','M'], # 6mA
    'D' : ['x','x'] # D
}

# and in minor Groove:
mG_funcGroups= {
    'A': ['A','n'],
    'T': ['x','A'],
    'C': ['x','A'],
    'G': ['A','D'],
    'U': ['x','A'], # U
    'X': ['x','A'], # 5mC
    'a': ['A','n'], # 7dA
    'g': ['A','D'], # 7dG
    'I': ['A','n'], # I
    'M': ['A','n'], # 6mA
    'D': ['x','x'], # D
}

fg_encode_map = {
    'x': [0, 0, 0, 0],
    'A': [0, 0, 0, 1],
    'D': [0, 0, 1, 0],
    'M': [0, 1, 0, 0],
    'N': [1, 0, 0, 0],
}


# Proteins
ETS1_LEN = 13
ETS1_PLUS_STRAND  = "GTGCCGGAAATGT"
ETS1_MINUS_STRAND = "ATTTCCGGCACTA"


MITF_LEN = 14
MITF_PLUS_STRAND  = "GTATCACGTGATAC"
MITF_MINUS_STRAND = "GTATCACGTGATAC"

EGR1_LEN = 11
EGR1_PLUS_STRAND  = "AGCGTGGGCAC"
EGR1_MINUS_STRAND = "GTGCCCACGCT"
