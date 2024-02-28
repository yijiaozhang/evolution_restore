protein_net_types_mirrorTn = ["fungi_mirrorTn", "human_mirrorTn", "fruit_fly_mirrorTn", "worm_mirrorTn", "bacteria_mirrorTn"]
protein_net_names_mirrorTn = ["fungi_mirrorTn%4", "human_mirrorTn%7", "fruit_fly_mirrorTn%5", "worm_mirrorTn%4", "bacteria_mirrorTn%2"]
coauthor_net_names_max_connect = ["chaos_maxconnect%", "complex networks_maxconnect%"]
coauthor_net_names = ["fluctuations%", "interfaces%", "phase_transitions%", "thermodynamics%", "complex_networks%", "chaos_new%"]
economy_net_names = ["WTW%"]
interaction_net_names = ["weaver%", "ants%"]
transport_net_names = ["Air%", "Coach%", "Ferry%"]

ba_net_names = ["BA_2k%", "BA_2k_cut3%", "BA_2k_cut5%", "BA_2k_cut7%"]
ba_model_nets = ["BA13%",  "BA15%"]
pso_net_names = ["PSO21%", "PSO22%"]
fitness_net_names = ["Fitness14%", "Fitness15%"]
gbg_net_names = ["gbg_0.2_time5%", "gbg_0.5_time3%"]

coauthor_net_names_lp_svd = ["fluctuations_lp%", "interfaces_lp%", "phase_transitions_lp%", "thermodynamics_lp%", "complex networks_maxconnect_lp%", "chaos_new_lp%"]
protein_net_names_mirrorTn_lp_svd = ["fungi_mirrorTn_lp%", "human_mirrorTn_lp%", "fruit_fly_mirrorTn_lp%", "worm_mirrorTn_lp%", "bacteria_mirrorTn_lp%"]
interaction_net_names_lp_svd = ["weaver_lp%", "ants_lp%"]
transport_net_names_lp_svd = ["Air_lp%", "Coach_lp%", "Ferry_lp%"]
economy_net_names_lp_svd = ["WTW_lp%"]


EDGE_FEATURE = ['bn', 'cn', 'degree', 'strength', 'cc', 'ra', 'aa', 'pa', 'lp', 'k_shell', 'pr']
# judge method name
BEST_SINGLE = 'best_single'
BN_PAIR_NN = "bn"
CN_PAIR_NN = "cn"
DEGREE_PAIR_NN = "degree"
STRENGTH_PAIR_NN = "strength"
CC_PAIR_NN = "cc"
RA_PAIR_NN = "ra"
AA_PAIR_NN = "aa"
PA_PAIR_NN = "pa"
LP_PAIR_NN = "lp"
KSHELL_PAIR_NN = "k_shell"
PR_PAIR_NN = "pr"

NODE2VEC_PAIR_NN = 'node2vec'
UNION_PAIR_NN = 'collection'
LINE_PAIR_NN = "line"
STRUC2VEC_PAIR_NN = "struct2vec"
DEEPWALK_PAIR_NN = "deepwalk"
SDNE_PAIR_NN = "sdne"
ENSEMBLE = 'ensemble'
JUDGE_METHOD = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, ENSEMBLE]

# embedding name
SINGLE = "single_feature"
NODE2VEC = "node2vec_hadamard"
COLLECTION = "simple"
LINE_EMBEDDING = "line"
STRUCT2VEC_EMBEDDING = "struct2vec"
DEEPWALK_EMBEDDING = "deepwalk"
SDNE_EMBEDDING = "sdne"

