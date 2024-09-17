import numpy as np
FCN_exp3_025 = np.array([
6.629E-05,
6.551E-05,
6.482E-05,
6.523E-05])

FCN_exp3_050 = np.array([
6.580E-05,
6.516E-05,
6.452E-05,
6.486E-05])

FCN_exp3_100 = np.array([
6.536E-05,
6.479E-05,
6.415E-05,
6.453E-05])

LRN_exp3_025 = np.array([
6.846E-05,
6.834E-05,
6.735E-05,
6.749E-05
])

LRN_exp3_050 = np.array([
6.737E-05,
6.675E-05,
6.588E-05,
6.621E-05
])

LRN_exp3_100 = np.array([
6.640E-05,
6.588E-05,
6.489E-05,
6.539E-05
])

Our_exp3_025 = np.array([
6.589E-05,
6.520E-05,
6.457E-05,
6.480E-05
])

Our_exp3_050 = np.array([
6.547E-05,
6.488E-05,
6.426E-05,
6.435E-05
])

Our_exp3_100 = np.array([
6.498E-05,
6.466E-05,
6.400E-05,
6.420E-05
])

# print("FCN_exp3_025: {:.2E} {:.2E}".format(np.mean(FCN_exp3_025), np.std(FCN_exp3_025)))
# print("FCN_exp3_050: {:.2E} {:.2E}".format(np.mean(FCN_exp3_050), np.std(FCN_exp3_050)))
# print("FCN_exp3_100: {:.2E} {:.2E}".format(np.mean(FCN_exp3_100), np.std(FCN_exp3_100)))
# print("LRN_exp3_025: {:.2E} {:.2E}".format(np.mean(LRN_exp3_025), np.std(LRN_exp3_025)))
# print("LRN_exp3_050: {:.2E} {:.2E}".format(np.mean(LRN_exp3_050), np.std(LRN_exp3_050)))
# print("LRN_exp3_100: {:.2E} {:.2E}".format(np.mean(LRN_exp3_100), np.std(LRN_exp3_100)))
# print("Our_exp3_025: {:.2E} {:.2E}".format(np.mean(Our_exp3_025), np.std(Our_exp3_025)))
# print("Our_exp3_050: {:.2E} {:.2E}".format(np.mean(Our_exp3_050), np.std(Our_exp3_050)))
# print("Our_exp3_100: {:.2E} {:.2E}".format(np.mean(Our_exp3_100), np.std(Our_exp3_100)))

FCN_lipsdp = np.array([
1.1927, 
1.4283,
1.8546,
1.0504
])

FCN_grid_010 = np.array([
0.4425,
0.5119,
0.6376,
0.4000
])

FCN_grid_005 = np.array([
0.3942,
0.4544,
0.5616,
0.3571
])

LRN_lipsdp = np.array([
4.5805,
3.2194,
4.3388,
3.6461
])

LRN_grid_010 = np.array([
1.4403,
1.0389,
1.3686,
1.1655
])

LRN_grid_005 = np.array([
1.2617,
0.9106,
1.1989,
1.0189
])

Our_lip = np.array([
0.9756,
0.9756,
0.9756,
0.9756
])

Our_grid_010 = np.array([
0.3790,
0.3790,
0.3785,
0.3784
])

Our_grid_005 = np.array([
0.3383,
0.3382,
0.3382,
0.3384
])

# print("FCN_lipsdp: {:.2f} {:.2f}".format(np.mean(FCN_lipsdp), np.std(FCN_lipsdp)))
# print("FCN_grid_010: {:.2f} {:.2f}".format(np.mean(FCN_grid_010), np.std(FCN_grid_010)))
# print("FCN_grid_005: {:.2f} {:.2f}".format(np.mean(FCN_grid_005), np.std(FCN_grid_005)))
# print("LRN_lipsdp: {:.2f} {:.2f}".format(np.mean(LRN_lipsdp), np.std(LRN_lipsdp)))
# print("LRN_grid_010: {:.2f} {:.2f}".format(np.mean(LRN_grid_010), np.std(LRN_grid_010)))
# print("LRN_grid_005: {:.2f} {:.2f}".format(np.mean(LRN_grid_005), np.std(LRN_grid_005)))
print("Our_lipsdp: {:.2f} {:.3f}".format(np.mean(Our_lip), np.std(Our_lip)))
print("Our_grid_010: {:.2f} {:.4f}".format(np.mean(Our_grid_010), np.std(Our_grid_010)))
print("Our_grid_005: {:.2f} {:.4f}".format(np.mean(Our_grid_005), np.std(Our_grid_005)))