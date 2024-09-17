import numpy as np
FCN_exp2_025 = np.array([
5.778E-04,
5.903E-04,
5.929E-04,
6.014E-04])

FCN_exp2_050 = np.array([
5.600E-04,
5.613E-04,
5.729E-04,
5.661E-04])

FCN_exp2_100 = np.array([
5.520E-04,
5.481E-04,
5.540E-04,
5.497E-04])

LRN_exp2_025 = np.array([
9.550E-04,
1.163E-03,
9.776E-04,
1.068E-03
])

LRN_exp2_050 = np.array([
6.662E-04,
6.875E-04,
6.773E-04,
6.874E-04
])

LRN_exp2_100 = np.array([
5.926E-04,
5.852E-04,
5.813E-04,
5.887E-04
])

Our_exp2_025 = np.array([
5.631E-04,
5.495E-04,
5.645E-04,
5.556E-04
])

Our_exp2_050 = np.array([
5.498E-04,
5.414E-04,
5.463E-04,
5.439E-04
])

Our_exp2_100 = np.array([
5.448E-04,
5.381E-04,
5.391E-04,
5.392E-04
])

# print("FCN_exp2_025: {:.2E} {:.2E}".format(np.mean(FCN_exp2_025), np.std(FCN_exp2_025)))
# print("FCN_exp2_050: {:.2E} {:.2E}".format(np.mean(FCN_exp2_050), np.std(FCN_exp2_050)))
# print("FCN_exp2_100: {:.2E} {:.2E}".format(np.mean(FCN_exp2_100), np.std(FCN_exp2_100)))
# print("LRN_exp2_025: {:.3E} {:.2E}".format(np.mean(LRN_exp2_025), np.std(LRN_exp2_025)))
# print("LRN_exp2_050: {:.2E} {:.2E}".format(np.mean(LRN_exp2_050), np.std(LRN_exp2_050)))
# print("LRN_exp2_100: {:.2E} {:.2E}".format(np.mean(LRN_exp2_100), np.std(LRN_exp2_100)))
# print("Our_exp2_025: {:.2E} {:.2E}".format(np.mean(Our_exp2_025), np.std(Our_exp2_025)))
# print("Our_exp2_050: {:.2E} {:.2E}".format(np.mean(Our_exp2_050), np.std(Our_exp2_050)))
# print("Our_exp2_100: {:.2E} {:.2E}".format(np.mean(Our_exp2_100), np.std(Our_exp2_100)))

FCN_lipsdp = np.array([
5.4962,
4.9369,
5.2492,
5.2828
])

FCN_grid_010 = np.array([
1.5252,
1.4145,
1.4751,
1.4833
])

FCN_grid_005 = np.array([
1.5219,
1.4095,
1.4738,
1.4806
])

LRN_lipsdp = np.array([
18.7946,
29.2469,
22.5555,
29.4673
])

LRN_grid_010 = np.array([
4.2326,
6.3649,
5.0031,
6.4185
])

LRN_grid_005 = np.array([
4.2326,
6.3649,
5.0031,
6.4185
])

Our_lip = np.array([
4.0275,
4.0275,
4.0275,
4.0275
])

Our_grid_010 = np.array([
1.2280,
1.2282,
1.2271,
1.2256
])

Our_grid_005 = np.array([
1.2239,
1.2239,
1.2242,
1.2230
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