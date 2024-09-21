import numpy as np
FCN_exp3_025 = np.array([
2.726E-03,
2.698E-03,
2.703E-03,
2.723E-03])

FCN_exp3_050 = np.array([
2.711E-03,
2.692E-03,
2.691E-03,
2.711E-03])

FCN_exp3_100 = np.array([
2.693E-03,
2.670E-03,
2.674E-03,
2.691E-03])

LRN_exp3_025 = np.array([
2.748E-03,
2.725E-03,
2.726E-03,
2.743E-03
])

LRN_exp3_050 = np.array([
2.734E-03,
2.712E-03,
2.715E-03,
2.730E-03
])

LRN_exp3_100 = np.array([
2.723E-03,
2.702E-03,
2.705E-03,
2.721E-03
])

Our_exp3_025 = np.array([
2.717E-03,
2.686E-03,
2.696E-03,
2.713E-03
])

Our_exp3_050 = np.array([
2.703E-03,
2.678E-03,
2.689E-03,
2.703E-03
])

Our_exp3_100 = np.array([
2.685E-03,
2.670E-03,
2.679E-03,
2.688E-03
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
7.65,
19.64,
19.96,
13.46
])

FCN_grid_010 = np.array([
2.47,
6.03,
6.12,
4.19
])

FCN_grid_005 = np.array([
2.15,
5.27,
5.35,
3.66
])

LRN_lipsdp = np.array([
11.05,
11.23,
11.46,
9.83
])

LRN_grid_010 = np.array([
3.48,
3.53,
3.60,
3.12
])

LRN_grid_005 = np.array([
3.04,
3.09,
3.14,
2.72
])

Our_lip = np.array([
2.55,
2.55,
2.55,
2.55
])

Our_grid_010 = np.array([
0.947,
0.946,
0.949,
0.950
])

Our_grid_005 = np.array([
0.837,
0.834,
0.834,
0.834
])

# print("FCN_lipsdp: {:.2f} {:.2f}".format(np.mean(FCN_lipsdp), np.std(FCN_lipsdp)))
# print("FCN_grid_010: {:.2f} {:.2f}".format(np.mean(FCN_grid_010), np.std(FCN_grid_010)))
# print("FCN_grid_005: {:.2f} {:.2f}".format(np.mean(FCN_grid_005), np.std(FCN_grid_005)))
# print("LRN_lipsdp: {:.2f} {:.2f}".format(np.mean(LRN_lipsdp), np.std(LRN_lipsdp)))
# print("LRN_grid_010: {:.2f} {:.2f}".format(np.mean(LRN_grid_010), np.std(LRN_grid_010)))
# print("LRN_grid_005: {:.2f} {:.2f}".format(np.mean(LRN_grid_005), np.std(LRN_grid_005)))
print("Our_lipsdp: {:.2f} {:.3f}".format(np.mean(Our_lip), np.std(Our_lip)))
print("Our_grid_010: {:.2f} {:.5f}".format(np.mean(Our_grid_010), np.std(Our_grid_010)))
print("Our_grid_005: {:.2f} {:.4f}".format(np.mean(Our_grid_005), np.std(Our_grid_005)))