import numpy as np
FCN_exp2_025 = np.array([
2.722E-03,
2.705E-03,
2.716E-03,
2.714E-03])

FCN_exp2_050 = np.array([
2.697E-03,
2.683E-03,
2.694E-03,
2.683E-03])

FCN_exp2_100 = np.array([
2.689E-03,
2.670E-03,
2.680E-03,
2.671E-03])

LRN_exp2_025 = np.array([
2.882E-03,
3.156E-03,
3.061E-03,
2.961E-03
])

LRN_exp2_050 = np.array([
2.773E-03,
2.792E-03,
2.828E-03,
2.768E-03
])

LRN_exp2_100 = np.array([
2.728E-03,
2.717E-03,
2.716E-03,
2.697E-03
])

Our_exp2_025 = np.array([
2.700E-03,
2.674E-03,
2.695E-03,
2.675E-03
])

Our_exp2_050 = np.array([
2.694E-03,
2.662E-03,
2.679E-03,
2.670E-03
])

Our_exp2_100 = np.array([
2.689E-03,
2.661E-03,
2.672E-03,
2.666E-03
])

# print("FCN_exp2_025: {:.2E} {:.2E}".format(np.mean(FCN_exp2_025), np.std(FCN_exp2_025)))
# print("FCN_exp2_050: {:.2E} {:.2E}".format(np.mean(FCN_exp2_050), np.std(FCN_exp2_050)))
# print("FCN_exp2_100: {:.2E} {:.2E}".format(np.mean(FCN_exp2_100), np.std(FCN_exp2_100)))
# print("LRN_exp2_025: {:.2E} {:.2E}".format(np.mean(LRN_exp2_025), np.std(LRN_exp2_025)))
# print("LRN_exp2_050: {:.2E} {:.2E}".format(np.mean(LRN_exp2_050), np.std(LRN_exp2_050)))
# print("LRN_exp2_100: {:.2E} {:.2E}".format(np.mean(LRN_exp2_100), np.std(LRN_exp2_100)))
# print("Our_exp2_025: {:.2E} {:.2E}".format(np.mean(Our_exp2_025), np.std(Our_exp2_025)))
# print("Our_exp2_050: {:.2E} {:.2E}".format(np.mean(Our_exp2_050), np.std(Our_exp2_050)))
# print("Our_exp2_100: {:.2E} {:.2E}".format(np.mean(Our_exp2_100), np.std(Our_exp2_100)))

FCN_lipsdp = np.array([
5.3749,
4.9403,
5.3808,
5.3797
])

FCN_grid_010 = np.array([
1.1326,
1.0686,
1.1356,
1.1349
])

FCN_grid_005 = np.array([
1.1118,
1.0482,
1.1161,
1.1154
])

LRN_lipsdp = np.array([
15.9901,
16.3228,
16.3726,
19.9640
])

LRN_grid_010 = np.array([
2.8206,
2.8648,
2.8742,
3.4444
])

LRN_grid_005 = np.array([
2.7548,
2.8129,
2.8137,
3.3728
])

Our_lip = np.array([
4.0161,
4.0161,
4.0161,
4.0161
])

Our_grid_010 = np.array([
0.9174,
0.9178,
0.9177,
0.9186
])

Our_grid_005 = np.array([
0.8998,
0.9004,
0.9002,
0.9011
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