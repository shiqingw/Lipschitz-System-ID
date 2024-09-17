import numpy as np
FCN_exp1_025 = np.array([
5.506E-03,
5.494E-03,
5.483E-03,
5.412E-03])

FCN_exp1_050 = np.array([
5.437E-03,
5.425E-03,
5.430E-03,
5.367E-03])

FCN_exp1_100 = np.array([
5.423E-03,
5.422E-03,
5.425E-03,
5.355E-03])

LRN_exp1_025 = np.array([
7.718E-03,
7.819E-03,
7.668E-03,
8.903E-03
])

LRN_exp1_050 = np.array([
6.279E-03,
6.259E-03,
6.178E-03,
6.412E-03
])

LRN_exp1_100 = np.array([
5.769E-03,
5.791E-03,
5.724E-03,
5.785E-03
])

Our_exp1_025 = np.array([
5.454E-03,
5.458E-03,
5.459E-03,
5.376E-03
])

Our_exp1_050 = np.array([
5.423E-03,
5.421E-03,
5.428E-03,
5.335E-03
])

Our_exp1_100 = np.array([
5.422E-03,
5.419E-03,
5.418E-03,
5.352E-03
])

# print("FCN_exp1_025: {:.2E} {:.2E}".format(np.mean(FCN_exp1_025), np.std(FCN_exp1_025)))
# print("FCN_exp1_050: {:.2E} {:.2E}".format(np.mean(FCN_exp1_050), np.std(FCN_exp1_050)))
# print("FCN_exp1_100: {:.2E} {:.2E}".format(np.mean(FCN_exp1_100), np.std(FCN_exp1_100)))
# print("LRN_exp1_025: {:.2E} {:.2E}".format(np.mean(LRN_exp1_025), np.std(LRN_exp1_025)))
# print("LRN_exp1_050: {:.2E} {:.2E}".format(np.mean(LRN_exp1_050), np.std(LRN_exp1_050)))
# print("LRN_exp1_100: {:.2E} {:.2E}".format(np.mean(LRN_exp1_100), np.std(LRN_exp1_100)))
# print("Our_exp1_025: {:.2E} {:.2E}".format(np.mean(Our_exp1_025), np.std(Our_exp1_025)))
# print("Our_exp1_050: {:.2E} {:.2E}".format(np.mean(Our_exp1_050), np.std(Our_exp1_050)))
# print("Our_exp1_100: {:.2E} {:.2E}".format(np.mean(Our_exp1_100), np.std(Our_exp1_100)))

FCN_lipsdp = np.array([
22.6307,
25.0971,
15.0283,
12.8966
])

FCN_grid_010 = np.array([
3.2786,
3.6068,
2.2834,
2.0072
])

FCN_grid_005 = np.array([
2.7687,
3.0367,
1.9305,
1.6931
])

LRN_lipsdp = np.array([
11.5212,
14.5223,
18.2131,
12.4996
])

LRN_grid_010 = np.array([
1.8227,
2.2362,
2.7104,
1.9884
])

LRN_grid_005 = np.array([
1.5683,
1.8669,
2.2961,
1.6533
])

Our_lip = np.array([
2.0105,
2.0105,
2.0105,
2.0105
])

Our_grid_010 = np.array([
0.6401,
0.6377,
0.6386,
0.6382
])

Our_grid_005 = np.array([
0.5027,
0.5065,
0.5046,
0.5056
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