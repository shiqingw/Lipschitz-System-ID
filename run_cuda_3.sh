mkdir eg3_results/060
python -u eg3_TwoLinkArm/train.py --exp_num 60 --device cuda:3 > eg3_results/060/output.out
mkdir eg3_results/064
python -u eg3_TwoLinkArm/train.py --exp_num 64 --device cuda:3 > eg3_results/064/output.out
mkdir eg3_results/140
python -u eg3_TwoLinkArm/train.py --exp_num 140 --device cuda:3 > eg3_results/140/output.out
mkdir eg3_results/144
python -u eg3_TwoLinkArm/train.py --exp_num 144 --device cuda:3 > eg3_results/144/output.out
