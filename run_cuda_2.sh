mkdir eg1_results/003
python -u eg1_Linear/train.py --exp_num 3 --device cuda:2 > eg1_results/003/output.out
mkdir eg1_results/007
python -u eg1_Linear/train.py --exp_num 7 --device cuda:2 > eg1_results/007/output.out
mkdir eg1_results/011
python -u eg1_Linear/train.py --exp_num 11 --device cuda:2 > eg1_results/011/output.out
