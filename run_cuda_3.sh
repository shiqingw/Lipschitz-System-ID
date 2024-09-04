mkdir eg1_results/004
python -u eg1_Linear/train.py --exp_num 4 --device cuda:3 > eg1_results/004/output.out
mkdir eg1_results/008
python -u eg1_Linear/train.py --exp_num 8 --device cuda:3 > eg1_results/008/output.out
mkdir eg1_results/012
python -u eg1_Linear/train.py --exp_num 12 --device cuda:3 > eg1_results/012/output.out
