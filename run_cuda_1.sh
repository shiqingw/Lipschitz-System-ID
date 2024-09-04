mkdir eg1_results/002
python -u eg1_Linear/train.py --exp_num 2 --device cuda:1 > eg1_results/002/output.out
mkdir eg1_results/006
python -u eg1_Linear/train.py --exp_num 6 --device cuda:1 > eg1_results/006/output.out
mkdir eg1_results/010
python -u eg1_Linear/train.py --exp_num 10 --device cuda:1 > eg1_results/010/output.out
