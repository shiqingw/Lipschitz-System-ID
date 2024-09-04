mkdir eg1_results/001
python -u eg1_Linear/train.py --exp_num 1 --device cuda:0 > eg1_results/001/output.out
mkdir eg1_results/005
python -u eg1_Linear/train.py --exp_num 5 --device cuda:0 > eg1_results/005/output.out
mkdir eg1_results/009
python -u eg1_Linear/train.py --exp_num 9 --device cuda:0 > eg1_results/009/output.out
