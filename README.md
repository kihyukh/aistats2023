# Single switch kernel bandit experiment

python run.py --config config/switching_kernel_bandit_opkb.yaml --seed 0
python run.py --config config/switching_kernel_bandit_gpucb.yaml --seed 0
python run.py --config config/switching_kernel_bandit_gpucb_discount.yaml --seed 0
python run.py --config config/switching_kernel_bandit_opnn.yaml --seed 0

# Two switches kernel bandit experiment

python run.py --config config/two_switching_kernel_bandit_opkb.yaml --seed 0
python run.py --config config/two_switching_kernel_bandit_gpucb.yaml --seed 0
python run.py --config config/two_switching_kernel_bandit_gpucb_discount.yaml --seed 0
python run.py --config config/two_switching_kernel_bandit_opnn.yaml --seed 0

# Stationary cosine bandit experiment

python run.py --config config/stationary_cosine_bandit_opkb.yaml --seed 0
python run.py --config config/stationary_cosine_bandit_gpucb.yaml --seed 0
python run.py --config config/stationary_cosine_bandit_opnn.yaml --seed 0
python run.py --config config/stationary_cosine_bandit_opnn0.yaml --seed 0

# Slowly-varying cosine bandit experiment

python run.py --config config/slowly_varying_bandit_opkb.yaml --seed 0
python run.py --config config/slowly_varying_bandit_gpucb.yaml --seed 0
python run.py --config config/slowly_varying_bandit_gpucb_discount.yaml --seed 0
python run.py --config config/slowly_varying_bandit_opnn.yaml --seed 0
