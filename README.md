# Adversairial Skil Learning for Robust Manipulation
Author: Pingcheng Jian, Chao Yang, Di Guo, Huaping Liu, Fuchun Sun

We provide the code for *Adversairial Skil Learning for Robust Manipulation* in this repository.
## Requirements
- This codebase requires [Robel](https://github.com/google-research/robel) and other dependencies in your conda environment with pip:
``` 
$ pip install -e .
```
- if you want to install some extra development tools, using:

``` 
$ pip install -e .[dev]
```

## Instruction to train
1. train the DoublePick-v1
```
python train/train.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --cuda True --seed 1
```

2. train the protagonist and adversary in DClawTurnFixed-v0
```
python train/adversarial_robel_train.py
```

3. train normal SAC policy in DClawTurnFixed-v0
```
python train/robel_train.py
```

## Instruction to test
1. test protagonist against adversary in DoublePick-v1
```
python test/adversarial_test_double_pick.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --seed 121
```

2. test normal SAC policy against adversary in DoublePick-v1
```
python test/adversarial_test_double_pick_normal.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --seed 121
```

3. test protagonist or normal SAC policy against random attack in DoublePick-v1
```
python test/adversarial_test_double_pick_random_noise.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --seed 121
```

4. test protagonist against adversary in DClawTurnFixed-v0
```
python test/ad_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent robust
```

5. test normal SAC policy against adversary in DClawTurnFixed-v0
```
python test/ad_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent normal
```

6. test protagonist against random noise in DClawTurnFixed-v0
```
python test/ad_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent robust
```

7. test normal SAC policy against random noise in DClawTurnFixed-v0
```
python test/ad_noise_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent normal
```
