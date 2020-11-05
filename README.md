# Adversairial Skil Learning for Robust Manipulation
This repository implements the code of ICRA2021 paper: Adversairial Skil Learning for Robust Manipulation

# Requirements
- run it directly in your python 3 environment with pip:

``` 
$ pip install -e .
```
- if you want to install soem extra development tools, using:

``` 
$ pip install -e .[dev]
```

# Instruction to run the code
1. train the DoublePick-v1
```
python train.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --cuda True --seed 1
```

2. train the protagonist and adversary in DClawTurnFiwed-v0
```
python adversarial_robel_train.py
```

3. train normal SAC policy in DClawTurnFiwed-v0
```
python robel_train.py
```

4. test protagonist against adversary in DoublePick-v1
```
python adversarial_test_double_pick.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --seed 121
```

5. test normal SAC policy against adversary in DoublePick-v1
```
python adversarial_test_double_pick_normal.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --seed 121
```

5. test protagonist or normal SAC policy against random attack in DoublePick-v1
```
python adversarial_test_double_pick_random_noise.py --algo adversarial_double_pick_sac --env-name DoublePick-v1 --seed 121
```

6. test protagonist against adversary in DClawTurnFiwed-v0
```
python ad_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent robust
```

6. test normal SAC policy against adversary in DClawTurnFiwed-v0
```
python ad_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent normal
```

6. test protagonist against random noise in DClawTurnFiwed-v0
```
python ad_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent robust
```

6. test normal SAC policy against random noise in DClawTurnFiwed-v0
```
python ad_noise_test_robel.py --ad_factor 0.5 --env-name DClawTurnFixed-v0 --po_agent normal
```