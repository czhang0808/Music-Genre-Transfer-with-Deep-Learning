# Music-Genre-Transfer-with-Deep-Learning

- This project follows sumu's Cyclegan project https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer

- Two extra methods are introduced: [Spctral Normalization](https://arxiv.org/abs/1802.05957) and [Self-Attention GAN](https://arxiv.org/abs/1805.08318). Their effects are tested.

- An alternative genre classifier can also be used, which is a multi-type classifier that directly classify the three genres instead of a binary classification.

- Original datas are here: https://goo.gl/ZK8wLW

- Some result samples: http://bit.ly/31VnTxS

## Training data
The directory 'notebook' contains files that generate the training data and do some data statistics.  
To generate or manipulate the training data, please follow the notebook 'generate_npy_from_midi.ipynb'

## Models
The directories 'cyclegan_both', 'cyclegan_sa', 'cyclegan_sn' and 'cyclegan_vanilla' are corresponding to cyclegan model with Self-Attention GAN and Spectral Normalization, cyclegan model with Self-Attention GAN, cyclegna model with Spectral Normalization, and the vanilla cyclegan model.

For cyclegan model details, please see [sumu's Cyclegan project](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer)

## Pipeline
- Train a cyclegan model:
```bash
python main.py --dataset_A_dir='JC_J' --dataset_B_dir='JC_C' --model='base'
```
For dataset we have: 'JC', 'JP' and 'CP'.  
For models: 'base', 'partial' and 'full'. 

- Test the model with test data:
```bash
python main.py --dataset_A_dir='JC_J' --dataset_B_dir='JC_C' --model='base' --phase='test' --which_direction='AtoB'
```
'AtoB' and 'BtoA' can be set in 'which_direction'.

- Train a genre classifier:
```bash
python main.py --dataset_A_dir='JC_J' --dataset_B_dir='JC_C' --type='classifier'
```

- Use genre classifier to evaluate midi files generated from the trained cyclegan model:
```bash
python main.py --dataset_A_dir='JC_J' --dataset_B_dir='JC_C' --type='classifier' --model='base' --phase='test' --which_direction='AtoB'
