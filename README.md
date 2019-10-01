# Music-Genre-Transfer-with-Deep-Learning

- This project follows sumu's Cyclegan project https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer

- Two extra methods are introduced: Spectral Normalization and Self-Attention GANs. Their effects are tested.

- an alternative genre classifier can also be used, which is a multi-type classifier that directly classify the three genres instead of a binary classification.

- original datas are here: https://goo.gl/ZK8wLW

- some result samples: http://bit.ly/31VnTxS

## Training data
The directory 'notebook' contains files that generate the training data and do some data statistics.
To generate or manipulate the training data, please follow the notebook 'generate_npy_from_midi.ipynb'

## Models
The directories 'cyclegan_both', 'cyclegan_sa', 'cyclegan_sn' and 'cyclegan_vanilla' are corresponding to cyclegan model with Self-Attention GAN and Spectral Normalization, cyclegan model with Self-Attention GAN, cyclegna model with Spectral Normalization, and the vanilla cyclegan model.

For cyclegan model details, please see sumu's Cyclegan project https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer

[Self-Attention GAN](https://arxiv.org/abs/1805.08318)

[Spctral Normalization](https://arxiv.org/abs/1802.05957)


