# VAAL (ICCV 2019)

Original Pytorch implementation of "Variational Adversarial Active Learning". 

### Requirements
The required Python3 packages can be installed using 
```
pip3 install -r requirements.txt
```
The code was tested for Python 3.5 and 3.6. The code is compatible with either GPU or CPU, but GPU is highly recommended. 

### Experiments
The code can simply be run using 
```
python3 main.py
```
when trying with a different datasets or different variants, the main parameters to tune are
```
--adversary_param --beta --num_vae_steps and --num_adv_steps
```

### Citation

If you find this work useful, consider citing our work:
```
@article{sinha2019variational,
  title={Variational Adversarial Active Learning},
  author={Sinha, Samarth and Ebrahimi, Sayna and Darrell, Trevor},
  journal={arXiv preprint arXiv:1904.00370},
  year={2019}
}
```
# Contact
If there are any questions or concerns feel free to send a message at samarth.sinha@mail.utoronto.ca