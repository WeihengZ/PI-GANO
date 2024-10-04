# PI-GANO

```
cd Main
python exp_pinn_darcy.py --model='self_defined' --phase='train'
python exp_pinn_plate.py --model='self_defined' --phase='train'
```

If you are interested in developing more advanced training algorithms, please check our the script "darcy_utils.py" and "plate_utils.py".

**If you think that the work of the PI-DCON is useful in your research, please consider citing our paper in your manuscript:**
```
@article{zhong2024physics,
  title={Physics-informed discretization-independent deep compositional operator network},
  author={Zhong, Weiheng and Meidani, Hadi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117274},
  year={2024},
  publisher={Elsevier}
}
```