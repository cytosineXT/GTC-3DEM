# install
```bash
git clone https://github.com/cytosineXT/GTC-3DEM.git
cd ./GTC-3DEM

conda create -n 3DEM python=3.9 -y
conda activate 3DEM
pip install -r requirements.txt

```

# train
```bash
python NNtrainGNN_arg4fold.py --cuda 'cpu'
```

when counter error about "RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.":


"/home/your_user_name/anaconda3/envs/3DEM/lib/python3.9/site-packages/torch/_tensor.py", line 1087

         if dtype is None:
             return self.numpy()
         else:
             return self.numpy().astype(dtype, copy=False)

modify to

         if dtype is None:
             return self.cpu().detach().numpy()
         else:
             return self.cpu().detach().numpy().astype(dtype, copy=False)

# inference
```bash
python NNval_GNN4foldbatch.py --pretrainweight 'output/testtrain/your_train_fold/your_trained_weight.pt'

```