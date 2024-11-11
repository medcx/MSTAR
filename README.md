# Classification-on-MSTAR
This is a codebase for classification on MSTAR

**Download dataset**

To get start, you must download the mstar dataset from [MSTAR](https://drive.google.com/file/d/1Mzt4Cjq1MvdIA6HVxfAgMVibO1rpzZFb/view?usp=sharing).

Then, save it to ```dataset``` folder.

**Training process**

Please run the following code to train the model.

```
python main.py --data_path ./dataset --epochs 300 --use_gpu True --train
```

**Test process**

Please run the following code to test on MSTAR.

```
python main.py --data_path ./dataset --use_gpu True --checkpoint ./checkpoint/model_best.pth 
```
