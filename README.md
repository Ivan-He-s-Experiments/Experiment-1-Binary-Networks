![image](https://github.com/user-attachments/assets/60a12a04-c4f0-4ab7-9fc8-6a8a7144043e)# Experiment-1-Binary-Networks

Youtube Video: https://www.youtube.com/watch?v=EdWQYypcFzc

## Installation

### CPU

1. Install reqruiements
```cmd
pip install -r requirements-cpu.txt
```

2. Traing MLP model
```cmd
python train-cpu.py
```

3. Start MLPeditor
```cmd
python MLPeditor-CPU.py
```

### GPU

1. Install reqruiements
```cmd
pip install -r requirements-gpu.txt
```

2. Traing MLP model
```cmd
python train-gpu.py
```
3. Start MLPeditor
```cmd
python MLPeditor-GPU.py
```

## MLPeditor Manual

### Button Functions

![image](https://github.com/user-attachments/assets/109b2547-ca2b-437d-8795-93a339f57bfa)
```Previous Layer``` moves to the previous weight matrix

![image](https://github.com/user-attachments/assets/1ef6c406-d357-4f61-be32-2ae02dbc5402)
```Next Layer``` moves to the next weight matrix

![image](https://github.com/user-attachments/assets/819d14ad-e362-44cd-baf6-56adeb441a14)
```Eval``` evaluates the model using the current weight matrix

![image](https://github.com/user-attachments/assets/0184c265-5c44-4a91-bfa5-e2677be442b6)
```Round``` rounds the selected region

![image](https://github.com/user-attachments/assets/5683e28f-1622-4d4f-8935-5e2f93d9ae76)
```Fill``` fills the selected region


![image](https://github.com/user-attachments/assets/9bcdcdd0-4dd5-4bbb-acf7-280e8263ab6d)
```Restore``` restores  the selected region to the original weight matrix (buggy)


![image](https://github.com/user-attachments/assets/71691bba-f19b-4af1-99c2-fe30362723e2)
```Restore All``` restore all of the weight matrix to the original weight matrix (buggy)


![image](https://github.com/user-attachments/assets/2684e895-5359-4d8d-ae80-6fdcd553ec19)
```Fill Value``` the value to fill the values in a selected region


![image](https://github.com/user-attachments/assets/9e09a0d5-81c2-4092-a36a-096c1997dbdd)
```Magnifying Glass``` resizes a selected region to the current matrix size


![image](https://github.com/user-attachments/assets/96b6fb7e-1505-4fa6-9789-5a99fc9c596e)
```Move``` allows to move the matrix when dragging

![image](https://github.com/user-attachments/assets/c702356f-a3e2-4311-bceb-21841692a307)
```Home``` allows to return to original size

### Text Display Functions

![image](https://github.com/user-attachments/assets/f431e610-558f-422e-8af7-b85241a210c8)

![image](https://github.com/user-attachments/assets/7cf160f0-a424-4f12-9105-7580976792c7)

0-9 Digits and "nan" values are the acuraccy of the model evaluated with 0-9 digits. Will be updated after evaluation


![image](https://github.com/user-attachments/assets/1238011d-5437-40a2-a971-ff4a37bf71ae)
Acuraccy will update after model evaluated.

### Matrix Functions

https://github.com/user-attachments/assets/6d888fef-f4d6-43dd-943a-39206ae01a21

