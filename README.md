# Surveillex Code but different flow.


### Set Dataset Path

link for downloading dataset : [link](https://drive.google.com/file/d/1ayZAy70DXMXg7c0wv8wH_V9cxuMa7ZYK/view?usp=sharing)
unzip above dataset and put list in the root

### using dataset.py

```bash
normal_train_dataset = ShanghaiDataset(train=True, is_normal=True, transform=None)
abnormal_train_dataset = ShanghaiDataset(train=True, is_normal=False, transform=None)
test_dataset = ShanghaiDataset(train=False, transform=None)
```
