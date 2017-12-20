# caffe-squeezeDet

#### This is the caffe version of squeezeDet. And I converted tensorflow  model directly into caffemodel. 
----
### Note
----
The convolution operation is different in tensorflow and caffe, especially the padding iterm. Thus, using directly converted model will cause problems. **Trick**:  using a pad = 2 convolution operation for CONV1 Layer to get a larger feature map and than crop into the right size. 

But for kernel size 3 x 3 or 1 x 1 and stride step 1 with padding SAME in tensorflow, we directly choose convolution param with pad = 1, kernel size = 3 and stride = 1
### Demo
``` python
cd caffe-squeezeDet
python ./src/demo.py
```
