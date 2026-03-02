The current implementation looks good. However, if the M dimension was tiled to 1, the weights have to be loaded several
  times. However, I don't think the current tensor_to_stream can express this case as it only expresses a linear read
  over the tensor. Add a stride and shape field additionally to tensor_to_stream so that it can express an affine read
  over the tensor to support other examples too.