# Yolov11 Tuning

## Error
* Get onnx model (in the ultralytics workspace - https://github.com/ultralytics/ultralytics)
  ```python
  from ultralytics import YOLO
  # Load the YOLO11 model
  model = YOLO("yolo11n.pt")

  # Export the model to ONNX format
  model.export(format="onnx")  # creates 'yolo11n.onnx'

  # Load the exported ONNX model
  onnx_model = YOLO("yolo11n.onnx")

  # Run inference
  results = onnx_model("https://ultralytics.com/images/bus.jpg")
  ```

* Move onnx model (`yolo11n.onnx`) to `data` directory

* Try tunning 
  ```bash
  python3 0_tuning_causes_error.py
  ```


## Cause: Dynamic handling of input data

* Yolo v11 handles dynamic shape of inputs by changing `anchors`, `strides`, and `shape`
* You can see them in `class Detect` of `head.py`
  
  (1) Initialization
    ```python
    class Detect(nn.Module):
        """YOLO Detect head for detection models."""
    
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        format = None  # export format
        end2end = False  # end2end
        max_det = 300  # max_det
        # !HERE! ############################
        shape = None
        anchors = torch.empty(0)  # init  
        strides = torch.empty(0)  # init
        ######################################
        legacy = False  # backward compatibility for v3/v5/v8/v9 models
    ```

  (2) Inference
    ```python
      def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        # HERE ###############################
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        ######################################
    ```

