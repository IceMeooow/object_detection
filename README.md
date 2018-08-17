# Object Detection

### Project folder structure
The following folders are required for this project:
- **object-detection**, **nets**, **slim** - this folders contains required files (models, config files, example of label map and ect.).
>**Note:** they are taken from [here](https://github.com/tensorflow/models/tree/master/research).

- **image** - all images for the project are stored here (they are labeled).

- **train_data**, **test_data** - here are saved images for the train and the test processes (copied from image folder).
>**Note:** `image`, `train_data`, `test_data` folders are not displayed here but they must be in the project.

- **ssd_mobilenet_v1_coco** - The [pre-prepared model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) is stored in this folder.

- **data** - this folder contains CSV and TFRecords files.
>**Note:** `data` is not displayed here but it must be in the project.

- **training** - folder for training process.
>**Note:** here must be stored config file for model and label map file.

- **eval** - folder for evaluation process.

- **testing** - folder for testing process.
>**Note:** this folder should contain two more: `test_image` (saved images for testing) and `results` (saved testing results)


### How to train the Model
1. Generate CSV files from images:
    ```
    python xml_to_csv.py
    ```

2. Generate TFRecords files:
    - create train data:
        ```
        python generate_tfrecord.py --csv_input=data/train_data_labels.csv --output_path=data/train_data.record
        ```
   
    - create test data:
        ```
        python generate_tfrecord.py --csv_input=data/test_data_labels.csv --output_path=data/test_data.record
        ```

3. Train the Model:
   ```
   python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1.config
   ```

4.  Evaluate the Model:
    ```
    python eval.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v1.config --checkpoint_dir=training/ --eval_dir=eval/
    ```

5. Export the Model:
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1.config --trained_checkpoint_prefix training/model.ckpt-500  --output_directory model_inference_graph
    ```
    **Note:** the Model will be exported in the **model_inference_graph** folder.

6. Test the Model:
    ```
    python test.py
    ```
    **Note:** the testing results will be saved in the `result` folder *(path: `testing/result`)*.

### How to visualize results
- the training results
    ```
    tensorboard --logdir=training
    ```
- the evaluation results
    ```
    tensorboard --logdir=eval/
    ```

### Help
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 
- [How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) 
- [TensorFlow Object Detection API tutorial. Training and Evaluating Custom Object Detector](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) 
