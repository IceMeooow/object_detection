## Object Detection

1. Generate CSV files from Images:
    ```
    python xml_to_csv.py
    ```

2. Generate TFRecords
   Create train data:
   ```
   python generate_tfrecord.py --csv_input=data/train_data_labels.csv  --output_path=data/train_data.record
   ```
   
   Create test data:
   ```
   python generate_tfrecord.py --csv_input=data/test_data_labels.csv  --output_path=data/test_data.record
   ```

3. Train the Model:
   ```
   python model_main.py --pipeline_config_path=training/ssd_mobilenet_v1.config --model_dir=ssd_mobilenet_v1_coco --alsologtostderr
   ```
   
