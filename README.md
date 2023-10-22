# Install requirements
```
pip install -r requirements.txt
```

# Train KGQA Models
Because github does not allow users to upload file with size larger than 100MB. To set-up the project, you firstlt need to train KGQA models. After
training the models, please put them on ./KGQA/preprocess/.
```
python train_detection.py --entity_detection_mode LSTM --fix_embed --gpu 0
python train_entity.py --qa_mode GRU --fix_embed --gpu 0
python train_pred.py --qa_mode GRU --fix_embed --gpu 0
```

# Train Content-based model
We use haystack to deploy content-based question answering. Details: https://haystack.deepset.ai/tutorials.
After training the model, please set the folder name as 'my_model' and put it on ./Content_based_QA/
