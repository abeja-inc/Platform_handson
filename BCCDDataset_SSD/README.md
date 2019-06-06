### アップロード
```
abeja datalake create-channel --name "BCCD_CHANNEL" --description "BCCD data"
abeja datalake upload --channel_id XXXXX --recursive ./BCCD_Dataset/BCCD/JPEGImages/
```

### クレデンシャルデータの抜き出し
```
abeja config show --format json > ./scripts/credential.json
```

### アノテーションデータ紐付け
```
python ./scripts/import_dataset_from_datalake.py \
          -o {OrganizationID} \
          -c {ChannelID} \
          -d BCCD_dataset_trainval \
          --split trainval \
          --max_workers 4
          
python ./scripts/import_dataset_from_datalake.py \
          -o {OrganizationID} \
          -c {ChannelID} \
          -d BCCD_dataset_trainval \
          --split test \
          --max_workers 4
```
