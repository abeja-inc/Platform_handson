### 概要
- BCCDデータセット（アノテーション済み）を使用し、ABEJA Platformにアップロードする
- アップロードされたデータを使用し、SSDを動かす

### データセット
- 既にアノテーション済みのデータセット
- 下記リンクからデータを引用済み    
https://github.com/Shenggan/BCCD_Dataset  
Copyright (c) 2017 shenggan

### 作業
#### アップロード
```
abeja datalake create-channel --name "BCCD_CHANNEL" --description "BCCD data"
abeja datalake upload --channel_id XXXXX --recursive ./BCCDDataset_SSD/BCCD/JPEGImages/
```
#### クレデンシャルデータの抜き出し
```
cd ./BCCDDataset_SSD
abeja config show --format json > ./scripts/credential.json
```

#### アノテーションデータ紐付け
- {OrganizationID} と {ChannelID}は適宜、置き換える
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
          -d BCCD_dataset_test \
          --split test \
          --max_workers 4
```
