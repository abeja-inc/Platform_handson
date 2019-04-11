## [言語処理]MeCabサーバを用意する
***手順***
- notebookで実施すること
  - MeCabをinstallする
  - [辞書]ipadicをinstallする
  - [辞書]Neologdをinstallする（軽量ver）
  - 稼働確認し、Pythonで動かすためのmecab-python3をinstallする（制約あり）
  - moduleをzip圧縮する

- モジュールをデプロイする
  - 環境変数の設定が必須
    - MECABRC:./local/etc
    - LD_LIBRARY_PATH:./local/lib
    - PATH:$PATH:./local/bin
![環境変数](https://user-images.githubusercontent.com/17213216/55971477-8797f280-5cbc-11e9-8297-10cf19461b6e.png)

- Jsonを用意してチェックしてみる
  - 例えば {"0":"マツコ・デラックス"}の結果
![mecab](https://user-images.githubusercontent.com/17213216/55970790-30dde900-5cbb-11e9-9752-ad54d1001a1b.png)
