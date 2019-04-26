# Platform_handson

### ハンズオンコンテンツを管理するためのレポジトリ

#### 補足
- [ABEJA PlatformのDeveopperサイト](https://developers.abeja.io/general/)  
※上記はMacOS or Linux上で動作保証済みとなります

#### コマンドのインストールまわり
https://developers.abeja.io/cli/overview/  
https://developers.abeja.io/sdk/

#### (仮)Windowsで ABEJA PLatform CLI や ABEJA SDKを使用する場合
- Windows Subsystem for Linuxを使う
  - [このqiita記事](https://qiita.com/Aruneko/items/c79810b0b015bebf30bb)に従って、windowsにubuntuをいれる
  - コマンドの実施 (下準備)
    - ```sudo apt-get update```
    - ```sudo apt-get install python3-pip```
    - ```curl -s https://packagecloud.io/install/repositories/abeja/platform-public/script.python.sh | bash```
    - ```pip3 install abejacli```
    - ```pip3 install abeja-sdk```
![説明](https://user-images.githubusercontent.com/17213216/56480165-ccf5c480-64f3-11e9-8320-df1e5922176e.png)
    - 一旦コマンドラインの再起動を実施後、```abeja configure```を打鍵してみる（以降の流れは[ここ](https://developers.abeja.io/cli/overview/)の**初期設定**を参考に  
![説明2](https://user-images.githubusercontent.com/17213216/56480415-f105d580-64f4-11e9-84b9-799f4adc7997.png)

- VirtualBoxを使う（Linux仮想化）
  - [このqiita記事](https://qiita.com/miyagaw61/items/b44a89eb636d16de010c)に従って、windowsでubuntuを使えるようにする
  - Ubuntuが使えるようになってから、[ここ](https://developers.abeja.io/cli/overview/)の作業を進めるでOK
  - メモリの割り当てや、パーティションの分割が発生するため、お手軽にやるならWSFL(上記)を使うことを進める（リソースが潤沢ではない場合はなおさら、圧迫する可能性もあるので）

- windows for dockerとかcygwinは未検証
