import requests
import json
import pprint
import os, time
from abeja.datalake import Client as DatalakeClient
from abeja.datalake import APIClient as APIDatalakeClient

f = open("credential.json","r")
credential = json.load(f)

user_id = credential["abeja-platform-user"]
personal_access_token = credential["personal-access-token"]
organization_id = credential["organization_id"]
channel_id = credential["channel_id"]
mecab_url = credential["mecab_url"]

credential = {
    'user_id': user_id,
    'personal_access_token': personal_access_token
}

def upload_datalake(file_path):

    client = DatalakeClient()
    datalake_client = DatalakeClient(organization_id=organization_id,  credential=credential)
    channel = datalake_client.get_channel(channel_id)

    metadata = {}
    file = channel.upload_file(file_path, metadata=metadata)

    return 0

def delete_datalake(channel_id, file_id):

    client = APIDatalakeClient(credential=credential)
    client.delete_channel_file(channel_id=channel_id, file_id=file_id)

    return 0

def all_delete_datalake(channel_id):

    client = APIDatalakeClient(credential=credential)
    while 1==1:
        channel_list = client.list_channel_files(channel_id=channel_id)
        for x in channel_list['files']:
            delete_datalake(channel_id, x['file_id'])
        if channel_list['next_page_token'] == None:
            break
        time.sleep(10)
        


if __name__ == '__main__':

    # clean datalake
    all_delete_datalake(channel_id)

    path = "./text"

    # Get category name (Ex:dokujo-tsushin, smax etc)
    cat_f_list = os.listdir(path)
    cat_f_list.sort()

    print(cat_f_list)
    #setting ignore file or folder
    ignore_list = ["LICENSE.txt","output","CHANGES.txt", "README.txt","result_mecab"]

    # Get folder name
    for cat_f_name in cat_f_list:
        if cat_f_name in ignore_list:
            pass
        else:
            target_f_path = os.path.join(path, cat_f_name)
            folder_list = os.listdir(target_f_path)

            di ={}
            result_text = ""
            out_file_path = os.path.join(".","result_mecab",str(cat_f_name)+".txt")

            if not os.path.exists("result_mecab"):
                    os.mkdir("result_mecab")

            for file_name in folder_list:
                if file_name in ignore_list:
                    pass
                else:
                    in_file_path = os.path.join(target_f_path, file_name)
                    with open(in_file_path,"r") as f:
                        lines = f.readlines()

                        # ignore line:1&2(Not have information)
                        #di= { index:line for index, line in enumerate(lines) if line !="\n" and index!=0 and index!=1}

                        doc = ""
                        for index, line in enumerate(lines):
                            if index != 0 and index != 1:
                                doc += line.rstrip("\n").lstrip("　")
                        di = {"text":doc}
                        text = json.dumps(di)

                        auth = (user_id,personal_access_token)
                        response = requests.post(mecab_url, text,headers={'Content-Type': 'application/json; charset=UTF-8'}, auth=auth)

                        result_text = result_text + response.json()

            with open(out_file_path, "w") as f:
                f.write(result_text)

            upload_datalake(out_file_path)
