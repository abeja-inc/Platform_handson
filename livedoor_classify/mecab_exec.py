import requests
import json
import pprint
import os, time
import credentials
from abeja.datalake import Client as DatalakeClient
from abeja.datalake import APIClient as APIDatalakeClient

user_id = credentials.user_id
personal_access_token = credentials.personal_access_token
organization_id = credentials.organization_id
channel_id = credentials.channel_id
mecab_url = credentials.mecab_url

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

    #setting ignore file or folder
    ignore_list = ["LICENSE.txt","output","CHANGES.txt", "README.txt"]

    # Get folder name
    for cat_f_name in cat_f_list:
        if cat_f_name in ignore_list:
            pass
        else:
            target_f_path = os.path.join(path, cat_f_name)
            folder_list = os.listdir(target_f_path)

            di ={}

            for file_name in folder_list:
                if file_name in ignore_list:
                    pass
                else:
                    in_file_path = os.path.join(target_f_path, file_name)
                    out_file_path = os.path.join(path,"result_mecab",file_name)
                    with open(in_file_path,"r") as f:
                        lines = f.readlines()

                        # ignore line:1&2(Not have information)
                        di= { index:line for index, line in enumerate(lines) if line !="\n" and index!=0 and index!=1}

                        text = json.dumps(di)

                        auth = (user_id,personal_access_token)
                        response = requests.post(mecab_url, text,headers={'Content-Type': 'application/json; charset=UTF-8'}, auth=auth)

                        with open(out_file_path, "w") as f:
                            pprint.pprint(response.json(), stream=f)

                        upload_datalake(out_file_path)
