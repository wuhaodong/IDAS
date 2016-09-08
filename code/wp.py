# coding: utf-8
import xml.etree.ElementTree as ET
import json
import sys
import pprint

if __name__ == "__main__":
    #just add attributes you want here
    #key=label in the xml file, value=the name you want to show in the json file
    #_id and url is needed
    attrs = {            
                "wp:attachment_url":"url",
                "dc:creator":"creator"
            }

    with open(sys.argv[1], "r") as f:
        fout = open("data.json", "w")
        #fout.write("["+"\n")
        flag = False
        #data = dict()
        data = {}
        data_list = []
        
        n_item = 0

        for line in f.readlines():
            #read data for each item (between <item> and </item>)
            if line.startswith("<item>"):
                flag = True
              
                #if flag_item and data["creator"]=="david":
                    #fout.write(","+"\n")
                #flag_item = False
                continue
            elif line.startswith("</item>"):
                flag = False
                data["media"] = "picture" # assume all the ads are picture
                if data["creator"] == "david":
                    #fout.write(str(data).replace("'", '"'))
                    #print data
                    data_list.append(data)
                    n_item += 1
                data = {}
                #data.clear()
               

            key = line[line.find("<")+1:line.find(">")]
            val = line[line.find(">")+1:line.rfind("<")]

            if flag:
                if key in attrs:
                    data[ attrs[key] ] = val
                elif key.startswith("category"):
                    key = key.split()
                    category = key[1].split('"')
                    nickname = key[2].split('"')[1]
                    nickname = nickname.split('-')[0]
                    if category[1] == "media_category":                   
                        cdata = val[val.rfind("[")+1 : val.find("]")]
                        cdata = cdata.split('-')[1]
                        if nickname in ["ir", "dd", "tp"]:
                        #if nickname in ["tp"]:
                            cdata = float(cdata)
                        data[nickname] = cdata
                      

        #fout.write("\n"+"]")
        pprint.pprint(data_list) 
        #write the file
        fout.write('['+'\n')
        for i in range(len(data_list)):
            data = data_list[i]
            if i == len(data_list)-1:
                fout.write(str(data).replace("'",'"')+'\n')
            else:
                fout.write(str(data).replace("'",'"')+','+'\n')
        fout.write(']')

