import urllib.request
import os
f = open("Dataset.txt", "r")
image_num = 1
for line in f:
    try:
            if os.path.exists(line[28:line.__len__()-1]) == False:
                urllib.request.urlretrieve("http://" + line, line[28:line.__len__()-1])
                print(line,'is downloaded :')
            image_num = image_num + 1
            print(image_num)
    except Exception as e:
        print(line)
        print(e)
        continue
