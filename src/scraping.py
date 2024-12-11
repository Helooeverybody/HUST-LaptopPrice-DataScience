import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import SessionNotCreatedException,WebDriverException
import time
import pandas as pd
import multiprocessing as mp

convert_dict = {"text-red-500" : 0, "text-green-500" : 1}


NUM_WORKERS = 8
START = 80000
END = 140000
SAVE_CHUNK = 400



def parse(data):
    retries = 3
    id,row = data
    print(id)
    for _ in range(retries):
        try:
            ffx = webdriver.Chrome()
            ffx.get(row[0] + "specs/")
            break
        except (SessionNotCreatedException,WebDriverException):
            print("Session not created, retrying...")
            time.sleep(3)
    try:
        element = WebDriverWait(ffx, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "inpageSections.loaded"))
        )
    except Exception:
        return (-1,{})
    try:
        soup = BeautifulSoup(ffx.page_source,"html.parser")
        hehe = soup.find("div",attrs={"class":"inpageSections loaded","id":"section-specs"})
        better_hehe = hehe.find("div",attrs={"class":"lm-catalog-specs border-b-2 border-dashed text-lm-darkBlue border-gray-300 pt-5 pb-10"})
    except AttributeError:
        return (-1,{})
    try:
        ffx.close()
    except Exception:
        pass
    next_hehe = better_hehe.find_all("ul")
    hahahaha = {}
    for data in next_hehe:
        hohoho = data.find_all("li")
        if len(hohoho)!=2:
            continue
        final_labels = [k.get_text().strip() for k in hohoho]
        if final_labels[1] == "":
            try:
                final_labels[1] = convert_dict[hohoho[1].find("i")["class"][1]]
            except TypeError:
                continue
        if "USB" in final_labels[0]:
            hehehe = final_labels[0].split()
            if len(hehehe)==3:
                hohoho = " ".join(hehehe[1:])
                hihihi = int(hehehe[0][0])
                hahahaha[hohoho] = hahahaha.get(hohoho,0) + hihihi  
            elif len(hehehe)==2:    
                hohoho = " ".join(hehehe)
                hahahaha[hohoho] = hahahaha.get(hohoho,0) + 1
            continue
        match final_labels[0]:
            case "CPU"|"GPU":
                hahahaha[final_labels[0]] = final_labels[1].split("#")[0]
            case "M.2 Slot":
                hahahaha[final_labels[0]] = final_labels[1].split("\n")[0]
            case _:
                if final_labels[0] in ['link','name','CPU','GPU','Display','HDD/SSD','RAM','OS','Body Material','Dimensions','Weight','M.2 Slot','USB Type-C','USB Type-A','HDMI','Bluetooth','Wi-Fi','Card Reader','Ethernet LAN','Web camera','Security Lock slot','Fingerprint reader','Backlit keyboard','Cost','Total Score','Portability Score','Display Score','Work Score','Play Score']:
                    hahahaha[final_labels[0]] = final_labels[1]  
    
    cost = soup.find("ul",attrs = {"class":"catalog-buttons-3 grid grid-cols-1 mt-5 gap-3 text-left"})
    score_part = soup.find("div",attrs = {"class":"grid gap-2 grid-cols-[1fr_auto]"})
    if cost is not None:
        better_cost = cost.find("div",attrs = {"class":"flex flex-wrap flex-row-reverse xl:flex-nowrap xl:flex-col items-center justify-center h-full text-lg sm:text-2xl xl:text-3xl font-bold py-1 px-2 gap-x-2"})
        if better_cost is not None:
            hahahaha["Cost"] = [line.strip() for line in better_cost.get_text().splitlines() if line.strip()][0]
    if score_part is not None:
        kirara = [line.strip() for line in score_part.get_text().splitlines() if line.strip()]    
        hahahaha["Total Score"] = kirara[0]
        hahahaha["Portability Score"] = kirara[2]
        hahahaha["Display Score"] = kirara[5]
        hahahaha["Work Score"] = kirara[8]
        hahahaha["Play Score"] = kirara[11]
    return id,hahahaha

if __name__ == "__main__":
    chunksize = 10
    myhahaha = pd.read_csv("laptop_2.csv",low_memory=False)
    num_workers = NUM_WORKERS
    
    start = START
    end= END
    args = [(id,row) for (id,row) in myhahaha[myhahaha["CPU"].isnull()].iterrows() if (id>=start and id<=end)]
    for hehe in range(0,len(args),SAVE_CHUNK):
        with mp.Pool(num_workers) as pool:
            res = pool.imap(parse, args[hehe:hehe+SAVE_CHUNK], chunksize=chunksize)
            for result_chunk in res:
                id,hahahahaha=result_chunk
                if not hahahahaha:
                    continue
                for key, value in hahahahaha.items():
                    myhahaha.at[id,key] = value
        myhahaha.to_csv("laptop_2.csv", index=False)