from selenium import webdriver
from time import sleep
import requests
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


driver = webdriver.Chrome()
driver.get("https://auth.fandom.com/signin")

User = driver.find_element(By.ID, "identifier")
User.send_keys("HaiBooBro") 

Pass = driver.find_element(By.ID, "password")
Pass.send_keys("0112201709012001")

#Submit form
Pass.send_keys(Keys.ENTER)
#get hero

def get_imgurl_hero(hero):
    driver.get("https://leagueoflegends.fandom.com/wiki/File:{}_OriginalCentered.jpg#Circles".format(hero))
    img_urls = []
    tags = ["gallery-2","gallery-3", "gallery-5"]
    for tag in tags:
        parents = driver.find_element(By.ID, tag)
        soup = BeautifulSoup(parents.get_attribute('outerHTML'), 'html.parser')
        gallery_items = soup.select('div.wikia-gallery-item')

        for item in gallery_items:
            img_tag = item.find('img', class_='thumbimage')
            
            # get url
            if img_tag:
                img_url = img_tag['src']
                # print(f"Image_url: {img_url}")
                img_urls.append(img_url)
    return img_urls

def save_image_url(img_urls, hero):

    if not os.path.exists(hero):
        os.makedirs("train_data/{}".format(hero))

    for i in range(len(img_urls)):
        response = requests.get(img_urls[i])
        if response.status_code == 200:
            with open("train_data/{}/{}_{}.png".format(hero, hero, i), 'wb') as f:
                f.write(response.content)

if __name__== "__main__":
    
    # with open('test_data\hero_names.txt', 'r') as file:
    #     heros_list = file.read().splitlines()

    # for hero in heros_list:
    img_urls = get_imgurl_hero("Kha%27Zix")
    save_image_url(img_urls, "Kha%27Zix")
    print("hero: {}, number of img:{}".format("Kha%27Zix", len(img_urls)))
    print("DONE!")