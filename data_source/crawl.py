import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from dotenv import dotenv_values

env_name = "period.env"
config = dotenv_values(env_name)

START_DATE = config["START_DATE"]
END_DATE = config["END_DATE"]

file_links = []
page_number = 1

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--user-data-dir=/tmp/chrome-user-data")

driver = webdriver.Chrome(options=options)

while True:
    url = f"https://thuvienphapluat.vn/banan/tim-ban-an?type_q=0&AgentId=0&CityId=65&sortType=1&Category=7&page={page_number}"
    driver.get(url)
    time.sleep(3)

    try:
        start_date_field = driver.find_element(By.NAME, "StartPublicDate2")
        start_date_field.clear()
        start_date_field.send_keys(START_DATE)
        start_date_field.send_keys(Keys.RETURN)
    except NoSuchElementException:
        print("No more content or StartPublicDate2 element not found.")
        break

    time.sleep(1)

    try:
        end_date_field = driver.find_element(By.NAME, "EndPublishDate2")
        end_date_field.clear()
        end_date_field.send_keys(END_DATE)
        end_date_field.send_keys(Keys.RETURN)
    except NoSuchElementException:
        print("No more content or EndPublishDate2 element not found.")
        break

    time.sleep(3)

    links = driver.find_elements(By.CSS_SELECTOR, "a.h5.font-weight-bold")
    if not links:
        print(f"No links found on page {page_number}")
        break

    for link in links:
        title = link.text.strip()
        href = link.get_attribute("href")
        file_links.append({
            "title": title,
            "url": href
        })

    print(f"Page {page_number}: Collected {len(links)} links.")
    page_number += 1

driver.quit()

filename = f"{START_DATE}_{END_DATE}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(file_links, f, ensure_ascii=False, indent=4)

print(f"\nTotal links collected between {START_DATE} and {END_DATE}: {len(file_links)}")
print(f"Data exported to {filename}")
