import sys
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import string
import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging
import re
import math
import predict
from nltk.corpus import stopwords
from pandasgui import show
import threading

logging.basicConfig(level=logging.INFO)
global all_results 
all_results=[]
global j
global review_count
review_count=0
global save_name
headers = {
    "authority": "www.amazon.in",
    "pragma": "no-cache",
    "cache-control": "no-cache",
    "dnt": "2",
    "upgrade-insecure-requests": "2",
    "user-agent": "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "sec-fetch-site": "none",
    "sec-fetch-mode": "navigate",
    "sec-fetch-dest": "document",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
}

def print_to_widget(*args, **kwargs):
    output_text.insert(tk.END, *args, **kwargs)

sys.stdout.write = print_to_widget

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def get_page_html(page_url: str) -> str:
    resp = requests.get(page_url, headers=headers)
    return resp.text

def get_reviews_from_html(page_html: str) -> BeautifulSoup:
    soup = BeautifulSoup(page_html, "lxml")
    reviews = soup.find_all("div", {"class": "a-section celwidget"})
    return reviews

def get_product_names(soup_object: BeautifulSoup):
    try:
        prname = soup_object.find('span', {'id': 'productTitle'}).get_text().strip()
        print(prname)
        return prname
    except:
        prname="titlenotfound"
        return prname

def get_product_category(soup_object: BeautifulSoup):
    try:
        cat = soup_object.find('a', class_='nav-a nav-hasImage')['href']
        cat = cat.split('/')[1]
        return cat
    except:
        try:
            cat = soup_object.find('span', {'class': 'nav-a-content'}).get_text().strip()
            return cat
        except:
            cat="amazon product"
            return cat

def get_review_date(soup_object: BeautifulSoup):
    date_string = soup_object.find("span", {"class": "review-date"}).get_text()
    return date_string

def get_review_text(soup_object: BeautifulSoup) -> str:
    try:
        review_text = soup_object.find("span", {"class": "a-size-base review-text review-text-content"}).get_text()
        return review_text.strip()
    except AttributeError:
        review_text="NULL"
        return review_text.strip()

def get_review_header(soup_object: BeautifulSoup) -> str:
    try:
        review_header = soup_object.find("a",
            {"class": "a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold" },).get_text()
        return review_header.strip()
    except AttributeError:
        review_header="NULL"
        return review_header.strip()

def get_number_stars(soup_object: BeautifulSoup) -> str:
    try:
        stars = soup_object.find("span", {"class": "a-icon-alt"}).get_text()
        return stars.strip()
    except AttributeError:
        stars="NULL"
        return stars.strip()

def get_product_cat():
    return category

def orchestrate_data_gathering(single_review: BeautifulSoup) -> dict:
    return {
        "review_text": get_review_text(single_review),
        "review_date": get_review_date(single_review),
        "review_title": get_review_header(single_review),
        "review_stars": get_number_stars(single_review),
        "product_category": get_product_cat(),
    }

def run_code():
    url = url_entry.get()
    model = model_entry.get()
    # Code to run
    thread = threading.Thread(target=run_scraping_and_processing, args=(url, model))
    thread.start()

def run_scraping_and_processing(url, model):
    resp = get_page_html(url)
    soup = BeautifulSoup(resp, "lxml")
    global prname
    prname = get_product_names(soup)
    global category
    category = get_product_category(soup)
    new_url = re.sub("/dp/", "/product-reviews/", url)
    new_url = re.sub("ref=.*$", "ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews", new_url)
    resp = requests.get(new_url, headers=headers)
    if resp.status_code != 200:
        print("Internet or browser error!")
        return
    soup = BeautifulSoup(resp.text, "lxml")
    reviewsn = soup.find_all("div", {"class":"a-row a-spacing-base a-size-base"})
    if not reviewsn:
        print("Review not found!")
        return
    text1 = reviewsn[0].get_text()
    review_count = re.search(r'(\d+(,\d+)?)\s+with reviews', text1).group(1)
    review_count = int(review_count.replace(",", ""))
    tots="Total review = "+str(review_count)
    print(tots)
    rev = math.ceil(review_count / 10)
    generate_review_urls(new_url, rev)
    savefile(model)

def generate_review_urls(base_url, rev):
    reviewerType = "?ie=UTF8&reviewerType=all_reviews&pageNumber="
    for page_number in range(1, rev+1):
        url = base_url + "/ref=cm_cr_arp_d_paging_btm_next_" + str(page_number) + reviewerType + str(page_number)
        print("Generating URL:", page_number)
        print(url)
        scrape(url)

def scrape(u):
    html = get_page_html(u)
    reviews = get_reviews_from_html(html)
    for rev in reviews:
        data = orchestrate_data_gathering(rev)
        all_results.append(data)

def savefile(model):
    out = pd.DataFrame.from_records(all_results)
    logging.info(f"{out.shape[0]} is the shape of the dataframe")
    save_name = prname.split()[0] + "_reviews" + ".csv"
    logging.info(f"Saving to 'File/{save_name}'")
    out.to_csv('scraper/FILE/' + save_name)
    logging.info('Extract done')
    df = predict.predictnew(save_name, int(model))
    df = df.drop(['review_date', 'review_title', 'review_stars'], axis=1)
    print(df)
    show(df)
    

# Creating tkinter window
window = tk.Tk()
window.title("Review Scraper")
window.geometry("1240x720")
window.attributes('-fullscreen', True)

# Creating URL label and entry
url_label = ttk.Label(window, text="Enter Amazon product URL:")
url_label.pack()
url_entry = ttk.Entry(window, width=360)
url_entry.pack()

# Creating model label and entry
model_label = ttk.Label(window, text="Enter model number (1-5):")
model_label.pack()
model_entry = ttk.Entry(window, width=360)
model_entry.pack()

# Creating submit button
submit_button = ttk.Button(window, text="Submit", command=run_code)
submit_button.pack()
# Creating output text widget
output_text = ScrolledText(window, width=960, height=600)
output_text.pack()



# Run tkinter event loop
window.mainloop()
