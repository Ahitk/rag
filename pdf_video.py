#pdf ve video dosyalarini bulmak icin yazilan ChatGPT kodu ama calismadi.

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

base_url = "https://www.telekom.de/hilfe"
visited_urls = set()

def find_files(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    files = []
    
    # PDF dosyalarını arayın
    pdf_files = soup.find_all('a', href=re.compile(r'\.pdf$'))
    files.extend([urljoin(base_url, link['href']) for link in pdf_files])
    
    # Video dosyalarını arayın
    video_files = soup.find_all('a', href=re.compile(r'\.(mp4|avi|mkv|mov)$'))
    files.extend([urljoin(base_url, link['href']) for link in video_files])
    
    return files

def crawl_site(url, depth=0, max_depth=10):
    if url in visited_urls or depth > max_depth:
        return []
    
    visited_urls.add(url)
    files = find_files(url)
    
    # Sayfadaki diğer linkleri tarayın
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all('a', href=True):
        full_url = urljoin(base_url, link['href'])
        
        # Geçersiz veya istenmeyen URL'leri filtrele
        if full_url.startswith(('javascript:', '#', 'tel:', 'mailto:')):
            continue
        files.extend(crawl_site(full_url, depth=depth+1, max_depth=max_depth))
    
    return files

# Web sitesini tarayın ve dosya türlerini bulun
found_files = crawl_site(base_url)

# Sonuçları bir txt dosyasına yazdır
with open("found_files.txt", "w") as file:
    for file_url in found_files:
        file.write(file_url + "\n")




