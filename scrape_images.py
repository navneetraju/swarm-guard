from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import os
import requests
from bs4 import BeautifulSoup
import csv


def get_driver():
    edge_driver_path = '/Users/prathamsolanki/PycharmProjects/DSCi-560/RA_application/edgedriver_mac64_m1/msedgedriver'

    options = Options()
    options.add_argument("--headless")  # Enable headless mode
    options.add_argument("--disable-gpu")  # Disable GPU acceleration (optional but recommended)
    options.add_argument("--window-size=1920,1080")  # Set window size if needed

    # Initialize the Edge WebDriver with the specified options
    service = EdgeService(executable_path=edge_driver_path)
    driver = webdriver.Edge(service=service, options=options)

    return driver

def write_html(html_content):
    # Beautify the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    formatted_html = soup.prettify()

    with open('temp.html', "w", encoding="utf-8") as file:
        file.write(formatted_html)
    print("Done Writing")

def download_image(image_url, tweet_id, user_id, folder="images_test"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Build the filename and full file path
    filename = f"{tweet_id}_{user_id}.jpg"
    filepath = os.path.join(folder, filename)

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an error on bad status
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Image downloaded as {filepath}")
        return filepath  # Return the full file path
    except requests.RequestException as e:
        print(f"Failed to download image: {e}")
        return None


from selenium.common.exceptions import TimeoutException


def get_webpage(tweet_url):
    try:
        driver = get_driver()
        driver.set_page_load_timeout(30)  # Set a high timeout (e.g., 300 seconds)

        try:
            driver.get(tweet_url)
        except TimeoutException as te:
            print(f"Page load timed out for {tweet_url}: {te}")
            # Only if a timeout error occurs, stop the page load
            driver.execute_script("window.stop();")

        driver.implicitly_wait(5)
        current_url = driver.current_url
        if "x.com" not in current_url:
            print(f"Redirected URL ({current_url}) does not contain 'x.com'. Returning empty string.")
            return ""
        driver.implicitly_wait(55)  # Wait up to 60 seconds total for elements to load

        # Try first selector, then fallback
        try:
            image_element = driver.find_element(By.CSS_SELECTOR, "img.css-9pa8cd[alt='Image']")
        except Exception as e:
            try:
                image_element = driver.find_element(By.CSS_SELECTOR, "img.css-9pa8cd[src*='card_img']")
            except Exception as e:
                print("No image element found using either selector.")
                image_element = None

        if image_element:
            image_url = image_element.get_attribute("src")
            if image_url:
                return image_url
            else:
                print("Image found, but no src attribute available.")
                return ""
        else:
            return ""
    finally:
        driver.quit()
        print("complete")


def get_single_image(tweet_url):
    image_url = get_webpage(tweet_url)

    download_image(image_url, 'id', 'user_id', folder="images")

# get_single_image("https://x.com/i/web/status/1032768723608842240")#court
# get_single_image("https://x.com/i/web/status/1021045358208716807")#dam
# get_single_image("https://t.co/Zzb5sr8MaN") # COURT2
# get_single_image("https://t.co/OEU0zwv2Eu")# papa johns
# get_single_image("https://t.co/HnkJcqdqok")#trump
# https://t.co/Zzb5sr8MaN



def get_multiple_images():
    # Input CSV file path (existing CSV with columns: index, file_name, id, user_id, link)
    input_csv_path = "/Users/prathamsolanki/PycharmProjects/DSCi-560/csci566_DLproject/scraped_links/test.csv"

    # Output CSV file path
    output_csv_path = "/Users/prathamsolanki/PycharmProjects/DSCi-560/csci566_DLproject/scraped_links/final_test6.csv"

    # Open input CSV for reading and output CSV for writing
    with open(input_csv_path, "r", newline="", encoding="utf-8") as infile, \
            open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the new header
        writer.writerow(["index", "file_name", "link", "image_url", "image_path"])

        # Optionally, skip the header in the input CSV
        header = next(reader, None)

        # Process each row from input
        for row in reader:
            # Expecting input row: index, file_name, id, user_id, link
            index = row[0]
            file_name = row[1]
            tweet_id = row[2]
            user_id = row[3]
            link = row[4]
            if int(index) > 660:
                try:
                    # Get the image URL by opening the tweet page
                    image_url = get_webpage(link)  # returns "" if not found

                    # Download the image (if image_url is non-empty)
                    image_path = ""
                    if image_url != "":
                        image_path = download_image(image_url, tweet_id, user_id, folder="images2") or ""

                    # Write out the new row with all columns
                    writer.writerow([index, file_name, link, image_url, image_path])
                    print(f"Processed {index}: {file_name} - {link} -> {image_url} -> {image_path}")
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    continue
            # if int(index) > 660:
            # try:
            #     # Get the image URL by opening the tweet page
            #     image_url = get_webpage(link)  # returns "" if not found
            #
            #     # Download the image (if image_url is non-empty)
            #     image_path = ""
            #     if image_url != "":
            #         image_path = download_image(image_url, tweet_id, user_id, folder="images2") or ""
            #
            #     # Write out the new row with all columns
            #     writer.writerow([index, file_name, link, image_url, image_path])
            #     print(f"Processed {index}: {file_name} - {link} -> {image_url} -> {image_path}")
            # else:
            #     continue

    print("New CSV creation complete!")

    return None

get_multiple_images()



