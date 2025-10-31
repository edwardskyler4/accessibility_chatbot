from playwright.async_api import async_playwright
import asyncio
import re
import csv
import os

async def login_and_create_save_state(byui_email, username, password):
    """
    Logs you into your BYUI account and creates a storage state that allows a program to access sharepoint sites without needing to repeat the login process.
    Params: School email, Church Username, Church Password.
    Returns: Technically nothing, but it will create a JSON with your new save state.
    """

    try:
      os.mkdir("./.auth/")
      print("Directory './.auth/' created.")
    except FileExistsError:
      print("Directory './.auth/' already exists.")

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        context = await browser.new_context()

        page = await context.new_page()

        await page.goto("https://webmailbyui.sharepoint.com/sites/digitalaccessibilityhub/SitePages/00%20Home%20Page.aspx")
        await page.wait_for_url(re.compile("login.microsoftonline.com"))

        await page.get_by_label("userName@byui.edu").fill(byui_email)
        await page.get_by_role("button", name="Next").click()

        await page.wait_for_url(re.compile("id.churchofjesuschrist.org"))

        await page.get_by_role("textbox").fill(username)
        await page.get_by_role("button", name="Next").click()

        await page.get_by_role("textbox", name="Password").wait_for(state="visible")
        await page.get_by_role("textbox", name="Password").fill(password)
        await page.get_by_role("button", name="Verify").click()

        await page.get_by_role("button", name="No, other people use this device").wait_for(state="visible")
        await page.get_by_role("button", name="No, other people use this device").click()

        await page.wait_for_url(re.compile("webmailbyui.sharepoint.com"))

        await context.storage_state(path=".\\.auth\\state.json")
        print("Save state created.")

        await context.close()

def append_to_file(content, file):
  try:
    with open(file, "a") as f:
        f.write(content)
  
  except FileNotFoundError:
    print(f"{file} not found")

def get_urls_from_csv(filename):
    """
    Collects urls from the exported excel file (which must be converted to a csv) listing all the site pages on a sharepoint domain. Formats the urls correctly then saves them to a list
    Params: the name/location of the csv file
    Returns: a list of usable urls

    Note: to be used for other sites, change the 'base' variable to reflect the correct site
    """
    urls = []

    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader)

            base = "https://webmailbyui.sharepoint.com/"

            for row in reader:
                name = row[0]
                formatted_name = name.replace(" ", "%20")

                path = row[11]

                link = base + path + "/" + formatted_name

                urls.append(link)

    except FileNotFoundError:
        print(f"{filename} not found among your files. Make sure your spelling and path are correct.")

    return urls

async def get_content_to_txt(page, url):
    """
    Note: saves content to a JSON.
    """
    await page.goto(url)
    
    all_dropdowns = await page.get_by_role('button', name="?").all()
    all_dropdowns_length = len(all_dropdowns)
    
    if all_dropdowns_length > 0:
        for i in range(all_dropdowns_length):
            button_to_click = all_dropdowns[i]
            await button_to_click.click()

    locators = await page.locator('title, h1, h2, h3, h4, h5, h6, p').all()

    structured_content = []

    print(f"Processing {len(locators)} content elements...")

    for locator in locators:
        # A. Get the HTML tag name (e.g., 'H1', 'P', 'TITLE') using evaluate
        tag_name = await locator.evaluate('el => el.tagName')
        tag_name_lower = tag_name.lower()

        # B. Get the text content
        if tag_name_lower == 'title':
            # The TITLE tag content is best retrieved via text_content()
            text = await locator.text_content()
        else:
            # For visible elements, inner_text() is usually best
            text = await locator.inner_text()

        # # C. Assign a descriptive label
        label = tag_name_lower
        
        # Only store non-empty text content
        if text and text.strip():
            structured_content.append({
            'type': label,
            'content': text.strip()
        })

    file = "hub_content.txt"

    append_to_file("---Beginning of Page---\n\n", file)

    for line in structured_content:
        tag = line['type']
        content = line['content']
        page_content = f"{tag} - {content}\n"

        append_to_file(page_content, file)
    
    append_to_file("\n\n---End of Page---\n", file)
  
async def main():
    await login_and_create_save_state("*", "*", "*")

    save_state = "accessibility_chatbot\\.auth\\state.json"
    urls = get_urls_from_csv("accessibility_chatbot\\hub_sites.csv")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(storage_state=save_state)
        page = await context.new_page()

        count = 0
        total = len(urls)
        for url in urls:
            count += 1
            print(f"\nProgress: {count}/{total}")
            print(f"Scraping {url}")
            try:
                await get_content_to_txt(page, url)
                print("done")
            except:
                print(f"Scraping {url} failed")
    
    return "done for finally!"

if __name__ == "__main__":
    asyncio.run(main())